import itertools
import time

import math

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torchvision import models
from torchvision.ops import DeformConv2d, FrozenBatchNorm2d
from transformers import BertConfig, BertLayer

from utils.distribute import is_main_process

GOP = 4
INDEX = 0


def prepare_gaussian_targets(targets, sigma=1):
    gaussian_targets = []
    for batch_idx in range(targets.shape[0]):
        t = targets[batch_idx]
        axis = torch.arange(len(t), device=targets.device)
        gaussian_t = torch.zeros_like(t)
        indices, = torch.nonzero(t, as_tuple=True)
        for i in indices:
            g = torch.exp(-(axis - i) ** 2 / (2 * sigma * sigma))
            gaussian_t += g

        gaussian_t = gaussian_t.clamp(0, 1)
        # gaussian_t /= gaussian_t.max()
        gaussian_targets.append(gaussian_t)
    gaussian_targets = torch.stack(gaussian_targets, dim=0)
    return gaussian_targets


def cosine_compare(inputs, similarity_module, k):
    """(b c t)"""
    B = inputs.shape[0]
    L = inputs.shape[-1]

    padded_inputs = F.pad(inputs, pad=(0, math.ceil(L / k) * k - L), mode='replicate')
    # pad_L = padded_inputs.shape[-1]

    outputs = torch.zeros_like(padded_inputs)
    for offset in range(k):
        left_x = F.pad(padded_inputs, pad=(k - offset, 0), mode='replicate')[:, :, :-(k - offset)]
        right_x = F.pad(padded_inputs, pad=(0, offset + 1), mode='replicate')[:, :, (offset + 1):]
        left_seq = einops.rearrange(left_x, 'b c (k nw) -> (b nw) c k', k=k)
        right_seq = einops.rearrange(right_x, 'b c (k nw) -> (b nw) c k', k=k)

        h = similarity_module(left_seq, right_seq, padded_inputs, offset)  # (b nw) c
        hidden_state = einops.rearrange(h, '(b nw) c -> b c nw', b=B)

        outputs[:, :, offset::k] = hidden_state

    outputs = einops.rearrange(outputs[:, :, :L], 'b c t -> b t c')  # (b t c)
    return outputs


class SimilarityModule(Module):
    def __init__(self, dim, k):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=k)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=k)
        self.merge = nn.Linear(dim * 2, dim)

    def forward(self, left_seq, right_seq, x, offset):
        """
        Args:
            left_seq: (b nw) c k
            right_seq: (b nw) c k
            x: (b c t)
            offset: int
        Returns:
        """
        feats1 = self.conv1(left_seq).squeeze(-1)
        feats2 = self.conv2(right_seq).squeeze(-1)

        return self.merge(torch.cat([feats1, feats2], dim=1))


class LeftRightFeatureExtractor(Module):
    def __init__(self, dim, stride, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.left_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride)
        self.right_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride)

        # self.similarity_module = SimilarityModule(dim, kernel_size)

    def forward(self, x):
        """x: (b c t)"""
        left_feats = self.left_conv(F.pad(x, pad=(self.kernel_size, 0), mode='replicate')[:, :, :-1])
        right_feats = self.right_conv(F.pad(x, pad=(0, self.kernel_size), mode='replicate')[:, :, 1:])

        feats = torch.cat([left_feats, right_feats], dim=1)  # (b c t)
        # feats = self.bn1(left_feats) + self.bn2(right_feats)

        # feats = cosine_compare(x, self.similarity_module, self.kernel_size)  # (b t c)

        return feats


class FPN(Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        self.layer_block = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.inner_blocks = nn.ModuleList()
        for c in in_channels:
            self.inner_blocks.append(nn.Conv2d(c, dim, 1), )

    def forward(self, features_list):
        target_idx = 1
        result_feature = self.inner_blocks[target_idx](features_list[target_idx])
        for idx, feature in enumerate(features_list):
            if idx != target_idx:
                feature = self.inner_blocks[idx](feature)
                feature = F.interpolate(feature, size=result_feature.shape[-2:], mode='bilinear', align_corners=False)
                result_feature += feature

        result_feature = self.layer_block(result_feature)
        return result_feature


class Backbone(Module):
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()
        self.cfg = cfg
        self.backbone_name = cfg.MODEL.BACKBONE.NAME

        if self.backbone_name == 'mobile_vit':
            from .ic_automl_mobile_cpu_vit_cls import get_IC_AutoML_Mobile_CPU_ViT
            flops = 20.4
            user_name = "licongcong.lufficc@bytedance.com"  # modify user_name to your email address
            resume = None
            if cfg.MODEL.PRETRAINED:
                resume = 'ic_automl_mobile_cpu_vit_cls_20.4M.pth.tar'
            model = get_IC_AutoML_Mobile_CPU_ViT(flops=flops,
                                                 user_name=user_name,
                                                 resume=resume,
                                                 ff_dropout=0.0,
                                                 attention_dropout=0.0,
                                                 path_dropout=0.0,
                                                 )
            self.out_features = model.stage_out_channels[-1]
            del model.feature_mix_layer
            del model.output
            self.backbone = model
        else:
            kwargs = {'pretrained': cfg.MODEL.PRETRAINED}
            if 'resnet' in self.backbone_name:
                kwargs['norm_layer'] = FrozenBatchNorm2d
            self.backbone = getattr(models, self.backbone_name)(**kwargs)

        if 'mobilenet' in self.backbone_name:
            if in_dim != 3:
                self.backbone.features[0][0] = nn.Conv2d(in_dim, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.out_features = self.backbone.classifier[-1].in_features
            del self.backbone.classifier

        elif 'shufflenet' in self.backbone_name:
            if in_dim != 3:
                self.backbone.conv1[0] = nn.Conv2d(in_dim, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.out_features = self.backbone.fc.in_features
            del self.backbone.fc

        elif 'resnet' in self.backbone_name:  # resnet
            self.out_features = self.backbone.fc.in_features
            if in_dim != 3:
                setattr(self.backbone, 'conv1', nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
                self.bn = nn.BatchNorm2d(in_dim)
            else:
                self.bn = nn.Identity()
            for param in itertools.chain(self.backbone.conv1.parameters(), self.backbone.bn1.parameters()):
                param.requires_grad = False

            del self.backbone.fc

        self.embedding = nn.Conv2d(self.out_features, out_dim, 3, 1, 1)

    def forward(self, x):
        if self.backbone_name == 'mobile_vit':
            x = self.backbone(x)
        elif 'mobilenet' in self.backbone_name:
            x = self.backbone.features(x)
        elif 'shufflenet' in self.backbone_name:
            x = self.backbone.conv1(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.stage2(x)
            x = self.backbone.stage3(x)
            x = self.backbone.stage4(x)
            x = self.backbone.conv5(x)
        elif 'resnet' in self.backbone_name:
            x = self.bn(x)
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)  # 64, 56, 56
            x = self.backbone.layer2(x)  # 128, 28, 28
            x = self.backbone.layer3(x)  # 256, 14, 14
            x = self.backbone.layer4(x)  # 512, 7, 7

        outputs = self.embedding(x)
        return outputs


class EstimatorDenseNetTiny(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(EstimatorDenseNetTiny, self).__init__()

        def Conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
                nn.LeakyReLU(0.1)
            )

        self.conv0 = Conv2D(ch_in, 8, kernel_size=3, stride=1)
        dd = 8
        self.conv1 = Conv2D(ch_in + dd, 8, kernel_size=3, stride=1)
        dd += 8
        self.conv2 = Conv2D(ch_in + dd, 6, kernel_size=3, stride=1)
        dd += 6
        self.conv3 = Conv2D(ch_in + dd, 4, kernel_size=3, stride=1)
        dd += 4
        self.conv4 = Conv2D(ch_in + dd, 2, kernel_size=3, stride=1)
        dd += 2
        self.predict_flow = nn.Conv2d(ch_in + dd, ch_out, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = torch.cat((self.conv0(x), x), 1)
        x = torch.cat((self.conv1(x), x), 1)
        x = torch.cat((self.conv2(x), x), 1)
        x = torch.cat((self.conv3(x), x), 1)
        x = torch.cat((self.conv4(x), x), 1)
        return self.predict_flow(x)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=8):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim)  # (T, C)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, C)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class GroupSimilarity(nn.Module):
    def __init__(self, cfg, dim, window_size, group=4, similarity_func='cosine', offset=0):
        super(GroupSimilarity, self).__init__()
        self.cfg = cfg
        self.out_channels = dim * 1
        self.group = group
        self.similarity_func = similarity_func
        self.offset = offset
        self.temporal_model = cfg.MODEL.TEMPORAL_MODEL
        self.use_group_similarity = cfg.MODEL.USE_GROUP_SIMILARITY

        num_layers = 6
        k = 5
        padding = (k - 1) // 2

        self.position_encoding = PositionalEncoding(dim, max_len=window_size * 2 + 1)
        if self.temporal_model == 'transformer':
            encoder_config = BertConfig(
                hidden_size=dim,
                num_attention_heads=8,
                intermediate_size=dim * 4,
            )
            self.encoders = nn.ModuleList([BertLayer(encoder_config) for _ in range(num_layers)])
        elif self.temporal_model == 'gru':
            self.encoders = nn.GRU(dim, dim, batch_first=True, num_layers=2)
        else:
            raise NotImplemented

        if self.use_group_similarity:
            self.fcn = nn.Sequential(
                BasicConv2d(self.group, dim, kernel_size=k, stride=1, padding=padding),
                BasicConv2d(dim, dim, kernel_size=k, stride=1, padding=padding),
                BasicConv2d(dim, dim, kernel_size=k, stride=1, padding=padding),
                BasicConv2d(dim, dim, kernel_size=k, stride=1, padding=padding),
            )
            # self.fcn = nn.Sequential(
            #     BasicConv2d(self.group, dim, kernel_size=k, stride=1, padding=padding),
            #     BasicConv2d(dim, dim, kernel_size=k, stride=1, padding=padding, groups=dim),
            #     BasicConv2d(dim, dim, kernel_size=1),
            #     BasicConv2d(dim, dim, kernel_size=k, stride=1, padding=padding, groups=dim),
            #     BasicConv2d(dim, dim, kernel_size=1),
            #     BasicConv2d(dim, dim, kernel_size=k, stride=1, padding=padding, groups=dim),
            #     BasicConv2d(dim, dim, kernel_size=1),
            # )
        else:
            self.lin = nn.Linear(dim * 2, dim) if self.temporal_model == 'gru' else nn.Identity()

    def get_similarity_map(self, x):
        B, L, C = x.shape
        x = x.view(B, L, self.group, C // self.group)  # (B, L, G, C')
        # (B, L, L, H)
        similarity_func = self.similarity_func

        if similarity_func == 'cosine':
            sim = F.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)  # batch, T, T, G
        else:
            raise NotImplemented

        sim = sim.permute(0, 3, 1, 2)  # batch, G, T, T
        return sim

    def recognize_patterns(self, left_seq, mid_seq, right_seq, offset=0):
        k = left_seq.shape[1]
        assert k > offset

        left_seq = left_seq[:, offset:]
        right_seq = right_seq[:, :(None if offset == 0 else -offset)]
        assert left_seq.shape[1] == right_seq.shape[1] == (k - offset)
        x = torch.cat([left_seq, mid_seq, right_seq], dim=1)
        x = self.position_encoding(x)

        if self.temporal_model == 'transformer':
            for encoder in self.encoders:
                x = encoder(x)[0]  # (B, L, C)
            h_n = x[:, -1]
        elif self.temporal_model == 'gru':
            x, h_n = self.encoders(x)  # (N, L, H_in), (D∗num_layers, N, H_out)
            h_n = h_n.permute(1, 0, 2).flatten(1)
        else:
            raise NotImplemented

        if self.use_group_similarity:
            sim = self.get_similarity_map(x)
            h = self.fcn(sim)  # batch, dim, T, T
            h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        else:
            h = self.lin(h_n)
        return h

    def forward(self, left_seq, mid_seq, right_seq):
        """
        left_seq = batch, T, dim
        mid_seq = batch, 1, dim
        right_seq = batch, T, dim
        """
        h = self.recognize_patterns(left_seq, mid_seq, right_seq, offset=self.offset)

        return h


def SPoS(inputs, ban, k):
    """(b c t)"""
    B = inputs.shape[0]
    L = inputs.shape[-1]
    C = inputs.shape[1]

    padded_inputs = F.pad(inputs, pad=(0, math.ceil(L / k) * k - L), mode='replicate')
    pad_L = padded_inputs.shape[-1]

    # outputs = torch.zeros_like(padded_inputs)
    outputs = torch.zeros(B, ban.out_channels, pad_L, dtype=inputs.dtype, device=inputs.device)
    for offset in range(k):
        left_x = F.pad(padded_inputs, pad=(k - offset, 0), mode='replicate')[:, :, :-(k - offset)]
        right_x = F.pad(padded_inputs, pad=(0, offset + 1), mode='replicate')[:, :, (offset + 1):]
        left_seq = einops.rearrange(left_x, 'b c (nw k) -> (b nw) k c', k=k)
        right_seq = einops.rearrange(right_x, 'b c (nw k) -> (b nw) k c', k=k)
        mid_seq = einops.rearrange(padded_inputs[:, :, offset::k], 'b c nw -> (b nw) 1 c')

        h = ban(left_seq, mid_seq, right_seq)  # (b nw) c
        hidden_state = einops.rearrange(h, '(b nw) c -> b c nw', b=B)
        outputs[:, :, offset::k] = hidden_state

    outputs = outputs[:, :, :L]  # (b c t)
    return outputs


class EstimatorDenseNetTiny2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        def Conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
                nn.LeakyReLU(0.1)
            )

        self.conv0 = Conv2D(ch_in, 8, kernel_size=3, stride=1)
        dd = 8
        self.conv1 = Conv2D(ch_in + dd, 8, kernel_size=3, stride=1)
        dd += 8
        self.conv2 = Conv2D(ch_in + dd, 6, kernel_size=3, stride=1)
        dd += 6
        self.conv3 = Conv2D(ch_in + dd, 4, kernel_size=3, stride=1)
        dd += 4
        self.conv4 = Conv2D(ch_in + dd, 2, kernel_size=3, stride=1)
        dd += 2
        self.predict_flow = nn.Conv2d(ch_in + dd, ch_out, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = torch.cat((self.conv0(x), x), 1)
        x = torch.cat((self.conv1(x), x), 1)
        x = torch.cat((self.conv2(x), x), 1)
        x = torch.cat((self.conv3(x), x), 1)
        x = torch.cat((self.conv4(x), x), 1)
        return self.predict_flow(x)


class CBAM(Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 16),
            nn.ReLU(),
            nn.Linear(dim // 16, dim)
        )
        self.spatial_module = EstimatorDenseNetTiny(dim + dim * 1, 1)
        self.channel_module = EstimatorDenseNetTiny(dim + dim * 1, dim)

    def forward(self, i_feats, mv_feats):
        """
        Args:
            i_feats:  N, C, H, W
            mv_feats: N, C, H, W
        Returns:
        """
        channel_weight = self.channel_module(torch.cat([i_feats, mv_feats], dim=1))
        channel_att = self.mlp(F.adaptive_avg_pool2d(channel_weight, 1).flatten(1))
        i_feats = i_feats * channel_att.sigmoid().unsqueeze(-1).unsqueeze(-1)

        spatial_weight = self.spatial_module(torch.cat([i_feats, mv_feats], dim=1))
        spatial_weight = F.softmax(spatial_weight.view(*spatial_weight.shape[:2], -1), dim=-1).view_as(spatial_weight)
        i_feats = (i_feats * spatial_weight).sum(dim=(2, 3))  # (bn gop) c
        p_features = i_feats + F.adaptive_avg_pool2d(mv_feats, 1).flatten(1)  # (bn gop) c
        return p_features


class E2ECompressedGEBDModelBeats(Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.MODEL.DIM

        self.kernel_size = cfg.MODEL.KERNEL_SIZE
        self.dim = dim
        self.backbone = Backbone(cfg, 3, dim)

        self.temporal_module = GroupSimilarity(cfg, dim=dim,
                                               window_size=self.kernel_size,
                                               group=4,
                                               similarity_func='cosine')

        self.classifier = nn.Sequential(
            nn.Conv1d(self.temporal_module.out_channels, dim, 1, 1, 0),
            nn.PReLU(),
            nn.Conv1d(dim, dim, 1, 1, 0),
            nn.PReLU(),
            nn.Conv1d(dim, 1, 1)
        )

        # self.classifier = nn.Sequential(
        #     nn.Conv2d(self.temporal_module.out_channels, dim, 1, 1, 0),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(dim, dim, 1, 1, 0),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(dim, 1, 1)
        # )

    def extract_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, inputs, targets=None):
        """
        Args:
            inputs(dict): imgs (B, T, C, H, W);
            targets:
        Returns:
        """
        imgs = inputs['imgs']  # (4, 100, 3, 224, 224)
        if 'frame_mask' in inputs:
            frame_mask = inputs['frame_mask']  # (4, 100)
        else:
            frame_mask = torch.ones(*imgs.shape[:2], dtype=torch.uint8, device=imgs.device)
        B = imgs.shape[0]
        time_cost = {}
        start = time.perf_counter()
        i_imgs = imgs.view(-1, *imgs.shape[-3:])
        i_features = self.extract_features(i_imgs)
        feats = F.adaptive_avg_pool2d(i_features, 1).flatten(1)
        time_cost['backbone'] = time.perf_counter() - start

        start = time.perf_counter()
        feats = einops.rearrange(feats, '(b t) c -> b c t', b=B)  # (4, 512, 100)
        feats = SPoS(feats, self.temporal_module, self.kernel_size)  # b c t
        logits = self.classifier(feats)  # b 1 t
        # logits = self.classifier(feats.unsqueeze(-1)).squeeze(-1)  # b 1 t

        if self.training:
            targets = targets.to(logits.dtype)
            # gaussian_targets = prepare_gaussian_targets(targets)
            gaussian_targets = targets
            frame_mask = frame_mask.view(-1) == 1

            loss = F.binary_cross_entropy_with_logits(logits.view(-1)[frame_mask], gaussian_targets.view(-1)[frame_mask])
            loss_dict = {'loss': loss}

            return loss_dict
        scores = torch.sigmoid(logits).flatten(1)
        time_cost['head'] = time.perf_counter() - start
        return scores, time_cost
