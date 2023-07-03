import itertools
import math

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torchvision import models
from torchvision.ops import DeformConv2d, FrozenBatchNorm2d

from utils.distribute import is_main_process

GOP = 4


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


# class FPN(Module):
#     def __init__(self, in_channels, dim):
#         super().__init__()
#         self.layer_blocks = nn.ModuleList()
#         self.inner_blocks = nn.ModuleList()
#         for c in in_channels:
#             self.layer_blocks.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
#             self.inner_blocks.append(nn.Conv2d(c, dim, 1), )
#
#     def forward(self, features_list):
#         last_inner = self.inner_blocks[-1](features_list[-1])
#         outputs = [self.layer_blocks[-1](last_inner)]
#         for i in range(len(features_list) - 2, -1, -1):
#             inner_lateral = self.inner_blocks[i](features_list[i])
#             top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode='bilinear', align_corners=False)
#             last_inner = inner_lateral + top_down
#             outputs.insert(0, self.layer_blocks[i](last_inner))
#
#         return outputs


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


class SidedataModel(Module):
    def __init__(self, cfg, dim, mode='res'):
        super().__init__()
        assert mode in ['res', 'mv', 'rgb']
        assert cfg.INPUT.USE_SIDE_DATA
        self.backbone = getattr(models, cfg.MODEL.BACKBONE.SIDE_DATA_NAME)(pretrained=True, norm_layer=FrozenBatchNorm2d)
        self.out_features = self.backbone.fc.in_features
        del self.backbone.fc

        if mode == 'mv':
            setattr(self.backbone, 'conv1', nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
            self.bn = nn.BatchNorm2d(2)
            # self.bn = nn.Identity()
        elif mode == 'res':
            self.bn = nn.BatchNorm2d(3)
            # self.bn = nn.Identity()
        elif mode == 'rgb':
            self.bn = nn.Identity()

        # self.fpn = FPN([64, 128, 256, 512], dim)
        self.embedding = nn.Conv2d(512, dim, 3, 1, 1)

    def extract_features(self, x):
        x = self.bn(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        outputs = []
        x = self.backbone.layer1(x)  # 64, 56, 56
        outputs.append(x)
        x = self.backbone.layer2(x)  # 128, 28, 28
        outputs.append(x)
        x = self.backbone.layer3(x)  # 256, 14, 14
        outputs.append(x)
        x = self.backbone.layer4(x)  # 512, 7, 7
        outputs.append(x)

        # outputs = self.fpn(outputs)[1]
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


# class EstimatorDenseNetTiny(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(EstimatorDenseNetTiny, self).__init__()
#
#         def Conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
#             return nn.Sequential(
#                 nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
#                 # nn.BatchNorm2d(out_planes),
#                 nn.ReLU(),
#             )
#
#         self.convs = nn.Sequential(
#             Conv2D(ch_in, 256, kernel_size=3, stride=1),
#             Conv2D(256, 256, kernel_size=3, stride=1),
#             Conv2D(256, 256, kernel_size=3, stride=1),
#             Conv2D(256, 256, kernel_size=3, stride=1),
#         )
#
#         self.predict_flow = nn.Conv2d(256, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
#
#     def forward(self, x):
#         x = self.convs(x)
#         return self.predict_flow(x)


# class UpsampleUpdatingModel2(Module):
#     def __init__(self, cfg, dim, mode='mv'):
#         super().__init__()
#         self._use_gan = cfg.MODEL.USE_GAN
#         self.backbone = SidedataModel(cfg, dim, mode=mode)
#         self.weight_predictor = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.LeakyReLU(),
#             nn.Linear(dim, dim),
#             nn.Sigmoid(),
#         )
#         in_dim = {
#             'mv': 2, 'res': 3
#         }[mode]
#         self.offset_predictor = EstimatorDenseNetTiny(in_dim + dim, dim)
#
#         # self.motion_convs = nn.Sequential(
#         #     nn.Conv2d(dim, dim, kernel_size=3, padding=1),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(dim, dim, kernel_size=3, padding=1),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(dim, dim, kernel_size=3, padding=1),
#         #     nn.ReLU(inplace=True)
#         # )
#
#     def forward(self, imgs, i_features, p_motions):
#         """
#         Args:
#             imgs: (4, 100, 3, 224, 224)
#             i_features: (100, 256, 7, 7)
#             p_motions: (100, 3, 2, 224, 224)
#         Returns:
#         """
#         B = imgs.shape[0]
#         num_gop = imgs.shape[1] // GOP
#
#         p_motions = einops.rearrange(p_motions, 'bn gop c h w -> (bn gop) c h w')
#         p_features = self.backbone.extract_features(p_motions)
#
#         p_motions_resized = F.interpolate(p_motions, size=p_features.shape[-2:], mode='bilinear', align_corners=False)
#         weight_feats = self.offset_predictor(torch.cat([p_motions_resized, p_features], dim=1))
#         weight = self.weight_predictor(F.adaptive_avg_pool2d(weight_feats, 1).flatten(1))  # (300, 256)
#
#         i_features = einops.rearrange(i_features, '(b n) c h w -> b n c h w', b=B)  # (4, 25, 256, 7, 7)
#         weight = einops.rearrange(weight, '(b n t) c -> b n t c', b=B, n=num_gop)  # (4, 25, 3, 256)
#
#         i_features_weighted = i_features.unsqueeze(2) * weight.unsqueeze(-1).unsqueeze(-1)  # (4, 25, 1->3, 256, 7, 7)
#
#         p_features_expand = einops.rearrange(p_features, '(b n k) c h w -> b n k c h w', b=B, n=num_gop)
#         p_features = i_features_weighted + p_features_expand
#
#         p_features = F.adaptive_avg_pool2d(p_features, 1).flatten(3)  # b n k c
#
#         return p_features


class UpsampleUpdatingModel2(Module):
    def __init__(self, cfg, dim, mode='mv'):
        super().__init__()
        self.mode = mode
        self._use_gan = cfg.MODEL.USE_GAN
        self.backbone = SidedataModel(cfg, dim, mode=mode)
        self.channel_weight_predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        in_dim = {
            'mv': 2, 'res': 3
        }[mode]

        self.spatial_module = EstimatorDenseNetTiny(in_dim + dim * 2, 1)
        self.channel_module = EstimatorDenseNetTiny(in_dim + dim * 2, dim)

        # self.motion_convs = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, imgs, i_features, p_motions):
        """
        Args:
            imgs: (4, 100, 3, 224, 224)
            i_features: (100, 256, 7, 7)
            p_motions: (100, 3, 2, 224, 224)
        Returns:
        """
        B = imgs.shape[0]
        num_gop = imgs.shape[1] // GOP
        i_features_o = i_features = i_features.unsqueeze(1).expand(-1, GOP - 1, -1, -1, -1).reshape(-1, *i_features.shape[-3:])  # (bn gop) c h w

        p_motions = einops.rearrange(p_motions, 'bn gop c h w -> (bn gop) c h w')
        p_features = self.backbone.extract_features(p_motions)

        p_motions_resized = F.interpolate(p_motions, size=p_features.shape[-2:], mode='bilinear', align_corners=False)

        channel_weight = self.channel_module(torch.cat([p_motions_resized, p_features, i_features], dim=1))
        weight = self.channel_weight_predictor(F.adaptive_avg_pool2d(channel_weight, 1).flatten(1))  # (300, 256)
        # weight = self.channel_weight_predictor(F.adaptive_max_pool2d(channel_weight, 1).flatten(1))  # (300, 256)

        i_features = i_features * weight.unsqueeze(-1).unsqueeze(-1)  # (bn gop) c h w

        spatial_weight = self.spatial_module(torch.cat([p_motions_resized, p_features, i_features], dim=1))
        spatial_weight = F.softmax(spatial_weight.view(*spatial_weight.shape[:2], -1), dim=-1).view_as(spatial_weight)
        i_features = (i_features * spatial_weight).sum(dim=(2, 3))  # (bn gop) c

        p_features = i_features + F.adaptive_avg_pool2d(p_features, 1).flatten(1)  # (bn gop) c
        p_features = einops.rearrange(p_features, '(b n t) c -> b n t c', b=B, n=num_gop)  # b n k c

        return p_features


class TemporalModule(nn.Module):
    def __init__(self, cfg, dim, kernel_size=8, out_dim=None):
        super().__init__()
        k_size = 8
        print(f'k={k_size}')
        # self.extractor = LeftRightFeatureExtractor(dim, stride=1, kernel_size=kernel_size)
        self.extractors = nn.ModuleList([
            LeftRightFeatureExtractor(dim, stride=1, kernel_size=k) for k in [k_size]
        ])

        self.out = None
        if out_dim is not None and out_dim != dim * 2:
            self.out = nn.Linear(dim * 2, out_dim)

    def forward(self, feats):
        """
        Args:
            feats: (b c t)
        """

        feats_list = []
        for extractor in self.extractors:
            feats_list.append(extractor(feats))
        feats = sum(feats_list) / len(feats_list)  # b c t
        # feats = self.extractor(feats)  # b c t
        if self.out is not None:
            feats = self.out(feats)
        return feats


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, ref_tensor):
        """
        (b, c, t)
        """
        seq_length = ref_tensor.shape[-1]
        x_embed = torch.arange(seq_length, dtype=torch.float32, device=ref_tensor.device).view(1, seq_length)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=ref_tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos_x


class E2ECompressedGEBDModel(Module):
    def __init__(self, cfg):
        super().__init__()
        # assert cfg.INPUT.USE_SIDE_DATA
        self._use_gan = cfg.MODEL.USE_GAN
        self._use_residual = cfg.MODEL.USE_RESIDUAL
        self._use_mv_as_deconv_params = cfg.MODEL.USE_MV_AS_DECONV_PARAMS

        if is_main_process():
            print('USE_GAN:', self._use_gan)
            print('USE_RESIDUAL:', self._use_residual)
            print('USE_MV_AS_DECONV_PARAMS:', self._use_mv_as_deconv_params)

        self.backbone_name = cfg.MODEL.BACKBONE.NAME
        if self.backbone_name == 'csn':
            from .backbone import CSN
            self.backbone = CSN()
            in_feat_dim = 2048
        elif self.backbone_name == 'tsn':
            from .backbone import TSN
            self.backbone = TSN()
            in_feat_dim = 2048
        else:
            self.backbone = getattr(models, cfg.MODEL.BACKBONE.NAME)(pretrained=True, norm_layer=FrozenBatchNorm2d)
            for param in itertools.chain(self.backbone.conv1.parameters(), self.backbone.bn1.parameters()):
                param.requires_grad = False
            in_feat_dim = self.backbone.fc.in_features
            del self.backbone.fc

        self.kernel_size = 8
        dim = 256

        # if self._use_residual:
        #     self.res_backbone = SidedataModel(cfg, dim, mode='res')
        # self.trans_res_embedding = nn.Conv2d(self.res_backbone.out_features, dim, kernel_size=1)

        if self._use_mv_as_deconv_params:
            self.mv_module = UpsampleUpdatingModel2(cfg, dim, mode='mv')
            self.res_module = UpsampleUpdatingModel2(cfg, dim, mode='res')
        else:
            self.mv_backbone = SidedataModel(cfg, dim, mode='mv')
            # self.mv_backbone = SidedataModel(cfg, dim, mode='rgb')
            # self.trans_mv_embedding = nn.Conv2d(self.mv_backbone.out_features, dim, kernel_size=1)

        self.temporal_module = TemporalModule(cfg, dim, kernel_size=self.kernel_size)

        self.output = nn.Sequential(
            nn.Conv1d(dim * 2, dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.PReLU(),
        )

        self.classifier = nn.Linear(
            in_features=dim,
            out_features=1,
        )

        # FPN
        # self.fpn = FPN([256, 512, 1024, 2048], dim)
        self.embedding = nn.Conv2d(2048, dim, 3, 1, 1)
        # self.pe = PositionEmbeddingSine(dim, normalize=True)

    def extract_features(self, x):
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        outputs = []
        x = self.backbone.layer1(x)  # (256 * 56 * 56)
        outputs.append(x)
        x = self.backbone.layer2(x)  # (512 * 28 * 28)
        outputs.append(x)
        x = self.backbone.layer3(x)  # (1024 * 14 * 14)
        outputs.append(x)
        x = self.backbone.layer4(x)  # (2048 * 7 * 7)
        outputs.append(x)

        # outputs = self.fpn(outputs)[1]
        outputs = self.embedding(x)
        return outputs

    def forward(self, inputs, targets=None):
        """
        Args:
            inputs(dict): imgs (B, T, C, H, W);
            targets:
        Returns:
        """
        imgs = inputs['imgs']  # (4, 100, 3, 224, 224)
        mv = inputs['mv']  # (4, 100, 2, 224, 224)
        res = inputs['res']  # (4, 100, 3, 224, 224)
        frame_mask = inputs['frame_mask']  # (4, 100)

        B = imgs.shape[0]

        i_imgs = imgs[:, ::GOP]  # (4, 8, 3, 224, 224)
        num_gop = i_imgs.shape[1]

        if self._use_gan and self.training:
            p_frame_mask = einops.rearrange(frame_mask, 'b (n gop) -> (b n) gop', gop=GOP)
            p_frame_mask = p_frame_mask[:, 1:]

            tmp_imgs = einops.rearrange(imgs, 'b (n gop) c h w -> (b n) gop c h w', gop=GOP)  # (100, 4, 3, 224, 224)
            p_imgs = tmp_imgs[:, 1:]

        p_motions = einops.rearrange(mv, 'b (n gop) c h w -> (b n) gop c h w', gop=GOP)  # (32, 12, 2, 224, 224)
        p_motions = p_motions[:, 1:]  # (32, 11, 2, 224, 224)

        if self.backbone_name in ['csn', 'tsn']:
            x = self.backbone(i_imgs)
            x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
            i_features = self.embedding(x)
        else:
            i_features = self.extract_features(i_imgs)  # (32, 2048, 7, 7)

        if self._use_mv_as_deconv_params:
            p_features = self.mv_module(imgs, i_features, p_motions)
            if self._use_residual:
                p_res = einops.rearrange(res, 'b (n gop) c h w -> (b n) gop c h w', gop=GOP)  # (32, 12, 3, 224, 224)
                p_res = p_res[:, 1:]  # (32, 11, 3, 224, 224)
                p_features += self.res_module(imgs, i_features, p_res)
            # p_res = einops.rearrange(res, 'b (n gop) c h w -> (b n) gop c h w', gop=GOP)  # (32, 12, 3, 224, 224)
            # p_res = p_res[:, 1:]  # (32, 11, 3, 224, 224)
            # p_features = self.res_module(imgs, i_features, p_res)
        else:
            p_features = self.mv_backbone.extract_features(einops.rearrange(p_motions, 'bn gop c h w -> (bn gop) c h w'))
            # p_features = self.trans_mv_embedding(p_features)
            p_features = einops.rearrange(p_features, '(b n gop) c h w -> b n gop c h w', b=B, n=num_gop)  # (4, 25, 3, 512, 7, 7)
            p_features = F.adaptive_avg_pool2d(p_features, 1).flatten(3)  # (4, 25, 3, 512)

        i_features = einops.rearrange(i_features, '(b n) c h w -> b n c h w', b=B)  # (4, 8, 512, 7, 7)
        i_features = F.adaptive_avg_pool2d(i_features, 1).flatten(2)

        feats = torch.zeros(B, num_gop, GOP, i_features.shape[-1], dtype=i_features.dtype, device=i_features.device)  # (4, 8, 3, c, h, w)
        feats[:, :, 0] = i_features
        feats[:, :, 1:] = p_features

        # feats = self.extract_features(einops.rearrange(imgs, 'b t c h w -> (b t) c h w'))  # (32, 2048, 7, 7)
        # feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        # feats = einops.rearrange(feats, '(b t) c -> b c t', b=B)

        feats = einops.rearrange(feats, 'b n gop c -> b (n gop) c')
        feats = einops.rearrange(feats, 'b t c -> b c t', b=B)  # (4, 512, 100)

        feats = self.temporal_module(feats)  # b c t

        feats = self.output(feats)
        logits = self.classifier(feats.permute(0, 2, 1))  # b t 1

        if self.training:
            targets = targets.to(logits.dtype)
            gaussian_targets = prepare_gaussian_targets(targets)
            frame_mask = frame_mask.view(-1) == 1

            loss = F.binary_cross_entropy_with_logits(logits.view(-1)[frame_mask], gaussian_targets.view(-1)[frame_mask])
            loss_dict = {'loss': loss}

            return loss_dict
        scores = torch.sigmoid(logits)[:, :, 0]

        return scores
