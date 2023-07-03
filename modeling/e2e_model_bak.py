import math

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torchvision import models
from torchvision.ops import sigmoid_focal_loss
from transformers import BertConfig, BertLayer

from modeling.detr.detr import build_detr

# class DETRModel(Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.model, self.criterion = build_detr(cfg)
#
#     def forward(self, inputs, targets=None):
#         """
#         Args:
#             inputs(dict): imgs (B, T, C, H, W);
#             targets:
#         Returns:
#         """
#         x = inputs['imgs']
#         outputs = self.model(x)
#
#         if self.training:
#             loss_dict = self.criterion(outputs, targets)
#             return loss_dict
#         outputs['pred_logits'] = F.softmax(outputs['pred_logits'], dim=-1)
#         return outputs
from utils.distribute import is_main_process


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


def temporal_features(inputs, temporal_model, k, mode='left'):
    """(b c t)"""
    B = inputs.shape[0]
    L = inputs.shape[-1]

    padded_inputs = F.pad(inputs, pad=(0, math.ceil(L / k) * k - L), mode='replicate')
    # pad_L = padded_inputs.shape[-1]

    outputs = torch.zeros_like(padded_inputs)
    for offset in range(k):
        if mode == 'left':
            x = F.pad(padded_inputs, pad=(k - offset, 0), mode='replicate')[:, :, :-(k - offset)]
        elif mode == 'right':
            x = F.pad(padded_inputs, pad=(0, offset + 1), mode='replicate')[:, :, (offset + 1):]
        else:
            raise NotImplementedError

        # print(list(range(pad_L))[offset::k])
        seq = einops.rearrange(x, 'b c (k nw) -> (b nw) k c', k=k)
        h = temporal_model(seq)
        hidden_state = einops.rearrange(h, '(b nw) c -> b c nw', b=B)

        outputs[:, :, offset::k] = hidden_state

    outputs = einops.rearrange(outputs[:, :, :L], 'b c t -> b t c')  # (b t c)
    return outputs


def cosine_compare(inputs, temporal_model, k):
    """(b c t)"""
    B = inputs.shape[0]
    L = inputs.shape[-1]

    padded_inputs = F.pad(inputs, pad=(0, math.ceil(L / k) * k - L), mode='replicate')
    # pad_L = padded_inputs.shape[-1]

    outputs = torch.zeros_like(padded_inputs)
    for offset in range(k):
        left_x = F.pad(padded_inputs, pad=(k - offset, 0), mode='replicate')[:, :, :-(k - offset)]
        right_x = F.pad(padded_inputs, pad=(0, offset + 1), mode='replicate')[:, :, (offset + 1):]
        left_seq = einops.rearrange(left_x, 'b c (k nw) -> (b nw) k c', k=k)
        right_seq = einops.rearrange(right_x, 'b c (k nw) -> (b nw) k c', k=k)

        # (b nw) 1 k c -> (b nw) c 1 c
        left_h = temporal_model(left_seq.unsqueeze(1)).squeeze(2)  # (b nw) c c
        right_h = temporal_model(right_seq.unsqueeze(1)).squeeze(2)  # (b nw) c c
        h = F.cosine_similarity(left_h, right_h, dim=2)  # (b nw) c
        hidden_state = einops.rearrange(h, '(b nw) c -> b c nw', b=B)

        outputs[:, :, offset::k] = hidden_state

    outputs = einops.rearrange(outputs[:, :, :L], 'b c t -> b t c')  # (b t c)
    return outputs


class LSTM(Module):
    def __init__(self, dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, batch_first=True)

    def forward(self, x):
        """b t c"""
        _, (h_n, c_n) = self.lstm(x)
        h_n = h_n[0]
        return h_n


class SelfAttention(Module):
    def __init__(self, dim):
        super().__init__()
        encoder_config = BertConfig(
            hidden_size=dim,
            num_attention_heads=8,
            intermediate_size=2048,
        )
        self.encoders = nn.ModuleList([BertLayer(encoder_config) for _ in range(6)])

    def forward(self, x):
        """b t c"""
        for encoder in self.encoders:
            x = encoder(x)[0]
        return x[:, 0]


class LeftRightFeatureExtractor(Module):
    def __init__(self, dim, stride, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.left_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, bias=False)
        self.right_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, bias=False)

    def forward(self, x):
        """x: (b c t)"""
        left_feats = self.left_conv(F.pad(x, pad=(self.kernel_size, 0), mode='replicate')[:, :, :-1])
        right_feats = self.right_conv(F.pad(x, pad=(0, self.kernel_size), mode='replicate')[:, :, 1:])

        feats = torch.cat([left_feats, right_feats], dim=1)  # (b c t)
        feats = einops.rearrange(feats, 'b c t -> b t c')  # (b t c)
        return feats


class E2EModel(Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone_name = cfg.MODEL.BACKBONE.NAME

        if self.backbone_name == 'csn':
            from .backbone import CSN
            self.backbone = CSN()
            in_feat_dim = 2048
        else:
            self.backbone = getattr(models, self.backbone_name)(pretrained=True)
            in_feat_dim = self.backbone.fc.in_features
            del self.backbone.fc

        dim = 512
        self.kernel_size = 8
        self.trans_imgs_embedding = nn.Sequential(
            nn.Linear(in_feat_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # kernel_size = 12
        # stride = 1
        # self.left_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, bias=False)
        # self.right_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=stride, bias=False)
        # self.kernel_size = kernel_size

        # self.extractor0 = LeftRightFeatureExtractor(dim, stride=1, kernel_size=12)
        # self.extractor1 = LeftRightFeatureExtractor(dim, stride=1, kernel_size=9)
        # self.extractor2 = LeftRightFeatureExtractor(dim, stride=1, kernel_size=6)

        self.extractor = LeftRightFeatureExtractor(dim, stride=1, kernel_size=self.kernel_size)

        # self.windows_weight = nn.Sequential(
        #     nn.Linear(dim * 2, dim),
        #     nn.ReLU(),
        #     nn.Linear(dim, 3),
        # )

        self.temporal_embedding = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )

        # self.left_temporal_model = LSTM(dim)
        # self.right_temporal_model = LSTM(dim)
        self.left_temporal_model = SelfAttention(dim)
        self.right_temporal_model = SelfAttention(dim)
        # self.temporal_model = nn.Conv2d(1, dim, kernel_size=(self.kernel_size, 1))

        self.output = nn.Sequential(
            nn.Linear(dim * 2 + dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim * 2),
            nn.Dropout(0.2),
            nn.LayerNorm(dim * 2)
        )

        self.classifier = nn.Linear(
            in_features=dim * 2,
            out_features=1,
        )

    def extract_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(self, inputs, targets=None):
        """
        Args:
            inputs(dict): imgs (B, T, C, H, W);
            targets: (B, T);
        Returns:
        """
        imgs = inputs['imgs']
        B = imgs.shape[0]
        if self.backbone_name == 'csn':
            x = self.backbone(imgs)
            import onnxruntime as ort

            np_imgs = einops.rearrange(imgs.cpu(), 'b t c h w -> b c t h w')
            np_imgs = np_imgs.numpy()
            ort_session = ort.InferenceSession("csn.onnx", providers=["CUDAExecutionProvider"])
            print(ort.get_device())
            outputs = ort_session.run(
                None,
                {"video": np_imgs},
            )[0]
            cpx = einops.rearrange(x, 'b t c h w -> b c t h w')
            print(outputs.shape, cpx.shape)
            print(torch.allclose(torch.from_numpy(outputs), cpx.cpu()))
            print(torch.from_numpy(outputs).sum((0, 1, 2)), cpx.cpu().sum((0, 1, 2)))

            x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        else:
            imgs = einops.rearrange(imgs, 'b t c h w -> (b t) c h w')
            x = self.extract_features(imgs)

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.trans_imgs_embedding(x)
        x = einops.rearrange(x, '(b t) c -> b c t', b=B)  # (4, 512, 100)
        origin_x = x

        x = self.temporal_embedding(x)

        # x0 = self.extractor0(x)
        # x1 = self.extractor1(x)
        # x2 = self.extractor2(x)  # b t c
        # feats = x0 + x1 + x2  # b t c
        # weights = F.softmax(self.windows_weight(feats), dim=-1)  # b t 3
        #
        # feats = x0 * weights[:, :, 0:1] + x1 * weights[:, :, 1:2] + x2 * weights[:, :, 2:3]

        feats = self.extractor(x)  # b t c

        # compare_feat = cosine_compare(x, self.temporal_model, k=self.kernel_size)

        left_temporal_feats = temporal_features(origin_x, self.left_temporal_model, k=self.kernel_size, mode='left')
        right_temporal_feats = temporal_features(origin_x, self.right_temporal_model, k=self.kernel_size, mode='right')
        # # print(x.shape, feats.shape, left_temporal_feats.shape, right_temporal_feats.shape)
        # # quit()
        #
        feats = torch.cat([feats, left_temporal_feats, right_temporal_feats], dim=-1)
        # feats = compare_feat
        feats = self.output(feats)
        logits = self.classifier(feats)  # b t 1

        # if is_main_process():
        #     print(feats.shape, logits.shape)

        if self.training:
            targets = targets.to(logits.dtype)
            gaussian_targets = prepare_gaussian_targets(targets)
            # gaussian_targets = targets

            # if is_main_process():
            #     for i in range(targets.shape[0]):
            #         import matplotlib.pyplot as plt
            #         img = gaussian_targets[i].reshape(1, -1).cpu().to(torch.float32).numpy()
            #         print(targets[i])
            #         plt.imshow(img)
            #         plt.savefig(f'a{i}.pdf')
            #
            # quit()
            # loss = F.l1_loss(logits.view(-1), gaussian_targets.view(-1))
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), gaussian_targets.view(-1))
            # loss = sigmoid_focal_loss(logits.view(-1), gaussian_targets.view(-1), reduction='mean')

            # loss = sigmoid_focal_loss(logits.view(-1), targets.view(-1), alpha=0.25, gamma=3, reduction='mean')

            # loss = F.binary_cross_entropy_with_logits(logits.view(-1), targets.view(-1), pos_weight=logits.new_tensor([.2, 1.0]))
            # logits = einops.rearrange(logits, '(b t) 1 -> b t', b=B)
            # loss = F.kl_div(logits, targets)
            # print(loss)

            # loss = F.cross_entropy(logits, targets.view(-1))
            loss_dict = {'loss': loss}
            return loss_dict
        # scores = F.softmax(logits, dim=1)[:, 1]
        scores = torch.sigmoid(logits)[:, :, 0]
        # scores = logits.view(-1)
        # scores = einops.rearrange(scores, '(b t) -> b t', b=B)
        return scores
