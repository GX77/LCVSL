import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torchvision import models
from torchvision.ops import DeformConv2d

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


class SidedataModel(Module):
    def __init__(self, cfg, mode='res'):
        super().__init__()
        assert mode in ['res', 'mv']
        assert cfg.INPUT.USE_SIDE_DATA
        self.backbone = getattr(models, cfg.MODEL.BACKBONE.SIDE_DATA_NAME)(pretrained=True)
        self.out_features = self.backbone.fc.in_features
        del self.backbone.fc

        if mode == 'mv':
            setattr(self.backbone, 'conv1', nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
            self.bn = nn.BatchNorm2d(2)
        elif mode == 'res':
            self.bn = nn.BatchNorm2d(3)

    def extract_features(self, x):
        # if is_main_process():
        #     print(x.shape)  # 224
        x = self.bn(x)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        # if is_main_process():
        #     print(x.shape)  # 56
        x = self.backbone.layer2(x)
        # if is_main_process():
        #     print(x.shape)  # 28
        x = self.backbone.layer3(x)
        # if is_main_process():
        #     print(x.shape)  # 14
        x = self.backbone.layer4(x)
        # if is_main_process():
        #     print(x.shape)  # 7
        # quit()
        return x


class E2ECompressedGEBDModel(Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.INPUT.USE_SIDE_DATA
        self.backbone = getattr(models, cfg.MODEL.BACKBONE.NAME)(pretrained=True)
        in_feat_dim = self.backbone.fc.in_features
        del self.backbone.fc
        # self.res_backbone = SidedataModel(cfg, mode='res')
        self.mv_backbone = SidedataModel(cfg, mode='mv')

        self.kernel_size = 8
        dim = 512
        self.trans_imgs_embedding = nn.Conv2d(in_feat_dim, dim, kernel_size=1)

        self.de_conv1 = DeformConv2d(dim, dim, kernel_size=1)

        self.extractor = LeftRightFeatureExtractor(dim, stride=1, kernel_size=self.kernel_size)
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

        self.output = nn.Sequential(
            nn.Linear(dim * 2, dim),
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
            targets:
        Returns:
        """
        imgs = inputs['imgs']  # (4, 96, 3, 224, 224)
        mv = inputs['mv']  # (4, 96, 2, 224, 224)
        # res = inputs['res']
        frame_mask = inputs['frame_mask']  # (4, 96)

        B = imgs.shape[0]

        i_imgs = imgs[:, ::GOP]  # (4, 8, 3, 224, 224)
        num_gop = i_imgs.shape[1]

        p_motions = einops.rearrange(mv, 'b (n gop) c h w -> (b n) gop c h w', gop=GOP)  # (32, 12, 2, 224, 224)
        p_motions = p_motions[:, 1:]  # (32, 11, 2, 224, 224)

        i_features = self.extract_features(einops.rearrange(i_imgs, 'b t c h w -> (b t) c h w'))  # (32, 2048, 7, 7)
        i_features = self.trans_imgs_embedding(i_features)
        i_features = einops.rearrange(i_features, '(b n) c h w -> b n c h w', b=B)  # (4, 8, 512, 7, 7)

        scale = i_features.shape[-1] / imgs.shape[-1]
        p_motions = F.interpolate(einops.rearrange(p_motions, 'b t c h w -> (b t) c h w'), size=i_features.shape[-2:], mode='bilinear', align_corners=False) * scale
        p_motions = einops.rearrange(p_motions, '(b n t) c h w -> b n t c h w', b=B, n=num_gop)  # (4, 8, 11, 2, 7, 7)

        i_features_expand = i_features.unsqueeze(2).repeat_interleave(p_motions.shape[2], dim=2)  # (4, 8, 11, 512, 7, 7)

        i_features_flatten = einops.rearrange(i_features_expand, 'a1 s2 a3 c h w -> (a1 s2 a3) c h w')
        p_motions_flatten = einops.rearrange(p_motions, 'a1 s2 a3 c h w -> (a1 s2 a3) c h w')

        p_features = self.de_conv1(i_features_flatten, p_motions_flatten)  # ((4, 8, 11), 512, 7, 7)
        p_features = einops.rearrange(p_features, '(b n k) c h w -> b n k c h w', b=B, n=num_gop)

        feats = torch.zeros(B, num_gop, GOP, *i_features.shape[-3:], dtype=i_features.dtype, device=i_features.device)  # (4, 8, 12, c, h, w)
        feats[:, :, 0] = i_features
        feats[:, :, 1:] = p_features

        feats = einops.rearrange(feats, 'b n gop c h w -> b (n gop) c h w')

        feats = F.adaptive_avg_pool2d(feats, 1).flatten(2)
        feats = einops.rearrange(feats, 'b t c -> b c t', b=B)  # (4, 512, 100)

        feats = self.temporal_embedding(feats)
        feats = self.extractor(feats)  # b t c

        feats = self.output(feats)
        logits = self.classifier(feats)  # b t 1

        if self.training:
            targets = targets.to(logits.dtype)
            gaussian_targets = prepare_gaussian_targets(targets)
            frame_mask = frame_mask.view(-1) == 1

            loss = F.binary_cross_entropy_with_logits(logits.view(-1)[frame_mask], gaussian_targets.view(-1)[frame_mask])
            loss_dict = {'loss': loss}
            return loss_dict
        scores = torch.sigmoid(logits)[:, :, 0]

        return scores
