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


class FPN(Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        self.layer_blocks = nn.ModuleList()
        self.inner_blocks = nn.ModuleList()
        for c in in_channels:
            self.layer_blocks.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            self.inner_blocks.append(nn.Conv2d(c, dim, 1), )

    def forward(self, features_list):
        last_inner = self.inner_blocks[-1](features_list[-1])
        outputs = [self.layer_blocks[-1](last_inner)]
        for i in range(len(features_list) - 2, -1, -1):
            inner_lateral = self.inner_blocks[i](features_list[i])
            top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode='bilinear', align_corners=False)
            last_inner = inner_lateral + top_down
            outputs.insert(0, self.layer_blocks[i](last_inner))

        return outputs


class SidedataModel(Module):
    def __init__(self, cfg, dim, mode='res'):
        super().__init__()
        assert mode in ['res', 'mv', 'rgb']
        assert cfg.INPUT.USE_SIDE_DATA
        self.backbone = getattr(models, cfg.MODEL.BACKBONE.SIDE_DATA_NAME)(pretrained=True)
        self.out_features = self.backbone.fc.in_features
        del self.backbone.fc

        if mode == 'mv':
            setattr(self.backbone, 'conv1', nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
            self.bn = nn.BatchNorm2d(2)
        elif mode == 'res':
            self.bn = nn.BatchNorm2d(3)
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

        # outputs = self.fpn(outputs)
        outputs = self.embedding(x)
        return outputs


class EstimatorDenseNetTiny(nn.Module):
    def __init__(self, ch_in):
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
        self.predict_flow = nn.Conv2d(ch_in + dd, 2, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = torch.cat((self.conv0(x), x), 1)
        x = torch.cat((self.conv1(x), x), 1)
        x = torch.cat((self.conv2(x), x), 1)
        x = torch.cat((self.conv3(x), x), 1)
        x = torch.cat((self.conv4(x), x), 1)
        return self.predict_flow(x)


class UpsampleUpdatingModel2(Module):
    def __init__(self, cfg, in_feat_dim, dim):
        super().__init__()
        self._use_gan = cfg.MODEL.USE_GAN
        self.mv_backbone = SidedataModel(cfg, dim, mode='mv')
        self.offset_predictor = EstimatorDenseNetTiny(2)
        self.de_conv1 = DeformConv2d(dim, dim, kernel_size=1)

        self.motion_convs = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

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

        p_motions = einops.rearrange(p_motions, 'bn gop c h w -> (bn gop) c h w')
        p_features_2d = p_features = self.mv_backbone.extract_features(p_motions)
        p_motions_resized = F.interpolate(p_motions, size=p_features.shape[-2:], mode='bilinear', align_corners=False)
        p_offsets = self.offset_predictor(p_motions_resized)

        i_features = einops.rearrange(i_features, '(b n) c h w -> b n c h w', b=B)  # (4, 25, 256, 7, 7)
        p_offsets = einops.rearrange(p_offsets, '(b n t) c h w -> b n t c h w', b=B, n=num_gop)  # (4, 25, 3, 256, 7, 7)

        i_features_expand = i_features.unsqueeze(2).repeat(1, 1, p_offsets.shape[2], 1, 1, 1)  # (4, 8, 11, 512, 7, 7)
        i_features_flatten = einops.rearrange(i_features_expand, 'b n p_gop c h w -> (b n p_gop) c h w')
        p_offsets_flatten = einops.rearrange(p_offsets, 'b n p_gop c h w -> (b n p_gop) c h w')

        recalibrated_features = self.de_conv1(i_features_flatten, p_offsets_flatten)
        recalibrated_features = self.motion_convs(recalibrated_features)
        #
        # if is_main_process():
        #     print(recalibrated_features.shape)
        #     print(p_features.shape)
        # quit()

        p_features = recalibrated_features + p_features

        p_features = einops.rearrange(p_features, '(b n k) c h w -> b n k c h w', b=B, n=num_gop)
        p_features = F.adaptive_avg_pool2d(p_features, 1).flatten(3)  # b n k c

        p_features_2d = None
        return p_features, p_features_2d


class E2ECompressedGEBDModel(Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.INPUT.USE_SIDE_DATA
        self._use_gan = cfg.MODEL.USE_GAN
        self._use_residual = cfg.MODEL.USE_RESIDUAL
        self._use_mv_as_deconv_params = cfg.MODEL.USE_MV_AS_DECONV_PARAMS

        if is_main_process():
            print('USE_GAN:', self._use_gan)
            print('USE_RESIDUAL:', self._use_residual)
            print('USE_MV_AS_DECONV_PARAMS:', self._use_mv_as_deconv_params)

        self.backbone = getattr(models, cfg.MODEL.BACKBONE.NAME)(pretrained=True)
        in_feat_dim = self.backbone.fc.in_features
        self.kernel_size = 8
        dim = 256

        del self.backbone.fc
        if self._use_residual:
            self.res_backbone = SidedataModel(cfg, dim, mode='res')
            # self.trans_res_embedding = nn.Conv2d(self.res_backbone.out_features, dim, kernel_size=1)

        if self._use_mv_as_deconv_params:
            self.deform_updating_modules = UpsampleUpdatingModel2(cfg, 256, dim)
        else:
            self.mv_backbone = SidedataModel(cfg, dim, mode='mv')
            # self.mv_backbone = SidedataModel(cfg, dim, mode='rgb')
            # self.trans_mv_embedding = nn.Conv2d(self.mv_backbone.out_features, dim, kernel_size=1)

        # self.trans_imgs_embedding = nn.Conv2d(in_feat_dim, dim, kernel_size=1)

        # self.extractor = LeftRightFeatureExtractor(dim, stride=1, kernel_size=self.kernel_size)
        # self.temporal_embedding = nn.Sequential(
        #     nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(),
        #     nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(),
        #     nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(),
        #     nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(),
        # )

        self.output = nn.Sequential(
            nn.Linear(dim * 1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim * 2),
            nn.Dropout(0.2),
            nn.LayerNorm(dim * 2)
        )

        self.classifier = nn.Linear(
            in_features=dim * 2,
            out_features=1,
        )

        # FPN
        # self.fpn = FPN([256, 512, 1024, 2048], dim)
        self.embedding = nn.Conv2d(2048, dim, 3, 1, 1)

    def extract_features(self, x):
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

        # outputs = self.fpn(outputs)
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

        if self._use_residual:
            p_res = einops.rearrange(res, 'b (n gop) c h w -> (b n) gop c h w', gop=GOP)  # (32, 12, 3, 224, 224)
            p_res = p_res[:, 1:]  # (32, 11, 3, 224, 224)

        i_features = self.extract_features(einops.rearrange(i_imgs, 'b t c h w -> (b t) c h w'))  # (32, 2048, 7, 7)

        p_features_origin = None
        if self._use_gan and self.training:
            p_features_origin = self.extract_features(einops.rearrange(p_imgs, 'bn gop c h w -> (bn gop) c h w'))

        if self._use_residual:
            res_feats = self.res_backbone.extract_features(einops.rearrange(p_res, 'bn gop c h w -> (bn gop) c h w'))
            # res_feats = self.trans_res_embedding(res_feats)
            res_feats = einops.rearrange(res_feats, '(b n gop) c h w -> b n gop c h w', b=B, n=num_gop)  # (4, 25, 3, 512, 7, 7)
            res_feats = F.adaptive_avg_pool2d(res_feats, 1).flatten(3)  # (4, 25, 3, 512)

        if self._use_mv_as_deconv_params:
            p_features, p_features_2d = self.deform_updating_modules(imgs, i_features, p_motions)
        else:
            p_features = self.mv_backbone.extract_features(einops.rearrange(p_motions, 'bn gop c h w -> (bn gop) c h w'))
            # p_features = self.trans_mv_embedding(p_features)
            p_features = einops.rearrange(p_features, '(b n gop) c h w -> b n gop c h w', b=B, n=num_gop)  # (4, 25, 3, 512, 7, 7)
            p_features = F.adaptive_avg_pool2d(p_features, 1).flatten(3)  # (4, 25, 3, 512)

        # i_features = self.trans_imgs_embedding(i_features)
        i_features = einops.rearrange(i_features, '(b n) c h w -> b n c h w', b=B)  # (4, 8, 512, 7, 7)
        i_features = F.adaptive_avg_pool2d(i_features, 1).flatten(2)

        feats = torch.zeros(B, num_gop, GOP, i_features.shape[-1], dtype=i_features.dtype, device=i_features.device)  # (4, 8, 3, c, h, w)
        feats[:, :, 0] = i_features
        if self._use_residual:
            feats[:, :, 1:] = p_features + res_feats
        else:
            feats[:, :, 1:] = p_features

        feats = einops.rearrange(feats, 'b n gop c -> b (n gop) c')

        # feats = einops.rearrange(feats, 'b t c -> b c t', b=B)  # (4, 512, 100)

        # feats_for_gan = feats = self.temporal_embedding(feats)
        # feats = self.extractor(feats)  # b t c
        # feats = einops.rearrange(feats, 'b c t -> b t c')

        feats = self.output(feats)
        logits = self.classifier(feats)  # b t 1

        if self.training:
            targets = targets.to(logits.dtype)
            gaussian_targets = prepare_gaussian_targets(targets)
            frame_mask = frame_mask.view(-1) == 1

            loss = F.binary_cross_entropy_with_logits(logits.view(-1)[frame_mask], gaussian_targets.view(-1)[frame_mask])
            loss_dict = {'loss': loss}

            if self._use_gan:
                p_frame_mask = p_frame_mask.reshape(-1) == 1

                # p_features_origin = einops.rearrange(p_features_origin, '(b n k) c h w -> b n k c h w', b=B, n=num_gop)
                # p_features_origin = F.adaptive_avg_pool2d(p_features_origin, 1).flatten(3)  # b n k c 1 1
                #
                # origin_feats = torch.zeros(B, num_gop, GOP, i_features.shape[-1], dtype=i_features.dtype, device=i_features.device)
                # origin_feats[:, :, 0] = i_features
                # origin_feats[:, :, 1:] = p_features_origin + res_feats
                #
                # origin_feats = einops.rearrange(origin_feats, 'b n gop c -> b (n gop) c')
                # origin_feats = einops.rearrange(origin_feats, 'b t c -> b c t', b=B)  # (4, 512, 100)
                # origin_feats = self.temporal_embedding(origin_feats)
                #
                # origin_feats = einops.rearrange(origin_feats, 'b c t -> (b t) c')
                # feats_for_gan = einops.rearrange(feats_for_gan, 'b c t -> (b t) c')
                #
                # align_loss = F.mse_loss(feats_for_gan[frame_mask], origin_feats[frame_mask].detach(), reduction='mean')
                # align_loss = align_loss.sum(dim=1).mean()

                # p_features_2d = F.adaptive_avg_pool2d(p_features_2d, 1).flatten(1)  # (300, 256, 56, 56)
                # p_features_origin = F.adaptive_avg_pool2d(p_features_origin, 1).flatten(1)  # (300, 256, 56, 56)

                p_features_2d = p_features_2d.view(p_features_2d.shape[0], -1)  # (300, 256, 56, 56)
                p_features_origin = p_features_origin.view(p_features_origin.shape[0], -1)  # (300, 256, 56, 56)
                align_loss = F.mse_loss(p_features_2d[p_frame_mask], p_features_origin[p_frame_mask].detach())

                loss_dict['align_loss'] = align_loss * 0.10

            return loss_dict
        scores = torch.sigmoid(logits)[:, :, 0]

        return scores


class E2EBaselineGEBDModel(Module):
    def __init__(self, cfg):
        super().__init__()
        print('Using E2EBaselineGEBDModel')
        assert cfg.INPUT.USE_SIDE_DATA
        self.backbone = getattr(models, cfg.MODEL.BACKBONE.NAME)(pretrained=True)
        in_feat_dim = self.backbone.fc.in_features
        del self.backbone.fc
        # self.res_backbone = SidedataModel(cfg, mode='res')
        # self.mv_backbone = SidedataModel(cfg, mode='mv')

        self.kernel_size = 8
        dim = 256

        self.trans_imgs_embedding = nn.Conv2d(in_feat_dim, dim, kernel_size=1)

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

        outputs = []
        x = self.backbone.layer1(x)  # (256 * 56 * 56)
        outputs.append(x)
        x = self.backbone.layer2(x)  # (512 * 28 * 28)
        outputs.append(x)
        x = self.backbone.layer3(x)  # (1024 * 14 * 14)
        outputs.append(x)
        x = self.backbone.layer4(x)  # (2048 * 7 * 7)
        outputs.append(x)

        return tuple(outputs)

    def forward(self, inputs, targets=None):
        """
        Args:
            inputs(dict): imgs (B, T, C, H, W);
            targets:
        Returns:
        """
        imgs = inputs['imgs']  # (4, 96, 3, 224, 224)
        mv = inputs['mv']  # (4, 96, 2, 224, 224)
        res = inputs['res']  # (4, 96, 3, 224, 224)
        frame_mask = inputs['frame_mask']  # (4, 96)

        B = imgs.shape[0]

        feats = self.extract_features(einops.rearrange(imgs, 'b t c h w -> (b t) c h w'))  # (32, 2048, 7, 7)
        feats = self.trans_imgs_embedding(feats[-1])
        feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)

        feats = einops.rearrange(feats, '(b t) c -> b c t', b=B)

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
