import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torchvision import models


class SidedataModel(Module):
    def __init__(self, cfg, mode='res'):
        super().__init__()
        assert mode in ['res', 'mv']
        assert cfg.INPUT.USE_SIDE_DATA
        self.backbone = getattr(models, cfg.MODEL.BACKBONE.SIDE_DATA_NAME)(pretrained=True)
        self.out_features = self.backbone.fc.in_features * 2
        del self.backbone.fc

        if mode == 'mv':
            setattr(self.backbone, 'conv1', nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
            self.bn = nn.BatchNorm2d(2)
        elif mode == 'res':
            self.bn = nn.BatchNorm2d(3)

    def extract_features(self, x):
        x = self.bn(x)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x


class CompressedGEBDModel(Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.INPUT.USE_SIDE_DATA
        self.backbone = getattr(models, cfg.MODEL.BACKBONE.NAME)(pretrained=True)
        in_feat_dim = self.backbone.fc.in_features * 2
        del self.backbone.fc

        # self.res_backbone = SidedataModel(cfg, mode='res')
        self.mv_backbone = SidedataModel(cfg, mode='mv')

        dim = 1024
        self.trans_imgs_embedding = nn.Sequential(
            nn.Linear(in_feat_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.trans_res_embedding = None
        # if self.res_backbone.out_features != dim:
        #     self.trans_res_embedding = nn.Sequential(
        #         nn.Linear(self.res_backbone.out_features, dim),
        #         nn.ReLU(),
        #         nn.Linear(dim, dim),
        #     )

        self.trans_mv_embedding = None
        if self.mv_backbone.out_features != dim:
            self.trans_mv_embedding = nn.Sequential(
                nn.Linear(self.mv_backbone.out_features, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            )

        self.classifier = torch.nn.Linear(
            in_features=dim,
            out_features=2,
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

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

    def forward(self, inputs, targets=None):
        """
        Args:
            inputs(dict): imgs (B, T, C, H, W);
            targets:
        Returns:
        """
        imgs = inputs['imgs']
        mv = inputs['mv']
        # res = inputs['res']
        i_frame_mask = inputs['i_frame_mask']  # (B, f * 2)
        p_frame_mask = 1 - i_frame_mask  # (B, f * 2)
        B = imgs.shape[0]

        # B = mv.shape[0]

        # imgs = einops.rearrange(imgs, 'b t c h w ->(b t) c h w')
        # imgs = self.extract_features(imgs)
        # imgs = einops.rearrange(imgs, '(b s f) c h w -> b s f c h w', b=B, s=2)  # 2 means preceding `f` frames and succeeding `f` frames
        # imgs = torch.mean(imgs, dim=2)
        # imgs = F.adaptive_avg_pool2d(imgs, 1)
        # imgs = imgs.flatten(1)

        def FF(sample, model, mask):
            sample = einops.rearrange(sample, 'b t c h w -> (b t) c h w')
            mask_flatten = einops.rearrange(mask, 'b t -> (b t)') == 1
            sample_flatten = model.extract_features(sample[mask_flatten])

            sample = torch.zeros(len(mask_flatten), *sample_flatten.shape[1:], dtype=sample_flatten.dtype, device=sample_flatten.device)
            sample[mask_flatten] = sample_flatten

            sample = einops.rearrange(sample, '(b s f) c h w -> b s f c h w', b=B, s=2)  # 2 means preceding `f` frames and succeeding `f` frames
            # ([32, 2, 5, 2048, 7, 7])
            mask = einops.rearrange(mask, 'b (s f) -> b s f 1 1 1', s=2)

            sample = (sample * mask).sum(dim=2) / (mask.sum(dim=2) + 1e-12)

            # sample = torch.mean(sample, dim=2)
            sample = F.adaptive_avg_pool2d(sample, 1)
            sample = sample.flatten(1)
            return sample

        imgs = FF(imgs, self, i_frame_mask)
        mv = FF(mv, self.mv_backbone, p_frame_mask)
        # res = FF(res, self.res_backbone, p_frame_mask)

        imgs = self.trans_imgs_embedding(imgs)
        if self.trans_mv_embedding is not None:
            mv = self.trans_mv_embedding(mv)
        # if self.trans_res_embedding is not None:
        #     res = self.trans_res_embedding(res)

        # embeddings = imgs + mv + res
        embeddings = imgs + mv

        logits = self.classifier(embeddings)

        if self.training:
            loss = F.cross_entropy(logits, targets)
            loss_dict = {'loss': loss}
            return loss_dict

        scores = F.softmax(logits, dim=-1)[:, 1]
        return scores
