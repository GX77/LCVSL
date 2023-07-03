import os

import einops
import torch
import torch.nn as nn

import mmaction
from mmaction.models import build_model
from mmcv import Config


class CSN(nn.Module):
    def __init__(self):
        super().__init__()
        mmaction_root = os.path.dirname(os.path.abspath(mmaction.__file__))
        config_file = os.path.join(mmaction_root, os.pardir, 'configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py')
        cfg = Config.fromfile(config_file)
        cfg.model.backbone.bn_frozen = False

        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        state_dict = torch.load(os.path.join(mmaction_root, os.pardir, 'ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth'))
        print('load from ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth', flush=True)
        model.load_state_dict(state_dict['state_dict'])
        del model.cls_head
        self.model = model

    def forward(self, x):
        x = einops.rearrange(x, 'b t c h w -> b c t h w')
        """csn backbone need (B, C, T, H, W) format"""
        x = self.model.extract_feat(x)
        x = einops.rearrange(x, 'b c t h w -> b t c h w')
        return x


class TSN(nn.Module):
    def __init__(self):
        super().__init__()
        mmaction_root = os.path.dirname(os.path.abspath(mmaction.__file__))
        config_file = os.path.join(mmaction_root, os.pardir, 'configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py')
        cfg = Config.fromfile(config_file)

        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        state_dict = torch.load(os.path.join(mmaction_root, os.pardir, 'tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'))
        print('load from tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth', flush=True)
        model.load_state_dict(state_dict['state_dict'])
        del model.cls_head
        self.model = model

    def forward(self, x):
        B = x.shape[0]
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.model.extract_feat(x)
        x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=B)
        return x
