import einops
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = getattr(models, 'resnet50' if cfg is None else cfg.MODEL.BACKBONE.NAME)(pretrained=True)
        self.num_channels = self.backbone.fc.in_features
        del self.backbone.fc

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

    def forward(self, x, ):
        """
        Args:
            x: (B, T, C, H, W);
        Returns:
            (B, T, C);
        """
        B = x.shape[0]
        x = einops.rearrange(x, 'b t c h w ->(b t) c h w')
        x = self.extract_features(x)  # (b t) c h w
        x = F.adaptive_avg_pool2d(x, 1)  # (b t) c 1 1
        x = einops.rearrange(x, '(b t) c 1 1 -> b t c', b=B)
        return x


def build_backbone(cfg):
    model = Backbone(cfg)
    return model
