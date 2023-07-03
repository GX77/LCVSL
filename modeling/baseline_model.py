import einops
import torch
import torch.nn.functional as F
from torch.nn import Module
from torchvision import models


class GEBDModel(Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = getattr(models, cfg.MODEL.BACKBONE.NAME)(pretrained=True)
        in_feat_dim = self.backbone.fc.in_features * 2
        del self.backbone.fc

        self.classifier = torch.nn.Linear(
            in_features=in_feat_dim,
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
        x = inputs['imgs']
        B = x.shape[0]
        x = einops.rearrange(x, 'b t c h w ->(b t) c h w')
        x = self.extract_features(x)
        x = einops.rearrange(x, '(b s f) c h w -> b s f c h w', b=B, s=2)  # 2 means preceding `f` frames and succeeding `f` frames
        x = torch.mean(x, dim=2)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.flatten(1)
        logits = self.classifier(x)

        if self.training:
            loss = F.cross_entropy(logits, targets)
            loss_dict = {'loss': loss}
            return loss_dict

        scores = F.softmax(logits, dim=-1)[:, 1]
        return scores
