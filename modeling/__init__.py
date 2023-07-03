import torch
from .config import _C as cfg

from .baseline_model import GEBDModel
from .compressed_model import CompressedGEBDModel

from .e2e_compressed_model_tip import E2ECompressedGEBDModel


def build_model(cfg):
    model = E2ECompressedGEBDModel(cfg)

    if cfg.MODEL.SYNC_BN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model
