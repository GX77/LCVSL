import torch
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import MultiStepLR
from .optim import BertAdam


def build_optimizer(cfg, params, data):
    kwargs = dict(lr=cfg.SOLVER.LR,
                  weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    if cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM, **kwargs)
    elif cfg.SOLVER.OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(params, **kwargs)
    elif cfg.SOLVER.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(params, **kwargs)
    elif cfg.SOLVER.OPTIMIZER == 'Ours':
        num_train_optimization_steps = len(data) * cfg.SOLVER.MAX_EPOCHS
        optimizer = BertAdam(params,
                             lr=cfg.SOLVER.LR,
                             warmup=0.1,
                             t_total=num_train_optimization_steps,
                             schedule="warmup_linear")
    else:
        raise NotImplementedError
    return optimizer
