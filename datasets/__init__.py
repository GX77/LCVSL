import copy
import os

from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.dataloader import default_collate

from utils.sampler import DistBalancedBatchSampler
# from .dataset_old import GEBDDataset, ClipShotsDataset, MeixueDataset
from .dataset import GEBDDataset

ROOT = os.getenv('GEBD_ROOT', '/mnt/bn/hevc-understanding/datasets/GEBD/')


def build_dataloader(cfg, args, splits, is_train):
    assert len(splits) >= 1

    def build_dataset(split):
        folder = {
            'train': 'GEBD_train_frames',
            'val': 'GEBD_val_frames',
            'minval': 'GEBD_val_frames',
            'val_minus_minval': 'GEBD_val_frames',
            'test': 'GEBD_test_frames',
        }
        assert split in folder, f'Dataset {split} not exists!'
        root = os.path.join(ROOT, folder[split])

        dataset = GEBDDataset(cfg, root=root,
                              split=split,
                              train=is_train)
        if args.device == 'cpu':
            dataset.annotations = dataset.annotations[:500]
        return dataset

    datasets = []
    for split in splits:
        datasets.append(build_dataset(split))

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        annotations = []
        for dataset in datasets:
            annotations.extend(copy.deepcopy(dataset.annotations))
        dataset = ConcatDataset(datasets)
        dataset.annotations = annotations

    if args.distributed:
        if is_train and not cfg.INPUT.END_TO_END:
            sampler = DistBalancedBatchSampler(dataset, num_classes=2, n_sample_classes=2, n_samples=cfg.SOLVER.BATCH_SIZE // 2)
        else:
            sampler = DistributedSampler(dataset, shuffle=is_train)
    else:
        sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)

    # collate_fn = (lambda x: x) if cfg.INPUT.END_TO_END else default_collate
    collate_fn = default_collate
    loader = DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                        sampler=sampler,
                        drop_last=False,
                        collate_fn=collate_fn,
                        # pin_memory=True,
                        num_workers=cfg.SOLVER.NUM_WORKERS)
    return loader
