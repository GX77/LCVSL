import argparse
import glob
import os
import pickle

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torchvision import transforms

from modeling import cfg, build_model
from utils.distribute import is_main_process


def get_idx_from_score_by_threshold(threshold=0.5, seq_indices=None, seq_scores=None):
    seq_indices = np.array(seq_indices)
    seq_scores = np.array(seq_scores)
    bdy_indices = []
    internals_indices = []
    for i in range(len(seq_scores)):
        if seq_scores[i] >= threshold:
            internals_indices.append(i)
        elif seq_scores[i] < threshold and len(internals_indices) != 0:
            bdy_indices.append(internals_indices)
            internals_indices = []

        if i == len(seq_scores) - 1 and len(internals_indices) != 0:
            bdy_indices.append(internals_indices)

    bdy_indices_in_video = []
    if len(bdy_indices) != 0:
        for internals in bdy_indices:
            center = int(np.mean(internals))
            bdy_indices_in_video.append(seq_indices[center])
    return bdy_indices_in_video


def image_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def load_frames(frames_dir):
    num_frames = len(os.listdir(frames_dir))
    frame_indices = np.arange(1, num_frames + 1, 1)
    frame_paths = []
    for idx in frame_indices:
        frame_paths.append(os.path.join(frames_dir, 'image_%05d.jpg' % idx))

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    imgs = [transform(image_loader(path)) for path in frame_paths]
    imgs = torch.stack(imgs, dim=0) * 2.0 - 1.0
    return imgs, frame_indices


@torch.no_grad()
def main(cfg, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg)
    model = model.to(device)
    model.eval()

    if args.resume:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        start_epoch = state_dict['epoch']
        if is_main_process():
            print('Loaded from {}, Epoch: {}'.format(args.resume, start_epoch), flush=True)

    if is_main_process():
        print(model)

    results = []
    for frames_dir in glob.glob(os.path.join(args.frames_dir, '*')):
        if not os.path.isdir(frames_dir):
            continue

        imgs, frame_indices = load_frames(frames_dir)

        inputs = {'imgs': imgs.to(device)[None]}

        scores = model(inputs)[0][0].cpu().numpy()
        threshold = 0.5
        cap = cv2.VideoCapture(frames_dir + '.mp4')
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(imgs.shape, fps)
        det_t = np.array(get_idx_from_score_by_threshold(threshold=threshold,
                                                         seq_indices=frame_indices,
                                                         seq_scores=scores)) / fps
        # with open(frames_dir + '.pkl', 'wb') as f:
        #     pickle.dump(det_t, f)
        results.append(
            '{}: {}'.format(frames_dir.split('/')[-1] + '.mp4', ', '.join(map(lambda x: str(round(x, 2)), det_t)))
        )
        np.save(os.path.basename(frames_dir), scores)
    with open('CLIPSHOTS_RGB.txt', 'w') as f:
        f.write('\n'.join(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help="path to config file", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--frames_dir", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    local_rank = args.local_rank
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            dist.barrier()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if is_main_process():
        print('Args: \n{}'.format(args))
        print('Configs: \n{}'.format(cfg))

    main(cfg, args)
