import glob
import json
import random
import time

import math
import os
from pickle5 import pickle as pickle5
import pickle
from typing import List

import cv2
import torchvision.transforms.functional
# import libde265
# import hevc_reader
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from decord import VideoReader
from decord import VideoLoader
from decord import cpu, gpu
import decord

decord.bridge.set_bridge('torch')

from tqdm import tqdm

from .prepare_data import read_compressed_features
from utils.distribute import synchronize, is_main_process

MV_SIZE = 224


def image_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def prepare_annotations(cfg, root, split):
    frame_per_side = cfg.INPUT.FRAME_PER_SIDE
    ds = cfg.INPUT.DOWNSAMPLE
    dynamic_downsample = cfg.INPUT.DYNAMIC_DOWNSAMPLE
    min_change_dur = 0.3

    ann_path = os.path.join('data', f'k400_mr345_{split}_min_change_duration0.3.pkl')

    sample = '{}'.format(f'_dynamic{ds}' if dynamic_downsample else ds)
    if cfg.INPUT.NO_DOWNSAMPLE:
        sample = '_no'

    filename = '{}-cache-fps{}-ds{}.pkl'.format(split, frame_per_side, sample)
    if cfg.INPUT.END_TO_END:
        filename = 'end_to_end{}_'.format(cfg.INPUT.SEQUENCE_LENGTH) + filename
    if cfg.INPUT.USE_GOP:
        filename = 'gop_' + filename
    cache_path = os.path.join('data', 'caches', filename)

    if is_main_process() and not os.path.exists(cache_path):
        with open(ann_path, 'rb') as f:
            dict_train_ann = pickle.load(f, encoding='lartin1')

        annotations = []
        neg = 0
        pos = 0

        max_boundary = -1
        for v_name in dict_train_ann.keys():
            v_dict = dict_train_ann[v_name]
            fps = v_dict['fps']
            f1_consis = v_dict['f1_consis']
            path_frame = v_dict['path_frame']
            video_duration = v_dict['video_duration']

            cls, frame_folder = path_frame.split('/')[:2]
            video_dir = os.path.join(root, cls, frame_folder)
            if not os.path.exists(video_dir):
                continue
            vlen = len(os.listdir(video_dir))

            if dynamic_downsample:
                downsample = max(math.ceil(fps / ds), 1)
            else:
                downsample = ds

            # select the annotation with highest f1 score
            highest = np.argmax(f1_consis)
            change_indices = v_dict['substages_myframeidx'][highest]
            change_timestamps = v_dict['substages_timestamps'][highest]
            if len(change_indices) > max_boundary:
                max_boundary = len(change_indices)

            if cfg.INPUT.END_TO_END:
                if cfg.INPUT.USE_GOP:
                    indices = np.arange(1, min(300, vlen) + 1, dtype=int)
                    if len(indices) < 300:
                        indices = np.concatenate((indices, np.ones((300 - len(indices),), dtype=int) * -1))

                    assert len(indices) == 300
                    selected_indices = indices[::3]
                elif cfg.INPUT.NO_DOWNSAMPLE:
                    selected_indices = np.arange(1, cfg.INPUT.SEQUENCE_LENGTH + 1, dtype=int)
                    selected_indices[vlen:] = -1
                else:
                    selected_indices = np.linspace(1, vlen, cfg.INPUT.SEQUENCE_LENGTH, dtype=int)

                half_dur_2_nframes = min_change_dur * fps / 2.

                labels = []
                for i in selected_indices:
                    labels.append(0)
                    for change in change_indices:
                        if change - half_dur_2_nframes <= i <= change + half_dur_2_nframes:
                            labels.pop()  # pop '0'
                            labels.append(1)
                            break

                assert len(labels) == len(selected_indices)

                record = {
                    'folder': f'{cls}/{frame_folder}',
                    'block_idx': selected_indices.tolist(),
                    'label': labels,
                    'vid': v_name
                }

                # record = {
                #     'folder': f'{cls}/{frame_folder}',
                #     'block_idx': selected_indices.tolist(),
                #     'label': [1] * len(change_timestamps),
                #     'time_pos': np.clip(np.array(change_timestamps, dtype=np.float32) / video_duration, 0, 1),
                #     'vid': v_name
                # }
                annotations.append(record)
            else:
                # (float)num of frames with min_change_dur/2
                half_dur_2_nframes = min_change_dur * fps / 2.

                start_offset = 1
                selected_indices = np.arange(start_offset, vlen, downsample)

                # should be tagged as positive(bdy), otherwise negative(bkg)
                GT = []
                for i in selected_indices:
                    GT.append(0)
                    for change in change_indices:
                        if change - half_dur_2_nframes <= i <= change + half_dur_2_nframes:
                            GT.pop()  # pop '0'
                            GT.append(1)
                            break

                for idx, (current_idx, lbl) in enumerate(zip(selected_indices, GT)):
                    record = {}
                    shift = np.arange(-frame_per_side, frame_per_side)
                    shift[shift >= 0] += 1
                    shift = shift * downsample
                    block_idx = shift + current_idx
                    block_idx[block_idx < 1] = 1
                    block_idx[block_idx > vlen] = vlen
                    block_idx = block_idx.tolist()

                    record['folder'] = f'{cls}/{frame_folder}'
                    record['current_idx'] = current_idx
                    record['block_idx'] = block_idx
                    record['label'] = lbl
                    record['vid'] = v_name
                    annotations.append(record)
                    if lbl == 0:
                        neg += 1
                    else:
                        pos += 1

        print(f'Split: {split}, GT: {len(dict_train_ann)}, Annotations: {len(annotations)}, Num pos: {pos}, num neg: {neg}, total: {pos + neg}')
        print('Max boundary:', max_boundary)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(annotations, f)

    synchronize()
    with open(cache_path, 'rb') as f:
        annotations = pickle.load(f)

    if is_main_process():
        print(f'Loaded from {cache_path}')

    return annotations


def prepare_clipshots_annotations(cfg, split, root='/mnt/bn/hevc-understanding/datasets/ClipShots/ClipShots'):
    split = split.split('_')[-1]
    trans = False
    cache_path = os.path.join('data', 'caches', f'clipshots_{split}_size{cfg.INPUT.SEQUENCE_LENGTH}_sample{cfg.INPUT.DOWNSAMPLE}.pkl')
    if is_main_process() and not os.path.exists(cache_path):
        with open(os.path.join(root, 'annotations', f'{split}.json')) as f:
            json_ann = json.load(f)
        half_dur_2_nframes = 3

        annotations = []
        for vid in tqdm(json_ann, total=len(json_ann)):
            if vid == '5186782609.mp4':  # empty video
                continue
            ann = json_ann[vid]
            frame_num = int(ann['frame_num'])

            change_indices = []
            selected_indices = list(range(0, frame_num, cfg.INPUT.DOWNSAMPLE))
            for s, e in ann['transitions']:
                if trans and abs(e - s) > 6:
                    change_indices.append([s, e])
                else:
                    change_indices.append(int((s + e) / 2.0))

            labels = []
            for i in selected_indices:
                labels.append(0)
                for change in change_indices:
                    if isinstance(change, int):
                        if change - half_dur_2_nframes <= i <= change + half_dur_2_nframes:
                            labels.pop()  # pop '0'
                            labels.append(1)
                            break
                    else:
                        if change[0] <= i <= change[1]:
                            labels.pop()  # pop '0'
                            labels.append(1)
                            break

            path = os.path.join(root, 'videos', split, vid)
            size = cfg.INPUT.SEQUENCE_LENGTH
            start = 0
            overlap = 5
            # split into overlapping chunks
            pad_frame_num = len(selected_indices)
            if len(selected_indices) < size:
                diff = size - len(selected_indices)
                selected_indices = selected_indices + [-1] * diff
                labels = labels + [-1] * diff
                pad_frame_num = size

            assert len(labels) == len(selected_indices)
            while True:
                chunk_indices = selected_indices[start: start + size]
                chunk_labels = labels[start: start + size]
                record = {
                    'path': path,
                    'block_idx': chunk_indices,
                    'label': chunk_labels,
                    'vid': vid
                }
                annotations.append(record)

                if start + size >= pad_frame_num:
                    break
                start = start + size - overlap
                if start + size > pad_frame_num:
                    start = pad_frame_num - size

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(annotations, f)

    synchronize()
    with open(cache_path, 'rb') as f:
        annotations = pickle.load(f)

    if is_main_process():
        print(f'Loaded from {cache_path}')
    return annotations


def prepare_QV_annotations(cfg, split, root='/mnt/bn/hevc-understanding/datasets/HighlightsDataset/'):
    split = split.split('_')[-1]
    trans = False
    cache_path = os.path.join('data', 'caches', f'qvhighlights_{split}_size{cfg.INPUT.SEQUENCE_LENGTH}_sample{cfg.INPUT.DOWNSAMPLE}.pkl')
    if is_main_process() and not os.path.exists(cache_path):
        json_ann = []
        print(os.path.join(root, 'qvhighlights', f'highlight_{split}_release.jsonl'))
        with open(os.path.join(root, 'qvhighlights', f'highlight_{split}_release.jsonl')) as f:
            for line in f.readlines():
                json_ann.append(json.loads(line))

        half_dur_2_nframes = 3

        annotations = []
        for ann in tqdm(json_ann, total=len(json_ann)):
            vid = ann['vid']
            path = os.path.join(root, 'videos', vid + '.mp4')
            try:
                vr = VideoReader(path, ctx=cpu(0))
                frame_num = len(vr)
            except:
                print(f'Skip {vid}')
                continue

            change_indices = []
            selected_indices = list(range(0, frame_num, cfg.INPUT.DOWNSAMPLE))
            for s, e in ann['relevant_windows']:
                if trans and abs(e - s) > 6:
                    change_indices.append([s, e])
                else:
                    change_indices.append(int((s + e) / 2.0))

            labels = []
            for i in selected_indices:
                labels.append(0)
                for change in change_indices:
                    if isinstance(change, int):
                        if change - half_dur_2_nframes <= i <= change + half_dur_2_nframes:
                            labels.pop()  # pop '0'
                            labels.append(1)
                            break
                    else:
                        if change[0] <= i <= change[1]:
                            labels.pop()  # pop '0'
                            labels.append(1)
                            break

            size = cfg.INPUT.SEQUENCE_LENGTH
            start = 0
            overlap = 5
            # split into overlapping chunks
            pad_frame_num = len(selected_indices)
            if len(selected_indices) < size:
                diff = size - len(selected_indices)
                selected_indices = selected_indices + [-1] * diff
                labels = labels + [-1] * diff
                pad_frame_num = size

            assert len(labels) == len(selected_indices)
            while True:
                chunk_indices = selected_indices[start: start + size]
                chunk_labels = labels[start: start + size]
                record = {
                    'path': path,
                    'block_idx': chunk_indices,
                    'label': chunk_labels,
                    'vid': vid
                }
                annotations.append(record)

                if start + size >= pad_frame_num:
                    break
                start = start + size - overlap
                if start + size > pad_frame_num:
                    start = pad_frame_num - size

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(annotations, f)

    synchronize()
    with open(cache_path, 'rb') as f:
        annotations = pickle.load(f)

    if is_main_process():
        print(f'Loaded from {cache_path}')
    return annotations


def prepare_beats_annotations(cfg, split, root='/mnt/bn/hevc-understanding/datasets/beat'):
    version = '_v2'
    # version = ''
    split = split.split('_')[-1]
    cache_path = os.path.join('data', 'caches', f'beats{version}_{split}_size{cfg.INPUT.SEQUENCE_LENGTH}_sample{cfg.INPUT.DOWNSAMPLE}.pkl')
    if is_main_process() and not os.path.exists(cache_path):
        with open(os.path.join(root, f'beats_{split}{version}.json')) as f:
            json_ann = json.load(f)
        half_dur_2_nframes = 3

        annotations = []
        for ann in tqdm(json_ann, total=len(json_ann)):
            vid = ann['vid']
            frame_num = int(ann['frame_num'])
            fps = ann['fps']

            change_indices = []
            selected_indices = list(range(0, frame_num, cfg.INPUT.DOWNSAMPLE))
            for time_stamp in ann['gt_time_stamp']:
                change_indices.append(int(time_stamp * fps))

            labels = []
            for i in selected_indices:
                labels.append(0)
                for change in change_indices:
                    if change - half_dur_2_nframes <= i <= change + half_dur_2_nframes:
                        labels.pop()  # pop '0'
                        labels.append(1)
                        break

            path = os.path.join(root, 'preview', ann['path'])
            size = cfg.INPUT.SEQUENCE_LENGTH
            start = 0
            overlap = 5
            # split into overlapping chunks
            pad_frame_num = len(selected_indices)
            if len(selected_indices) < size:
                diff = size - len(selected_indices)
                selected_indices = selected_indices + [-1] * diff
                labels = labels + [-1] * diff
                pad_frame_num = size

            assert len(labels) == len(selected_indices)
            while True:
                chunk_indices = selected_indices[start: start + size]
                chunk_labels = labels[start: start + size]
                record = {
                    'path': path,
                    'block_idx': chunk_indices,
                    'label': chunk_labels,
                    'vid': vid
                }
                annotations.append(record)

                if start + size >= pad_frame_num:
                    break
                start = start + size - overlap
                if start + size > pad_frame_num:
                    start = pad_frame_num - size

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(annotations, f)

    synchronize()
    with open(cache_path, 'rb') as f:
        annotations = pickle.load(f)

    if is_main_process():
        print(f'Loaded from {cache_path}')
    return annotations


def prepare_meixue_annotations(cfg, split, root='/mnt/bn/compressed-highlight'):
    split = split.split('_')[-1]
    cache_path = os.path.join('data', 'caches', f'meixue_{split}.pkl')
    if is_main_process() and not os.path.exists(cache_path):
        with open(os.path.join(root, f'meixue_{split}.json')) as f:
            json_ann = json.load(f)

        annotations = []
        for ann in tqdm(json_ann, total=len(json_ann)):
            vid = ann['vid']
            selected_indices = np.array(ann['indices'])
            selected_indices, indices = np.unique(selected_indices, return_index=True)
            np_labels = np.array(ann['scores'])[indices]

            selected_indices = selected_indices.tolist()
            labels = []
            for i in range(len(np_labels)):
                label = [np_labels[i], -1 if i == 0 else 0, -1 if i == len(np_labels) - 1 else 0]
                if i > 0:
                    label[1] = np_labels[i] - np_labels[i - 1]

                if i < len(np_labels) - 1:
                    label[2] = np_labels[i] - np_labels[i + 1]

                labels.append(label)

            path = os.path.join(root, 'douyin_videos/partition_0000', vid + '.mp4')

            size = cfg.INPUT.SEQUENCE_LENGTH
            start = 0
            overlap = 5
            # split into overlapping chunks
            pad_frame_num = len(selected_indices)
            if len(selected_indices) < size:
                diff = size - len(selected_indices)
                selected_indices = selected_indices + [-1] * diff
                for _ in range(diff):
                    labels.append([-1, -1, -1])
                pad_frame_num = size

            assert len(labels) == len(selected_indices)
            while True:
                chunk_indices = selected_indices[start: start + size]
                chunk_labels = labels[start: start + size]
                record = {
                    'path': path,
                    'block_idx': chunk_indices,
                    'label': chunk_labels,
                    'vid': vid
                }
                annotations.append(record)

                if start + size >= pad_frame_num:
                    break
                start = start + size - overlap
                if start + size > pad_frame_num:
                    start = pad_frame_num - size

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(annotations, f)

    synchronize()
    with open(cache_path, 'rb') as f:
        annotations = pickle.load(f)

    if is_main_process():
        print(f'Loaded from {cache_path}')
    return annotations


def is_I_frame(frame_idx):
    if frame_idx == -1:
        return False
    assert frame_idx >= 1
    return (frame_idx - 1) % 12 == 0


# def preprocess_residual(res):
#     size = 20
#     res = (res * (127.5 / size)).astype(np.int32)
#     res += 128
#     res = (np.minimum(np.maximum(res, 0), 255)).astype(np.uint8)
#     res = cv2.resize(res, (224, 224), interpolation=cv2.INTER_LINEAR)
#     res = (res.astype(np.float32) / 255.0 - 0.5) / np.array([0.229, 0.224, 0.225])
#     res = np.transpose(res, (2, 0, 1)).astype(np.float32)
#     return torch.from_numpy(res)


def preprocess_residual(res):
    res = res.float().permute(2, 0, 1).contiguous().to(torch.float32)
    res = F.interpolate(res[None], size=(MV_SIZE, MV_SIZE), mode='bilinear', align_corners=False)[0]
    res = res / 255.0 - 0.5
    return res


def preprocess_y(y):
    """(H, W)"""
    y = y[None].float().contiguous().to(torch.float32)
    y = F.interpolate(y[None], size=(MV_SIZE, MV_SIZE), mode='bilinear', align_corners=False)[0]
    y = y / 255.0 - 0.5
    return y


def preprocess_motion_vector_old(mv):
    # Motion Vectors is (H W 2) format and (y x) order
    use_raw_coord = False
    if use_raw_coord:
        H, W, C = mv.shape
        scale_x = float(MV_SIZE) / W
        scale_y = float(MV_SIZE) / H
        mv = torch.from_numpy(mv).permute(2, 0, 1).to(torch.float32)[None]
        mv = F.interpolate(mv, size=(224, 224), mode='nearest')[0]
        mv[0] *= scale_y
        mv[1] *= scale_x
        mv *= -1
    else:
        size = 20
        mv = (mv * (127.5 / size))
        mv += 128
        mv = mv.permute(2, 0, 1)[None]
        mv = F.interpolate(mv, size=(MV_SIZE, MV_SIZE), mode='nearest')[0]
        mv = mv / 255.0 - 0.5
        # mv = (mv - 0.5) / 0.226
    return mv


def preprocess_motion_vector(mv):
    # x0,y0,x1,y1
    mv[:, :, :] *= (0.25 * 0.25)
    mv[:, :, 0::2] /= mv.shape[1]
    mv[:, :, 1::2] /= mv.shape[0]
    # assert torch.all(-1 <= mv) and torch.all(mv <= 1)
    mv = mv.permute(2, 0, 1)[None]
    mv = F.interpolate(mv, size=(MV_SIZE, MV_SIZE), mode='nearest')[0]
    return mv


def preprocess_origin_motion_vector(mv):
    # x0,y0,x1,y1
    mv[:, :, 0:4:2] *= (0.25 * 56 / mv.shape[1])
    mv[:, :, 1:4:2] *= (0.25 * 56 / mv.shape[0])
    # assert torch.all(-1 <= mv) and torch.all(mv <= 1)
    mv = mv.permute(2, 0, 1)[None]
    mv = F.interpolate(mv, size=(56, 56), mode='nearest')[0]
    return mv


class GEBDDataset(Dataset):
    def __init__(self, cfg, root, split, train=True):
        annotations = prepare_annotations(cfg, root, split)
        self._use_side_data = cfg.INPUT.USE_SIDE_DATA
        self._use_gan = cfg.MODEL.USE_GAN

        self.ann_path = os.path.join('data', f'k400_mr345_{split}_min_change_duration0.3.pkl')
        self.cfg = cfg
        self.root = root
        self.split = split
        self.train = train
        self.annotations = annotations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        def torch_transform(img):
            img = F.interpolate(img[None], size=(224, 224), mode='bilinear', align_corners=False).div(255)[0]
            img = torchvision.transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225], inplace=True)
            return img

        self.torch_transform = torch_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        vid = item['vid']
        block_indices = np.array(item['block_idx'], dtype='int32')
        folder = item['folder']

        video_path = os.path.join(self.root[:-len('frames')] + 'videos_hevc', folder + '.mp4')
        rgb_path = os.path.join(self.root, folder)
        compressed_path = os.path.join(self.root[:-len('frames')] + 'videos_hevc_info', folder)

        mv_list = None
        res_list = None
        # ref_mv_list = None
        origin_mv_list = None
        frame_mask = None
        rgb_frame_mask = None
        decode_order = None
        if self.cfg.INPUT.END_TO_END:
            imgs = torch.zeros(len(block_indices), 3, MV_SIZE, MV_SIZE, dtype=torch.float32)
            mv_list = torch.zeros(len(block_indices), 4 + 0, MV_SIZE, MV_SIZE, dtype=torch.float32)
            # ref_mv_list = torch.zeros(len(block_indices), 4, MV_SIZE, MV_SIZE, dtype=torch.float32)
            origin_mv_list = torch.zeros(len(block_indices), 6, MV_SIZE // 4, MV_SIZE // 4, dtype=torch.float32)
            res_list = torch.zeros(len(block_indices), 3, MV_SIZE, MV_SIZE, dtype=torch.float32)
            y_list = torch.zeros(len(block_indices), 1, MV_SIZE, MV_SIZE, dtype=torch.float32)

            frame_mask = [(1 if frame_idx >= 1 else 0) for frame_idx in block_indices]
            frame_indices = sorted(set([frame_idx - 1 for frame_idx in block_indices if frame_idx >= 1]))
            frame_mask = torch.tensor(frame_mask)
            img_ids = []

            rgb_frame_mask = torch.zeros_like(frame_mask)

            # video_reader = libde265.VideoReader(video_path,
            #                                     is_bitstream=video_path.endswith('.bin'),
            #                                     num_threads=2)
            # image_list = video_reader.read_all()
            # start = time.time()
            image_list = hevc_reader.read_video(video_path,
                                                -1,  # num_threads
                                                1,  # disable_deblocking
                                                1,  # disable_sao
                                                1,  # without_decoding
                                                1,  # without_reference
                                                frame_indices
                                                )
            # print('read from', video_path)
            # image_list = read_compressed_features(video_path)
            # print('cost {:.2f}'.format(time.time() - start))

            key_frames = []
            for info in image_list:
                if info['pict_type'] == 'I':
                    key_frames.append(info['frame_idx'] + 1)

            for key_idx in key_frames:
                if key_idx not in block_indices:
                    block_indices[np.argmin(np.abs(block_indices - key_idx))] = key_idx

            for i, frame_idx in enumerate(block_indices):
                if frame_idx != -1:
                    img_info = image_list[frame_idx - 1]
                    frame_type = img_info['pict_type']
                    if frame_type in ('I',):
                        # rgb = np.array(image_loader(os.path.join(rgb_path, 'image_{:05d}.jpg'.format(frame_idx))))
                        if 'rgb' in img_info:
                            rgb = img_info['rgb']
                            imgs[i] = self.torch_transform(torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32))
                            rgb_frame_mask[i] = 1

                    # mv0_x, mv0_y, mv1_x, mv1_y
                    # speed = torch.from_numpy(img_info['motion_vector'][:, :, 4:6]).float()
                    # speed = speed.permute(2, 0, 1)[None]
                    # speed = F.interpolate(speed, size=(MV_SIZE, MV_SIZE), mode='nearest')[0]
                    # speed = speed.sigmoid() - 0.5

                    mv = preprocess_motion_vector(torch.from_numpy(img_info['motion_vector'][:, :, :4]).float())
                    # ref_mv = preprocess_motion_vector(torch.from_numpy(img_info['ref_motion_vector'][:, :, :4]).float())
                    origin_mv = preprocess_origin_motion_vector(torch.from_numpy(img_info['ref_motion_vector'][:, :, :6]).float())
                    res = preprocess_residual(torch.from_numpy(img_info['residual']))
                    y = preprocess_y(torch.from_numpy(img_info['y']))

                    # mv0_x, mv0_y, mv1_x, mv1_y
                    # motion_vector = np.load(os.path.join(compressed_path, 'mv_{:05d}.npy'.format(frame_idx)))
                    # residual = np.array(image_loader(os.path.join(compressed_path, 'res_{:05d}.jpg'.format(frame_idx))))
                    # mv = preprocess_motion_vector(torch.from_numpy(motion_vector[:, :, :4]).float())
                    # res = preprocess_residual(torch.from_numpy(residual))

                    # mv_list[i] = torch.cat((mv, speed), dim=0)
                    mv_list[i] = mv
                    # ref_mv_list[i] = ref_mv
                    origin_mv_list[i] = origin_mv
                    res_list[i] = res
                    y_list[i] = y
                    img_ids.append(img_info['id'])

            decode_order = torch.full((len(block_indices),), fill_value=-1, dtype=torch.int32)
            decode_order[:len(img_ids)] = torch.from_numpy(np.argsort(img_ids))

        else:
            imgs = [self.transform(image_loader(os.path.join(self.root, folder, 'image_{:05d}.jpg'.format(i)))) for i in block_indices]
            imgs = torch.stack(imgs, dim=0)

        # print(rgb_frame_mask)
        sample = {
            'imgs': imgs,
            'labels': torch.tensor(item['label'], dtype=torch.int64),
            'vid': vid,
            'video_path': video_path,
        }
        if self.cfg.INPUT.END_TO_END:
            sample['frame_indices'] = torch.tensor(block_indices)
            sample['frame_mask'] = frame_mask
            sample['decode_order'] = decode_order
            sample['rgb_frame_mask'] = rgb_frame_mask
        else:
            current_idx = item['current_idx']
            sample['frame_idx'] = current_idx
            sample['path'] = os.path.join(self.root, folder, 'image_{:05d}.jpg'.format(current_idx))

        if self._use_side_data:
            sample['mv'] = mv_list
            # sample['ref_mv'] = ref_mv_list
            sample['origin_mv'] = origin_mv_list
            sample['res'] = res_list
            sample['y'] = y_list

        return sample


def augmentation(imgs):
    """(N, C, H ,W)"""
    if random.random() < 0.5:
        imgs = torchvision.transforms.functional.hflip(imgs)
    if random.random() < 0.2:
        imgs = torchvision.transforms.functional.rgb_to_grayscale(imgs, 3)

    # imgs = torchvision.transforms.AugMix()(imgs)
    return imgs


class ClipShotsDataset(Dataset):
    def __init__(self, cfg, split, train=True):
        if 'clipshots' in split:
            self.annotations = prepare_clipshots_annotations(cfg, split)
        elif 'beats' in split:
            self.annotations = prepare_beats_annotations(cfg, split)
            # if train and is_main_process():
            #     print(f'{split} before filer:', len(self.annotations))
            #     self.annotations = list(filter(lambda record: 1 in record['label'], self.annotations))
            #     print(f'{split} after filer:', len(self.annotations))
        elif 'QV' in split:
            self.annotations = prepare_QV_annotations(cfg, split)
        else:
            raise NotImplemented

        self.cfg = cfg
        self.split = split
        self.train = train
        self.image_size = cfg.INPUT.IMAGE_SIZE

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        def get_sample(index):
            try:
                item = self.annotations[index]
                vid = item['vid']
                block_indices = np.array(item['block_idx'])
                video_path = item['path']
                vr = VideoReader(video_path, ctx=cpu(0))
                valid_indices = block_indices[block_indices != -1]
                imgs = vr.get_batch(valid_indices.tolist()).permute((0, 3, 1, 2))

                if self.train:
                    imgs = augmentation(imgs)

                resize_mode = random.choice(['bilinear', 'nearest', 'area']) if self.train else 'bilinear'
                align_corners = False if resize_mode == 'bilinear' else None

                imgs = F.interpolate(imgs.to(torch.float32), size=(self.image_size, self.image_size), mode=resize_mode, align_corners=align_corners).div_(127.5) - 1.0
                # imgs = F.interpolate(imgs.to(torch.float32), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).div_(255)
                # imgs = torchvision.transforms.functional.normalize(imgs, mean=[0.485, 0.456, 0.406],
                #                                                    std=[0.229, 0.224, 0.225], inplace=True)
                if len(valid_indices) != len(block_indices):
                    imgs = torch.cat([imgs, torch.zeros(len(block_indices) - len(valid_indices), 3, self.image_size, self.image_size, dtype=torch.float32)], dim=0)
                frame_mask = torch.from_numpy(block_indices != -1)

                sample = {
                    'imgs': imgs,
                    'labels': torch.tensor(item['label'], dtype=torch.int64),
                    'vid': vid,
                    'video_path': video_path,
                    'frame_indices': torch.tensor(block_indices),
                    'frame_mask': frame_mask
                }
            except:
                sample = None
            return sample

        sample = get_sample(index)
        if sample is None:
            while sample is None:
                index = random.randint(0, len(self.annotations) - 1)
                sample = get_sample(index)
        return sample


class MeixueDataset(Dataset):
    def __init__(self, cfg, split, train=True):
        self.annotations = prepare_meixue_annotations(cfg, split)

        self.cfg = cfg
        self.split = split
        self.train = train
        self.image_size = cfg.INPUT.IMAGE_SIZE

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        vid = item['vid']
        block_indices = np.array(item['block_idx'])
        video_path = item['path']
        vr = VideoReader(video_path, ctx=cpu(0))
        valid_indices = block_indices[block_indices != -1]

        imgs = vr.get_batch(valid_indices.tolist()).permute((0, 3, 1, 2))
        imgs = F.interpolate(imgs.to(torch.float32), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).div_(255)
        imgs = torchvision.transforms.functional.normalize(imgs, mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225], inplace=True)
        if len(valid_indices) != len(block_indices):
            imgs = torch.cat([imgs, torch.zeros(len(block_indices) - len(valid_indices), 3, self.image_size, self.image_size, dtype=torch.float32)], dim=0)
        frame_mask = torch.from_numpy(block_indices != -1)

        sample = {
            'imgs': imgs,
            'labels': torch.tensor(item['label'], dtype=torch.float32),
            'vid': vid,
            'video_path': video_path,
            'frame_indices': torch.tensor(block_indices),
            'frame_mask': frame_mask
        }
        return sample
