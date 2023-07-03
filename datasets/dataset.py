import math
import os
import pickle
from typing import List

import cv2
from videoio import read_video
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

from utils.distribute import synchronize, is_main_process


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
    filename = '{}-cache-fps{}-ds{}.pkl'.format(split, frame_per_side, f'_dynamic{ds}' if dynamic_downsample else ds)
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


def is_I_frame(frame_idx):
    if frame_idx == -1:
        return False
    assert frame_idx >= 1
    return (frame_idx - 1) % 12 == 0


def preprocess_residual(res):
    size = 20
    res = (res * (127.5 / size)).astype(np.int32)
    res += 128
    res = (np.minimum(np.maximum(res, 0), 255)).astype(np.uint8)
    res = cv2.resize(res, (224, 224), interpolation=cv2.INTER_LINEAR)
    res = (res.astype(np.float32) / 255.0 - 0.5) / np.array([0.229, 0.224, 0.225])
    res = np.transpose(res, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(res)


# def preprocess_motion_vector(mv):
#     mv += 128
#     mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)
#     resized_mv = np.stack([cv2.resize(mv[..., i], (224, 224), interpolation=cv2.INTER_LINEAR) for i in range(2)], axis=2)
#     resized_mv = resized_mv.astype(np.float32) / 255.0 - 0.5
#     resized_mv = np.transpose(resized_mv, (2, 0, 1)).astype(np.float32)
#     resized_mv = torch.from_numpy(resized_mv)
#     return resized_mv


def preprocess_motion_vector(mv):
    # mv += 128
    # mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)
    # resized_mv = np.stack([cv2.resize(mv[..., i], (224, 224), interpolation=cv2.INTER_LINEAR) for i in range(2)], axis=2)
    # resized_mv = resized_mv.astype(np.float32) / 255.0 - 0.5
    # resized_mv = np.transpose(resized_mv, (2, 0, 1)).astype(np.float32)
    # mv = torch.tensor(mv).permute(2, 0, 1).to(torch.float32)[None]
    # mv = F.interpolate(mv, size=(224, 224), mode='bilinear', align_corners=True)[0]

    mv += 128
    mv = mv.astype(np.uint8)
    resized_mv = np.stack([cv2.resize(mv[..., i], (224, 224), interpolation=cv2.INTER_LINEAR) for i in range(2)], axis=0)
    resized_mv -= 128
    resized_mv = resized_mv.astype(np.float32)

    return torch.from_numpy(resized_mv)


def load_side_data(video_path: str, frame_idxs: List):
    options = {'load_residuals': 1}
    frame_idxs.sort()
    options['target_list'] = frame_idxs

    sidedata_dict = {}
    # try:
    #     info_list = read_video(video_path, options)
    # except RuntimeError:
    #     info_list = []
    #     for idx in frame_idxs:
    #         info_list.append({
    #             'frame_number': idx,
    #         })

    info_list = read_video(video_path, options)

    # for info, idx in zip(info_list, frame_idxs):
    #    frame_idx = info['frame_number']
    #    assert frame_idx == idx

    for frame_idx in frame_idxs:
        info = info_list[frame_idx-1]
        if 'residuals' in info:
            res = preprocess_residual(info['residuals'])
        else:
            res = torch.zeros(3, 224, 224, dtype=torch.float32)

        if 'motion_vector' in info:
            mv = preprocess_motion_vector(info['motion_vector'])
        else:
            mv = torch.zeros(2, 224, 224, dtype=torch.float32)

        sidedata_dict[frame_idx] = {
            'res': res,
            'mv': mv,
        }

    return sidedata_dict


class GEBDDataset(Dataset):
    def __init__(self, cfg, root, split, train=True):
        annotations = prepare_annotations(cfg, root, split)
        self._use_side_data = cfg.INPUT.USE_SIDE_DATA
        self._load_mv_res = cfg.MODEL.NAME in ('CompressedGEBDModel', 'E2ECompressedGEBDModel')

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

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        vid = item['vid']
        block_idx = item['block_idx']
        folder = item['folder']

        video_path = os.path.join(self.root[:-len('frames')] + 'videos_mpeg4', folder + '.mp4')

        mv_list = None
        res_list = None
        frame_mask = None
        if self._use_side_data:
            # side data baseline
            imgs = [(self.transform(image_loader(os.path.join(self.root, folder, 'image_{:05d}.jpg'.format(frame_idx))))
                     if is_I_frame(frame_idx) else torch.zeros(3, 224, 224, dtype=torch.float32))
                    for frame_idx in block_idx]

            if self._load_mv_res:
                mv_list = []
                res_list = []
                frame_mask = [(1 if frame_idx >= 1 else 0) for frame_idx in block_idx]
                side_data_frame_idxs = [frame_idx for frame_idx in block_idx if frame_idx != -1 and not is_I_frame(frame_idx)]

                # side_data_frame_idxs = [i for i in block_idx if not is_I_frame(i)]
                sidedata_dict = load_side_data(video_path, side_data_frame_idxs)
                for frame_idx in block_idx:
                    if is_I_frame(frame_idx) or frame_idx == -1:
                        mv = torch.zeros(2, 224, 224, dtype=torch.float32)
                        res = torch.zeros(3, 224, 224, dtype=torch.float32)
                    else:
                        mv = sidedata_dict[frame_idx]['mv']
                        res = sidedata_dict[frame_idx]['res']

                    mv_list.append(mv)
                    res_list.append(res)

        else:
            imgs = [self.transform(image_loader(os.path.join(self.root, folder, 'image_{:05d}.jpg'.format(i)))) for i in block_idx]

        imgs = torch.stack(imgs, dim=0)
        sample = {
            'imgs': imgs,
            'labels': torch.tensor(item['label'], dtype=torch.int64),
            'vid': vid,
            'video_path': video_path,
        }
        if self.cfg.INPUT.END_TO_END:
            sample['frame_indices'] = torch.tensor(block_idx)
            # sample['frame_mask'] = torch.tensor(frame_mask)
            # sample['time_pos'] = torch.tensor(item['time_pos'], dtype=torch.float32)
        else:
            current_idx = item['current_idx']
            sample['frame_idx'] = current_idx
            sample['path'] = os.path.join(self.root, folder, 'image_{:05d}.jpg'.format(current_idx))

        if self._use_side_data and self._load_mv_res:
            sample['mv'] = torch.stack(mv_list, dim=0)
            sample['res'] = torch.stack(res_list, dim=0)
            sample['frame_mask'] = torch.tensor(frame_mask)

        return sample
