import glob

from pickle5 import pickle
from skvideo.utils import first
from .hevc_feature_decoder import HevcFeatureReader
import time
import numpy as np
import cv2
import os
import PIL.Image as Image
import argparse
import json
import tqdm
import torch.multiprocessing as mp


def read_compressed_features(input_mp4_name):
    timeStarted = time.time()
    reader = HevcFeatureReader(input_mp4_name, nb_frames=None, n_parallel=1)
    num_frames = reader.getFrameNums()
    decode_order = reader.getDecodeOrder()
    width, height = reader.getShape()

    frame_idx = 0
    frame_types = {
        0: 'I',
        1: 'P',
        2: 'B',
    }

    img_list = []
    for feature in reader.nextFrame():
        img_info = {
            'frame_idx': frame_idx,
            'width': width,
            'height': height,
            'pict_type': frame_types[feature[0]],
            'rgb': np.array(feature[2]),
            'residual': np.array(feature[10]),
            'motion_vector': np.stack([feature[3], feature[4], feature[5], feature[6], feature[7], feature[8]], axis=-1),
        }
        img_list.append(img_info)

        frame_idx += 1
        # frame_type.append(feature[0])
        # quadtree_stru.append(feature[1])
        # rgb.append(feature[2])
        # mv_x_L0.append(feature[3])
        # mv_y_L0.append(feature[4])
        # mv_x_L1.append(feature[5])
        # mv_y_L1.append(feature[6])
        # ref_off_L0.append(feature[7])
        # ref_off_L1.append(feature[8])
        # bit_density.append(feature[9])
        # residual.append(feature[10])

    reader.close()
    return img_list


def convert(local_rank, items, total_rank, f_root):
    items = items[local_rank::total_rank]
    for cat_name, v_name, video_path in tqdm.tqdm(items, total=len(items)):
        output_dir = os.path.join(f_root, cat_name, v_name)
        if os.path.exists(output_dir):
            continue
        try:
            img_list = read_compressed_features(video_path)
        except:
            print('Error: ', video_path)
            continue
        os.makedirs(output_dir, exist_ok=True)
        frame_types = []
        for info in img_list:
            frame_types.append(info['pict_type'])
            np.save(os.path.join(output_dir, 'mv_{:05d}'.format(info['frame_idx'] + 1)), info['motion_vector'])
            # np.save(os.path.join(output_dir, 'res_{:05d}'.format(info['frame_idx'] + 1)), info['residual'])
            cv2.imwrite(os.path.join(output_dir, 'res_{:05d}.jpg'.format(info['frame_idx'] + 1)), info['residual'])

        with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump({
                'frame_types': frame_types
            }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--v-root', type=str, required=True)
    parser.add_argument('--total-rank', type=int, default=32)
    args = parser.parse_args()
    v_root = args.v_root
    total_rank = args.total_rank

    items = []
    output_dir = '/mnt/bn/hevc-understanding/datasets/GEBD/GEBD_val_videos_hevc_info'
    for cat_name in list(os.listdir(v_root)):
        cat_dir = os.path.join(v_root, cat_name)
        video_paths = glob.glob(os.path.join(cat_dir, '*.mp4'))

        for video_path in video_paths:
            v_name = os.path.basename(video_path)[0:-4]
            items.append((cat_name, v_name, video_path))

    print('Total: ', len(items))
    if total_rank <= 1:
        convert(0, items, total_rank, output_dir)
    else:
        mp.spawn(convert, (items, total_rank, output_dir), nprocs=total_rank, daemon=False)
