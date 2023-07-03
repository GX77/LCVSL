import json

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def predictions_to_scenes(predictions):
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # just fix if all predictions are 1
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)


def convert_gt_scene(gt_data):
    scene_dict = {}
    for k in gt_data:
        n_frames = int(gt_data[k]['frame_num'])
        translations = np.array(gt_data[k]["transitions"])
        plus = 0
        if len(translations) == 0:
            scenes = np.array([[0, n_frames - 1]])
        else:
            scene_ends_zeroindexed = translations[:, 0] + plus
            scene_starts_zeroindexed = translations[:, 1] + plus
            scene_starts_zeroindexed = np.concatenate([[0], scene_starts_zeroindexed])
            scene_ends_zeroindexed = np.concatenate([scene_ends_zeroindexed, [n_frames - 1]])
            scenes = np.stack([scene_starts_zeroindexed, scene_ends_zeroindexed], 1)

        scene_dict[k] = scenes
    return scene_dict


def evaluate_scenes(gt_scenes, pred_scenes, return_mistakes=False, n_frames_miss_tolerance=2):
    """
    Adapted from: https://github.com/gyglim/shot-detection-evaluation
    The original based on: http://imagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19
    n_frames_miss_tolerance:
        Number of frames it is possible to miss ground truth by, and still being counted as a correct detection.
    Examples of computation with different tolerance margin:
    n_frames_miss_tolerance = 0
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[5.5, 5.5]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[5.5, 5.5]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[4.5, 4.5]] -> MISS
    n_frames_miss_tolerance = 1
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[5.0, 6.0]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[5.0, 6.0]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[4.0, 5.0]] -> HIT
      gt_scenes:   [[0, 3], [4, 9]] -> gt_trans:   [[3.0, 4.0]] -> MISS
    n_frames_miss_tolerance = 2
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[4.5, 6.5]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[4.5, 6.5]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[3.5, 5.5]] -> HIT
      gt_scenes:   [[0, 3], [4, 9]] -> gt_trans:   [[2.5, 4.5]] -> HIT
      gt_scenes:   [[0, 2], [3, 9]] -> gt_trans:   [[1.5, 3.5]] -> MISS
    """

    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])

    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)

    i, j = 0, 0
    tp, fp, fn = 0, 0, 0
    fp_mistakes, fn_mistakes = [], []

    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1
            fn_mistakes.append(gt_trans[i])
            i += 1
        elif i == len(gt_trans):
            fp += 1
            fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            fn_mistakes.append(gt_trans[i])
            i += 1
        else:
            i += 1
            j += 1
            tp += 1

    if tp + fp != 0:
        p = tp / (tp + fp)
    else:
        p = 0

    if tp + fn != 0:
        r = tp / (tp + fn)
    else:
        r = 0

    if p + r != 0:
        f1 = (p * r * 2) / (p + r)
    else:
        f1 = 0

    assert tp + fn == len(gt_trans)
    assert tp + fp == len(pred_trans)

    if return_mistakes:
        return p, r, f1, (tp, fp, fn), fp_mistakes, fn_mistakes
    return p, r, f1, (tp, fp, fn)


def evaluate(predicts, gt_path=None, threshold=0.5, gt_data=None, tol=2):
    total_stats = {"tp": 0, "fp": 0, "fn": 0}
    if gt_data is None:
        with open(gt_path) as f:
            gt_data = json.load(f)

    gt_data = convert_gt_scene(gt_data)
    for vid in predicts:
        p = predicts[vid]
        predict_indices = predictions_to_scenes((np.array(p['scores']) >= threshold).astype(np.uint8))
        scenes = []
        for s, e in predict_indices:
            scenes.append([p['frame_idx'][s], p['frame_idx'][e]])
        scenes = np.array(scenes, dtype=np.int32)

        gt_scene = gt_data[vid]

        _, _, _, (tp, fp, fn), fp_mistakes, fn_mistakes = evaluate_scenes(gt_scene, scenes, return_mistakes=True, n_frames_miss_tolerance=tol)

        total_stats["tp"] += tp
        total_stats["fp"] += fp
        total_stats["fn"] += fn

    p = total_stats["tp"] / (total_stats["tp"] + total_stats["fp"] + 1e-12)
    r = total_stats["tp"] / (total_stats["tp"] + total_stats["fn"] + 1e-12)
    f1 = (p * r * 2) / (p + r + 1e-12)

    return p, r, f1


# ------------------------------------------------------------------------------------------------------------------------------------------------
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
            # center = int(np.mean(internals))
            # bdy_indices_in_video.append(seq_indices[center])
            bdy_indices_in_video.append([seq_indices[internals[0]], seq_indices[internals[-1]]])
    return bdy_indices_in_video


def real_eval(pred_dict, gt_dict, tol=2, threshold=0.4):
    assert tol >= 0

    total_num_predictions = 0
    total_num_gts = 0

    tp, fp, fn = 0, 0, 0
    fp_mistakes, fn_mistakes = [], []
    for vid in pred_dict:
        pred = pred_dict[vid]
        pred_idx = np.array(get_idx_from_score_by_threshold(threshold=threshold,
                                                            seq_indices=pred['frame_idx'],
                                                            seq_scores=pred['scores']))

        gt_trans = np.array(gt_dict[vid]['transitions'])

        total_num_predictions += len(pred_idx)
        total_num_gts += len(gt_trans)

        i, j = 0, 0
        while i < len(gt_trans) or j < len(pred_idx):
            if j == len(pred_idx):
                fn += 1
                fn_mistakes.append(gt_trans[i])
                i += 1
            elif i == len(gt_trans):
                fp += 1
                fp_mistakes.append(pred_idx[j])
                j += 1
            elif gt_trans[i, 0] - pred_idx[j, 1] > tol:
                fp += 1
                fp_mistakes.append(pred_idx[j])
                j += 1
            elif pred_idx[j, 0] - gt_trans[i, 1] > tol:
                fn += 1
                fn_mistakes.append(gt_trans[i])
                i += 1
            else:
                i += 1
                j += 1
                tp += 1

    assert tp + fn == total_num_gts
    assert tp + fp == total_num_predictions

    if tp + fp != 0:
        p = tp / (tp + fp)
    else:
        p = 0

    if tp + fn != 0:
        r = tp / (tp + fn)
    else:
        r = 0

    if p + r != 0:
        f1 = (p * r * 2) / (p + r)
    else:
        f1 = 0

    return p, r, f1
