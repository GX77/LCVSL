import argparse
import json
import os
import pickle
import time
from collections import defaultdict, OrderedDict
from contextlib import suppress
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from datasets import build_dataloader
from modeling import cfg, build_model
from solver import build_optimizer
from utils.distribute import synchronize, all_gather, is_main_process
from utils.eval import eval_f1
from utils.misc import SmoothedValue, MetricLogger


def make_inputs(inputs, device):
    keys = ['imgs', 'mv', 'ref_mv', 'origin_mv', 'res', 'frame_mask', 'decode_order', 'video_path', 'rgb_frame_mask', 'y']
    results = {}
    if isinstance(inputs, dict):
        for key in keys:
            if key in inputs:
                val = inputs[key]
                if isinstance(val, torch.Tensor):
                    val = val.to(device)
                results[key] = val
    elif isinstance(inputs, list):
        targets = defaultdict(list)
        for item in inputs:
            for key in keys:
                if key in item:
                    val = item[key]
                    targets[key].append(val)

        for key in targets:
            results[key] = torch.stack(targets[key], dim=0).to(device)
    else:
        raise NotImplementedError
    return results


def make_targets(cfg, inputs, device):
    if False and cfg.INPUT.END_TO_END:
        targets = []
        for item in inputs:
            targets.append({
                'labels': item['labels'].to(device),
                'time_pos': item['time_pos'].view(-1, 1).to(device),
            })
    else:
        targets = inputs['labels'].to(device)

    return targets


def train_one_epoch(cfg, args, model, device, optimizer, data_loader, summary_writer, auto_cast, loss_scaler, epoch):
    model.train()

    start = time.time()
    for i, inputs in enumerate(data_loader):
        # ------------ inputs ----------
        samples = make_inputs(inputs, device)
        targets = make_targets(cfg, inputs, device)
        # ------------ inputs ----------

        model_start = time.time()
        with auto_cast():
            loss_dict = model(samples, targets)
            total_loss = sum(loss_dict.values())

        # ------------ training operations ----------
        optimizer.zero_grad()
        if cfg.SOLVER.AMPE:
            loss_scaler.scale(total_loss).backward()

            # print('^'*88)
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(name)

            if cfg.SOLVER.CLIP_GRAD > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD)
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            total_loss.backward()
            if cfg.SOLVER.CLIP_GRAD > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD)
            optimizer.step()
        # ------------ training operations ----------

        # ------------ logging ----------
        if is_main_process():
            summary_writer.global_step += 1

            if len(loss_dict) > 1:
                summary_writer.update(**loss_dict)

            summary_writer.update(lr=optimizer.param_groups[0]['lr'], total_loss=total_loss,
                                  total_time=time.time() - start, model_time=time.time() - model_start)
            start = time.time()

            speed = summary_writer.total_time.avg
            eta = str(timedelta(seconds=int((len(data_loader) - i - 1) * speed)))
            if i % 10 == 0:
                print('Epoch{:02d} ({:04d}/{:04d}): {}, Eta:{}'.format(epoch,
                                                                       i,
                                                                       len(data_loader),
                                                                       str(summary_writer),
                                                                       eta
                                                                       ), flush=True)


@torch.no_grad()
def validate(cfg, model, device, data_loader):
    if cfg.INPUT.END_TO_END:
        return validate_end_to_end(cfg, model, device, data_loader)

    model_pred_dict = {}
    model.eval()
    start_time = time.time()
    num_frames = 0
    for i, inputs in enumerate(tqdm(data_loader, total=len(data_loader))):
        samples = make_inputs(inputs, device)
        num_frames += samples['imgs'].shape[0]

        vids = inputs['vid']
        frame_idxs = inputs['frame_idx']
        scores = model(samples)

        scores = scores.cpu().numpy()
        for vid, frame_idx, score in zip(vids, frame_idxs, scores):
            if vid not in model_pred_dict.keys():
                model_pred_dict[vid] = {}
                model_pred_dict[vid]['frame_idx'] = []
                model_pred_dict[vid]['scores'] = []
            model_pred_dict[vid]['frame_idx'].append(frame_idx)
            model_pred_dict[vid]['scores'].append(score)

        # if num_frames >= 10000:
        #     break

    synchronize()
    metrics = {}
    data_list = all_gather(model_pred_dict)
    if not is_main_process():
        metrics['F1'] = 0.00
        return metrics
    total_time = time.time() - start_time
    print('Cost {:.2f}s for evaluating {} videos, {:.4f}s/video'.format(total_time, len(data_loader.dataset), total_time / len(data_loader.dataset)))
    print('{:.9f}ms/frame'.format(total_time * 1000 / num_frames))

    model_pred_dict = defaultdict(dict)
    for p in data_list:
        for vid in p:
            if 'frame_idx' not in model_pred_dict[vid]:
                model_pred_dict[vid]['frame_idx'] = []
                model_pred_dict[vid]['scores'] = []

            model_pred_dict[vid]['frame_idx'].extend(p[vid]['frame_idx'])
            model_pred_dict[vid]['scores'].extend(p[vid]['scores'])

    for vid in model_pred_dict:
        frame_idx = np.array(model_pred_dict[vid]['frame_idx'])
        scores = np.array(model_pred_dict[vid]['scores'])
        _, indices = np.unique(frame_idx, return_index=True)
        frame_idx = frame_idx[indices]
        scores = scores[indices]

        indices = np.argsort(frame_idx)
        model_pred_dict[vid]['frame_idx'] = frame_idx[indices].tolist()
        model_pred_dict[vid]['scores'] = scores[indices].tolist()

    gt_path = f'data/k400_mr345_{data_loader.dataset.split}_min_change_duration0.3.pkl'
    f1, rec, prec = eval_f1(model_pred_dict, gt_path, threshold=cfg.TEST.THRESHOLD)
    print('F1: {:.4f}, Rec: {:.4f}, Prec: {:.4f}'.format(f1, rec, prec))
    metrics['F1'] = f1
    metrics['Rec'] = rec
    metrics['Prec'] = prec
    return metrics


@torch.no_grad()
def validate_end_to_end(cfg, model, device, data_loader):
    metrics = OrderedDict()
    if cfg.TEST.PRED_FILE:
        if not is_main_process():
            return metrics
        with open(cfg.TEST.PRED_FILE, 'rb') as f:
            results_dict = pickle.load(f)
        print(f'Load results from {cfg.TEST.PRED_FILE}.')
    else:
        model_pred_dict = defaultdict(dict)
        model.eval()

        model_time_cost = 0
        backbone_time_cost = 0
        head_time_cost = 0
        num_frames = 0
        all_start = time.time()
        for i, inputs in enumerate(tqdm(data_loader, total=len(data_loader))):
            samples = make_inputs(inputs, device)
            num_frames += (samples['imgs'].shape[0] * samples['imgs'].shape[1])
            start_time = time.time()
            if args.device == 'cuda':
                torch.cuda.synchronize()
            outputs, time_cost = model(samples)  # (b, t)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            model_time_cost += (time.time() - start_time)
            head_time_cost += time_cost['head']
            backbone_time_cost += time_cost['backbone']
            for batch_idx, frame_indices in enumerate(inputs['frame_indices']):
                vid = inputs['vid'][batch_idx]
                video_path = inputs['video_path'][batch_idx]
                scores = outputs[batch_idx]
                # labels = inputs['labels'][batch_idx]
                # if is_main_process():
                #     labels = inputs['labels'][batch_idx]
                if 'frame_idx' not in model_pred_dict[vid]:
                    model_pred_dict[vid]['frame_idx'] = []
                    model_pred_dict[vid]['scores'] = []

                model_pred_dict[vid]['frame_idx'].extend(frame_indices.tolist())
                model_pred_dict[vid]['scores'].extend(scores.tolist())

            # if num_frames >= 10000:
            #     break

        synchronize()
        data_list = all_gather(model_pred_dict)
        if not is_main_process():
            metrics['F1'] = 0.00
            return metrics
        all_total_time = time.time() - all_start
        print('Cost {:.2f}s for evaluating {} videos, {:.4f}s/video'.format(all_total_time, len(data_loader.dataset), all_total_time / len(data_loader.dataset)))
        print('Model {:.9f}ms/frame'.format(model_time_cost * 1000 / num_frames))
        print('Backbone {:.9f}ms/frame'.format(backbone_time_cost * 1000 / num_frames))
        print('Head {:.9f}ms/frame'.format(head_time_cost * 1000 / num_frames))
        print('All   {:.9f}ms/frame'.format(all_total_time * 1000 / num_frames))

        # model_pred_dict = {}
        # for p in data_list:
        #     model_pred_dict.update(p)
        # print(len(data_list))
        # print(data_list)
        model_pred_dict = defaultdict(dict)
        for p in data_list:
            for vid in p:
                for frame_idx, score in zip(p[vid]['frame_idx'], p[vid]['scores']):
                    if frame_idx not in model_pred_dict[vid]:
                        model_pred_dict[vid][frame_idx] = []

                    model_pred_dict[vid][frame_idx].append(score)

        # print('model_pred_dict', len(model_pred_dict))
        # print(model_pred_dict)

        results_dict = defaultdict(dict)
        for vid in model_pred_dict:
            frame_idx_list = []
            scores_list = []
            for frame_idx in model_pred_dict[vid]:
                frame_idx_list.append(frame_idx)
                scores_list.append(np.mean(model_pred_dict[vid][frame_idx]))
            frame_idx = np.array(frame_idx_list)
            scores = np.array(scores_list)

            #
            valid_frame_mask = frame_idx != -1
            frame_idx = frame_idx[valid_frame_mask]
            scores = scores[valid_frame_mask]

            indices = np.argsort(frame_idx)
            results_dict[vid]['frame_idx'] = frame_idx[indices].tolist()
            results_dict[vid]['scores'] = scores[indices].tolist()

    if not cfg.TEST.PRED_FILE:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, 'model_pred_dict_{}.pkl'.format(time.strftime('%Y%m%d%H%M%S')))
        with open(save_path, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f'Saved results to {save_path}.')

    if args.all_thres:
        rel_dis_thres = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    else:
        rel_dis_thres = [0.05]

    gt_path = f'data/k400_mr345_{data_loader.dataset.split}_min_change_duration0.3.pkl'
    results, pred_dict, gt_dict = eval_f1(results_dict, gt_path,
                                          threshold=cfg.TEST.THRESHOLD,
                                          return_pred_dict=True,
                                          rel_dis_thres=rel_dis_thres)
    list_rec = []
    list_prec = []
    list_f1 = []

    for th in rel_dis_thres:
        f1, rec, prec = results[th]
        list_rec.append(rec)
        list_prec.append(prec)
        list_f1.append(f1)

    headers = rel_dis_thres + ['Avg']

    avg_rec = np.mean(list_rec)
    avg_prec = np.mean(list_prec)
    avg_F1 = np.mean(list_f1)

    tabulate_data = [
        ['Recall'] + list_rec + [avg_rec],
        ['Precision'] + list_prec + [avg_prec],
        ['F1'] + list_f1 + [avg_F1],
    ]
    # print(f'Results for {data_loader.dataset.name}_{data_loader.dataset.split}:')
    print(tabulate(tabulate_data, headers=headers, floatfmt='.4f'))

    f1, rec, prec = results[0.05]
    print('F1@0.05: {:.4f}, Rec: {:.4f}, Prec: {:.4f}'.format(f1, rec, prec))
    metrics['F1'] = f1
    metrics['Rec'] = rec
    metrics['Prec'] = prec

    with open('GEBD_pred.json', 'w') as f:
        json.dump(pred_dict, f)
    return metrics


def main(cfg, args):
    train_data_loader = build_dataloader(cfg, args, cfg.DATASETS.TRAIN, is_train=True)
    val_data_loader = build_dataloader(cfg, args, cfg.DATASETS.TEST, is_train=False)

    device = torch.device(args.device)

    model = build_model(cfg).to(device)

    if is_main_process():
        print(model)

    output_dir = '_'.join([cfg.OUTPUT_DIR,
                           cfg.MODEL.BACKBONE.NAME,
                           '{}x{}'.format(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE),
                           cfg.MODEL.TEMPORAL_MODEL,
                           'group_similarity' if cfg.MODEL.USE_GROUP_SIMILARITY else 'no_group_similarity',
                           ])
    if cfg.MODEL.PRETRAINED:
        output_dir += '_pretrained'
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    start_epoch = -1
    if args.resume:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        start_epoch = state_dict['epoch']
        if is_main_process():
            print('Loaded from {}, Epoch: {}'.format(args.resume, start_epoch), flush=True)

    if args.test_only:
        validate(cfg, model, device, val_data_loader)
        return

    with open(os.path.join(args.output_dir, 'config.yml'), 'w') as f:
        f.write(cfg.dump())

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    optimizer = build_optimizer(cfg, [p for p in model.parameters() if p.requires_grad], train_data_loader)
    scheduler = MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES)
    if args.resume:
        for name, obj in [('optimizer', optimizer), ('scheduler', scheduler)]:
            if name in state_dict:
                obj.load_state_dict(state_dict[name])
                if is_main_process():
                    print('Loaded {} from {}'.format(name, args.resume), flush=True)

    summary_writer = MetricLogger(log_dir=os.path.join(output_dir, 'logs')) if is_main_process() else None
    if summary_writer is not None:
        summary_writer.add_meter('lr', SmoothedValue(fmt='{value:.5f}'))
        summary_writer.add_meter('total_time', SmoothedValue(fmt='{avg:.3f}s'))
        summary_writer.add_meter('model_time', SmoothedValue(fmt='{avg:.3f}s'))

    auto_cast = torch.cuda.amp.autocast if cfg.SOLVER.AMPE else suppress
    loss_scaler = torch.cuda.amp.GradScaler() if cfg.SOLVER.AMPE else None

    for epoch in range(start_epoch + 1, cfg.SOLVER.MAX_EPOCHS):
        train_one_epoch(cfg, args, model, device, optimizer, train_data_loader, summary_writer, auto_cast, loss_scaler, epoch)
        metrics = validate(cfg, model, device, val_data_loader)
        scheduler.step()

        if is_main_process():
            model_state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            save_path = os.path.join(output_dir, f'model_epoch{epoch:02d}.pth')
            torch.save({
                'model': model_state_dict,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'metrics': metrics
            }, save_path)
            with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
                s = ', '.join(map(lambda kv: f'{kv[0]}: {kv[1]:.4f}', metrics.items()))
                f.write('Epoch: {:02d}, {}\n'.format(epoch, s))
            print('Saved to {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help="path to config file", type=str, default="config/end_to_end_sidedata_mv_res.yaml")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--test-only", default=False)
    parser.add_argument("--all-thres", default=True, help='test using all thresholds [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.local_rank:
        args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1

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
