# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time

import numpy as np
import torch
from mmengine import Config
from mmengine.config import DictAction
from mmengine.fileio import dump
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist

from mmseg.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the results will be dumped '
              'into the directory as json'))
    parser.add_argument('--repeat-times', type=int, default=1)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--cudnn-benchmark',
        action='store_true',
        default=None,
        help='enable torch.backends.cudnn.benchmark for faster inference '
        '(fixed input shape only). Default: keep config env setting.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='override test_dataloader.batch_size (throughput test).')
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        metavar=('H', 'W'),
        default=None,
        help='override test input resolution, e.g. --input-size 512 512. '
        'Also updates data_preprocessor size and Resize pipeline.')
    parser.add_argument(
        '--raw',
        action='store_true',
        help='skip mmseg predict post-processing, measure pure model '
        'forward (backbone + decode head) only.')
    parser.add_argument(
        '--cuda-graph',
        action='store_true',
        help='capture the model forward as a CUDA Graph and replay it '
        '(requires CUDA kernel path with fixed input shape; skips '
        'predict post-processing).')
    args = parser.parse_args()
    return args


def _apply_input_size(pipeline, size):
    """覆盖 test pipeline 的 Resize scale，并把 data_preprocessor 的
    size 一并调整。返回新 pipeline（对 mmengine ConfigDict 就地改）。"""
    h, w = size
    for t in pipeline:
        if t.get('type') in ('Resize', 'RandomResize'):
            t['scale'] = (w, h)   # mmseg scale=(W, H)
        if t.get('type') == 'RandomCrop':
            t['crop_size'] = (w, h)
    return pipeline


def main():
    args = parse_args()
    if args.repeat_times <= 0:
        raise ValueError('--repeat-times must be a positive integer')
    if args.log_interval <= 0:
        raise ValueError('--log-interval must be a positive integer')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # -- 推理参数覆盖（cudnn benchmark / batch size / input size）--
    if args.cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        print(f'cudnn.benchmark set to {args.cudnn_benchmark}')
    if args.batch_size is not None:
        cfg.test_dataloader.batch_size = args.batch_size
        print(f'test_dataloader.batch_size = {args.batch_size}')
    if args.input_size is not None:
        h, w = int(args.input_size[0]), int(args.input_size[1])
        cfg.test_dataloader.dataset.pipeline = _apply_input_size(
            cfg.test_dataloader.dataset.pipeline, (h, w))
        # 同步 data_preprocessor 的 pad 尺寸，避免输入被 padding 到旧尺寸
        pre = cfg.model.get('data_preprocessor')
        if pre is not None:
            pre['size'] = (h, w)
        # 诊断：打印修改后的 pipeline 与 preprocessor，确认参数生效
        _resize = [t for t in cfg.test_dataloader.dataset.pipeline
                   if t.get('type') in ('Resize', 'RandomResize')]
        print(f'test input size = ({h}, {w})')
        print(f'  pipeline Resize scale = {_resize}')
        print(f'  data_preprocessor.size = {pre.get("size") if pre else None}')

    init_default_scope(cfg.get('default_scope', 'mmseg'))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.work_dir is not None:
        mkdir_or_exist(osp.abspath(args.work_dir))
        json_file = osp.join(args.work_dir, f'fps_{timestamp}.json')
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mkdir_or_exist(osp.abspath(work_dir))
        json_file = osp.join(work_dir, f'fps_{timestamp}.json')

    repeat_times = args.repeat_times
    # cudnn.benchmark：默认关闭（保守），--cudnn-benchmark 时开启
    # （已在前面 args 处理时设置，这里不再硬编码覆盖）
    cfg.model.pretrained = None

    benchmark_dict = dict(config=args.config, unit='img / s')
    overall_fps_list = []
    # 单图延迟测试默认 batch_size=1；--batch-size 已在前面对 cfg 设置
    if args.batch_size is None:
        cfg.test_dataloader.batch_size = 1
    for time_index in range(repeat_times):
        print(f'Run {time_index + 1}:')
        # build the dataloader
        data_loader = Runner.build_dataloader(cfg.test_dataloader)
        if len(data_loader) == 0:
            raise RuntimeError('The test dataloader is empty')

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = MODELS.build(cfg.model)

        load_checkpoint(model, args.checkpoint, map_location='cpu')

        if torch.cuda.is_available():
            model = model.cuda()

        model = revert_sync_batchnorm(model)

        model.eval()

        # the first several iterations may be very slow so skip them
        pure_inf_time = 0
        total_iters = max(200, len(data_loader))
        num_warmup = min(5, total_iters - 1)
        data_iter = iter(data_loader)

        # ── CUDA Graph 捕获（--cuda-graph）────────────────────────────
        # 原理：把模型 forward 捕获成一张图，消除每次推理的 kernel launch
        # 与 Python 调度开销。要求：CUDA 核路径 + 固定输入形状。
        graph = None
        graph_input = None
        if args.cuda_graph and torch.cuda.is_available():
            # 检查必须使用 CUDA 核路径：torch 路径有动态形状（x[..., :max_len]）
            # 无法被 CUDA Graph 捕获，会捕获失败或结果错误。
            _bk = cfg.model.get('backbone', {})
            _backend = _bk.get('topp_flash_backend')
            if _backend not in ('cuda', 'cuda_forward'):
                raise RuntimeError(
                    '--cuda-graph 需要 model.backbone.topp_flash_backend=cuda '
                    '(torch 路径有动态路由形状，无法捕获)。'
                    f'当前 backend={_backend}')
            print('CUDA Graph: 捕获模型 forward...')
            # 取第一张输入确定形状（先 build dataloader 已存在）
            g_inputs = None
            g_samples = None
            for gi in range(3):  # 少量预热，触发 optimize_for_inference
                try:
                    gdata = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    gdata = next(data_iter)
                gdata = model.data_preprocessor(gdata, False)
                g_inputs = gdata['inputs']
                g_samples = gdata['data_samples']
                with torch.no_grad():
                    model(g_inputs, g_samples, mode='tensor')
            if not isinstance(g_inputs, torch.Tensor):
                raise RuntimeError('--cuda-graph 需要 inputs 是单张量')
            # 预热 CUDA 核（触发路由/Flash 核编译）
            for _ in range(5):
                with torch.no_grad():
                    model(g_inputs, g_samples, mode='tensor')
            torch.cuda.synchronize()

            # 静态输入缓冲：Graph 重放固定用这块内存
            graph_input = g_inputs.detach().clone()
            static_samples = g_samples
            # 用静态输入捕获
            for _ in range(2):
                with torch.no_grad():
                    model(graph_input, static_samples, mode='tensor')
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad():
                    model(graph_input, static_samples, mode='tensor')
            graph = g
            torch.cuda.synchronize()
            print('CUDA Graph: 捕获完成')

        # benchmark with enough batches and take the average
        for i in range(total_iters):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                data = next(data_iter)

            data = model.data_preprocessor(data, False)
            inputs = data['inputs']
            data_samples = data['data_samples']
            # 诊断：打印第一张真实输入张量的形状，确凿验证 --input-size 是否生效
            if i == 0 and isinstance(inputs, torch.Tensor):
                print(f'[diag] 实际输入 tensor shape = {tuple(inputs.shape)}')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                if args.cuda_graph:
                    graph_input.copy_(inputs)
                    graph.replay()
                elif args.raw:
                    # 纯模型 forward（backbone + decode head），跳过 predict 后处理
                    model(inputs, data_samples, mode='tensor')
                else:
                    model(inputs, data_samples, mode='predict')

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % args.log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(f'Done image [{i + 1:<3}/ {total_iters}], '
                          f'fps: {fps:.2f} img / s')

        if total_iters > num_warmup and pure_inf_time > 0:
            fps = (total_iters - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.2f} img / s\n')
            benchmark_dict[f'overall_fps_{time_index + 1}'] = round(fps, 2)
            overall_fps_list.append(fps)
        else:
            print(f'Warning: not enough iterations ({total_iters}) '
                  f'to compute FPS (need > {num_warmup})')
    benchmark_dict['average_fps'] = round(np.mean(overall_fps_list), 2)
    benchmark_dict['fps_variance'] = round(np.var(overall_fps_list), 4)
    print(f'Average fps of {repeat_times} evaluations: '
          f'{benchmark_dict["average_fps"]}')
    print(f'The variance of {repeat_times} evaluations: '
          f'{benchmark_dict["fps_variance"]}')
    dump(benchmark_dict, json_file, indent=4)


if __name__ == '__main__':
    main()
