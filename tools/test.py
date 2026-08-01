# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmseg.utils import register_all_modules


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
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
        help='override test_dataloader.batch_size.')
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        metavar=('H', 'W'),
        default=None,
        help='override test input resolution, e.g. --input-size 512 512.')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def apply_input_size(pipeline, size):
    """覆盖 test pipeline 的 Resize scale（对 mmengine ConfigDict 就地改）。"""
    h, w = size
    for t in pipeline:
        if t.get('type') in ('Resize', 'RandomResize'):
            t['scale'] = (w, h)
        if t.get('type') == 'RandomCrop':
            t['crop_size'] = (w, h)
    return pipeline


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir

    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if cfg.get('default_scope', None) is None:
        cfg.default_scope = 'mmseg'
    register_all_modules(init_default_scope=True)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # -- 推理参数覆盖 --
    if args.cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        print(f'cudnn.benchmark set to {args.cudnn_benchmark}')
    if args.batch_size is not None:
        cfg.test_dataloader.batch_size = args.batch_size
        print(f'test_dataloader.batch_size = {args.batch_size}')
    if args.input_size is not None:
        h, w = int(args.input_size[0]), int(args.input_size[1])
        cfg.test_dataloader.dataset.pipeline = apply_input_size(
            cfg.test_dataloader.dataset.pipeline, (h, w))
        pre = cfg.model.get('data_preprocessor')
        if pre is not None:
            pre['size'] = (h, w)
        print(f'test input size = ({h}, {w})')

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
