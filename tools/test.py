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
    parser.add_argument(
        '--cuda-graph',
        action='store_true',
        help='capture the model forward as a CUDA Graph on first test image '
        'and replay it afterwards (requires topp_flash_backend=cuda and '
        'fixed input shape). predict post-processing runs outside the graph.')
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


class _CudaGraphPredictWrapper(torch.nn.Module):
    """用 CUDA Graph 重放替代 predict 的 GPU forward。

    第一次调用时捕获 model._forward 成图，之后每次 replay；图外补
    predict_by_feat(resize) + postprocess_result，与 predict 语义一致。
    内部 model 作为子模块注册，显式转发 nn.Module 关键接口，保证
    mmengine runner 依赖的 eval()/train()/state_dict() 等可用。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._graph = None
        self._graph_input = None
        self._graph_output = None
        self._captured = False

    # ── 转发 nn.Module 关键接口到内部 model ──
    def eval(self, *args, **kwargs):
        return self.model.eval(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.model.train(*args, **kwargs)

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        return self.model.cuda(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def test_step(self, data):
        """复刻 mmengine BaseModel.test_step：data_preprocessor → predict。

        data_preprocessor 返回 dict(keys=('inputs','data_samples'))，
        然后走 CUDA Graph 重放的 predict 路径。
        """
        data = self.model.data_preprocessor(data, False)
        return self._predict(data['inputs'], data['data_samples'])

    def val_step(self, data):
        return self.test_step(data)

    def _ensure_captured(self, inputs, data_samples):
        if self._captured:
            return
        g_inputs = inputs.detach().clone()
        # 预热：触发 optimize_for_inference + CUDA 核编译
        with torch.no_grad():
            for _ in range(8):
                self.model._forward(inputs, data_samples)
        torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(3):
                self.model._forward(g_inputs, data_samples)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                out = self.model._forward(g_inputs, data_samples)
        self._graph = g
        self._graph_input = g_inputs
        self._graph_output = out
        self._captured = True
        torch.cuda.synchronize()
        print('CUDA Graph: 捕获完成')

    def __call__(self, inputs, data_samples=None, mode='predict'):
        if mode != 'predict':
            return self.model(inputs, data_samples, mode=mode)
        return self._predict(inputs, data_samples)

    def predict(self, inputs, data_samples=None):
        """兼容 runner 直接调 model.predict(...) 的路径。"""
        return self._predict(inputs, data_samples)

    def _predict(self, inputs, data_samples=None):
        self._ensure_captured(inputs, data_samples)
        with torch.no_grad():
            self._graph_input.copy_(inputs)
            self._graph.replay()
            torch.cuda.synchronize()
            if data_samples is not None:
                metas = [s.metainfo for s in data_samples]
            else:
                metas = [dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])] * inputs.shape[0]
            seg_logits = self.model.decode_head.predict_by_feat(
                self._graph_output, metas)
            return self.model.postprocess_result(seg_logits, data_samples)


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

    # --cuda-graph：包装模型 predict 用 CUDA Graph 重放 GPU forward
    if args.cuda_graph:
        if not torch.cuda.is_available():
            raise RuntimeError('--cuda-graph 需要 CUDA 设备')
        _bk = cfg.model.get('backbone', {})
        _backend = _bk.get('topp_flash_backend')
        if _backend not in ('cuda', 'cuda_forward'):
            raise RuntimeError(
                '--cuda-graph 需要 model.backbone.topp_flash_backend=cuda '
                '(torch 路径有动态路由形状，无法捕获)。'
                f'当前 backend={_backend}')
        # 手动加载权重到原模型，避免 runner.test() 内部重复加载时
        # wrapper 拦截 _load_from_state_dict 而崩溃。
        from mmengine.runner import load_checkpoint as _lc
        _lc(runner.model, args.checkpoint, map_location='cpu')
        runner.model = _CudaGraphPredictWrapper(runner.model)
        print('[test] CUDA Graph wrapper 已启用，等待首次推理捕获...')
        # 关闭 runner 的自动加载
        runner.load_from = None
        runner._load_from = None

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
