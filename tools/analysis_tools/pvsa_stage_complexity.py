import argparse
import tempfile
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope

import mmseg.models  # noqa: F401
from mmseg.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile PVSA backbone stage complexity.')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    parser.add_argument(
        '--device',
        default='cuda',
        help='cuda or cpu')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options')
    return parser.parse_args()


def _input_shape(shape):
    if len(shape) == 1:
        return 3, shape[0], shape[0]
    if len(shape) == 2:
        return 3, shape[0], shape[1]
    raise ValueError('invalid input shape')


def _count_params(module, prefixes):
    total = 0
    for name, param in module.named_parameters():
        if any(name == prefix or name.startswith(prefix + '.')
               for prefix in prefixes):
            total += param.numel()
    return total


def _sum_flops(by_module, prefixes):
    return sum(float(by_module.get(prefix, 0.0)) for prefix in prefixes)


def _format_flops(value):
    return f'{value / 1e6:.2f}M'


def _format_params(value):
    return f'{value / 1e6:.3f}M'


def _stage_prefixes(backbone, stage):
    fam_stages = set(getattr(backbone, 'fam_stages', (0, 1, 2, 3)))
    fusion_stages = set(getattr(backbone, 'fusion_stages', (0, 1, 2, 3)))
    prefixes = {
        'cnn': [f'downsample_layers2.{stage}'],
        'transformer': [f'downsample_layers.{stage}', f'stages.{stage}'],
        'FAM': [f'FAM.{stage}'] if stage in fam_stages else [],
        'vote_fusion': [],
        'out_norm': [f'extra_norms.{stage}'],
    }
    if stage in fusion_stages:
        prefixes['vote_fusion'].extend([
            f'conv11.{stage}', f'conv12.{stage}',
            f'bn11.{stage}', f'bn12.{stage}',
        ])
    return prefixes


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f'config file not found: {cfg_path}')

    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('scope', 'mmseg'))
    if cfg.model.get('backbone', None) is not None:
        cfg.model.backbone.topp_flash_backend = None
        cfg.model.backbone.topp_flash_debug = False

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')
    model = MODELS.build(cfg.model)
    model.eval().to(device)
    backbone = model.backbone
    backbone.eval()
    if hasattr(backbone, '_disable_inference_fusion'):
        backbone._disable_inference_fusion = True

    input_shape = _input_shape(args.shape)
    dummy = torch.randn(1, *input_shape, device=device)

    # fvcore 对 nn.Identity 的别名处理有 bug，用逐层 hook 手动统计 FLOPs
    flops_dict = {}
    hooks = []

    def _make_hook(name):
        def hook_fn(module, inp, out):
            flops = 0
            if isinstance(module, torch.nn.Linear):
                # FLOPs = batch * in_features * out_features
                flops = inp[0].shape[0] * module.in_features * module.out_features
            elif isinstance(module, torch.nn.Conv2d):
                out_h, out_w = out.shape[2], out.shape[3]
                flops = module.in_channels * module.out_channels * \
                    module.kernel_size[0] * module.kernel_size[1] * out_h * out_w // module.groups
            elif isinstance(module, torch.nn.BatchNorm2d):
                flops = inp[0].numel() * 2
            if flops > 0:
                flops_dict[name] = flops
        return hook_fn

    for name, module in backbone.named_modules():
        hooks.append(module.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        backbone(dummy)

    for h in hooks:
        h.remove()

    def _sum_flops_by_prefix(prefixes):
        total = 0.0
        for name, flops in flops_dict.items():
            for prefix in prefixes:
                if name == prefix or name.startswith(prefix + '.'):
                    total += flops
                    break
        return total

    print('stage | cnn | transformer | FAM | vote_fusion | out_norm')
    for stage in range(4):
        cells = []
        prefixes_by_group = _stage_prefixes(backbone, stage)
        for group in ('cnn', 'transformer', 'FAM', 'vote_fusion',
                      'out_norm'):
            prefixes = prefixes_by_group[group]
            group_flops = _sum_flops_by_prefix(prefixes)
            group_params = _count_params(backbone, prefixes)
            cells.append(
                f'{_format_flops(group_flops)}/{_format_params(group_params)}')
        print(f'{stage} | ' + ' | '.join(cells))

    total_params = sum(p.numel() for p in backbone.parameters())
    total_flops = sum(flops_dict.values())
    print(f'\nTotal: {total_flops/1e9:.2f}G FLOPs, {total_params/1e6:.2f}M params')


if __name__ == '__main__':
    main()
