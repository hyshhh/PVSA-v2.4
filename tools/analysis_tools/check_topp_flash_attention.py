import argparse
import copy
import importlib.util
import sys
import time
import types
from pathlib import Path

import torch


def load_kernel_module():
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    path = root / 'mmseg' / 'models' / 'utils' / 'topp_flash_kernel.py'
    spec = importlib.util.spec_from_file_location('topp_flash_kernel', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_topp_attention_class():
    try:
        from mmseg.models.utils.top_p_bra import ToppAttention
        return ToppAttention
    except Exception:
        pass

    root = Path(__file__).resolve().parents[2]
    package_paths = {
        'mmseg': root / 'mmseg',
        'mmseg.models': root / 'mmseg' / 'models',
        'mmseg.models.utils': root / 'mmseg' / 'models' / 'utils',
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    kernel_name = 'mmseg.models.utils.topp_flash_kernel'
    if kernel_name not in sys.modules:
        kernel_path = package_paths['mmseg.models.utils'] / 'topp_flash_kernel.py'
        kernel_spec = importlib.util.spec_from_file_location(
            kernel_name, kernel_path)
        kernel_module = importlib.util.module_from_spec(kernel_spec)
        sys.modules[kernel_name] = kernel_module
        kernel_spec.loader.exec_module(kernel_module)

    module_name = 'mmseg.models.utils.top_p_bra'
    path = package_paths['mmseg.models.utils'] / 'top_p_bra.py'
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.ToppAttention


def make_inputs(args, device, dtype):
    torch.manual_seed(args.seed)
    p2 = args.n_win * args.n_win
    q_h = args.height // args.n_win
    q_w = args.width // args.n_win
    q_len = q_h * q_w
    q_pix = torch.randn(
        args.batch, p2, q_len, args.qk_dim, device=device, dtype=dtype)
    kv_pix = torch.randn(
        args.batch,
        p2,
        args.kv_len,
        args.qk_dim + args.dim,
        device=device,
        dtype=dtype)
    r_weight = torch.rand(
        args.batch, p2, args.topk, device=device, dtype=dtype)
    r_idx = torch.randint(0, p2, (args.batch, p2, args.topk), device=device)
    r_mask = torch.rand(args.batch, p2, args.topk, device=device) > 0.25
    r_mask[..., 0] = True
    return {
        'q_pix': q_pix,
        'kv_pix': kv_pix,
        'r_weight': r_weight,
        'r_idx': r_idx,
        'r_mask': r_mask,
        'num_heads': args.num_heads,
        'qk_dim': args.qk_dim,
        'dim': args.dim,
        'scale': args.qk_dim**-0.5,
        'n_win': args.n_win,
        'H': args.height,
        'W': args.width,
    }


def clone_for_grad(inputs):
    cloned = dict(inputs)
    for name in ('q_pix', 'kv_pix', 'r_weight'):
        cloned[name] = inputs[name].detach().clone().requires_grad_(True)
    return cloned


def max_error(a, b):
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-12)
    return diff.max().item(), (diff / denom).max().item()


def sync_if_needed(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark(fn, device, repeat, warmup):
    for _ in range(warmup):
        fn()
    sync_if_needed(device)
    peak_memory = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync_if_needed(device)
    elapsed = (time.perf_counter() - start) * 1000.0 / repeat
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    return elapsed, peak_memory


def run_direct_check(args, kernel, device, dtype):
    inputs = make_inputs(args, device, dtype)
    with torch.no_grad():
        out_ref = kernel.topp_attention_reference(**inputs)
        out_flash = kernel.topp_flash_attention(
            **inputs, block_windows=args.block_windows, backend=args.backend)
    abs_err, rel_err = max_error(out_flash, out_ref)

    ref_grad_inputs = clone_for_grad(inputs)
    flash_grad_inputs = clone_for_grad(inputs)
    out_ref = kernel.topp_attention_reference(**ref_grad_inputs)
    out_flash = kernel.topp_flash_attention(
        **flash_grad_inputs,
        block_windows=args.block_windows,
        backend=args.backend)
    out_ref.square().mean().backward()
    out_flash.square().mean().backward()
    grad_errors = {}
    for name in ('q_pix', 'kv_pix', 'r_weight'):
        grad_errors[name] = max_error(
            flash_grad_inputs[name].grad,
            ref_grad_inputs[name].grad)

    ref_time, ref_memory = benchmark(
        lambda: kernel.topp_attention_reference(**inputs), device, args.repeat,
        args.warmup)
    flash_time, flash_memory = benchmark(
        lambda: kernel.topp_flash_attention(
            **inputs, block_windows=args.block_windows, backend=args.backend),
        device, args.repeat, args.warmup)

    print('direct_check:')
    print(f'  backend: {args.backend}')
    print(f'  device: {device}')
    print(f'  dtype: {dtype}')
    print(f'  forward_max_abs_error: {abs_err:.6e}')
    print(f'  forward_max_rel_error: {rel_err:.6e}')
    for name, (abs_grad, rel_grad) in grad_errors.items():
        print(f'  grad_{name}_max_abs_error: {abs_grad:.6e}')
        print(f'  grad_{name}_max_rel_error: {rel_grad:.6e}')
    print(f'  reference_time_ms: {ref_time:.4f}')
    print(f'  selected_backend_time_ms: {flash_time:.4f}')
    if ref_memory is not None and flash_memory is not None:
        print(f'  reference_peak_memory_mb: {ref_memory:.2f}')
        print(f'  selected_backend_peak_memory_mb: {flash_memory:.2f}')


def run_full_module_check(args, device):
    if device.type != 'cuda':
        print('full_module_check: skipped, current TopkRouting uses cuda:0.')
        return
    try:
        ToppAttention = load_topp_attention_class()
    except Exception as exc:
        print(f'full_module_check: skipped, import failed: {exc}')
        return

    torch.manual_seed(args.seed)
    module_kwargs = dict(
        dim=args.dim,
        num_heads=args.num_heads,
        n_win=7,
        qk_dim=args.qk_dim,
        kv_downsample_mode='identity',
        topk=6,
        auto_pad=False,
        use_topp_flash=False)
    ref = ToppAttention(**module_kwargs).to(device).eval()
    flash = copy.deepcopy(ref).to(device).eval()
    flash.use_topp_flash = True
    flash.topp_flash_block_windows = args.block_windows
    flash.topp_flash_backend = args.backend

    x = torch.randn(1, 14, 14, args.dim, device=device)
    with torch.no_grad():
        out_ref = ref(x, None)
        out_flash = flash(x, None)
    abs_err, rel_err = max_error(out_flash, out_ref)

    x_ref = x.detach().clone().requires_grad_(True)
    x_flash = x.detach().clone().requires_grad_(True)
    out_ref = ref(x_ref, None)
    out_flash = flash(x_flash, None)
    out_ref.square().mean().backward()
    out_flash.square().mean().backward()
    grad_abs, grad_rel = max_error(x_flash.grad, x_ref.grad)

    ref_time, ref_memory = benchmark(lambda: ref(x, None), device, args.repeat,
                                     args.warmup)
    flash_time, flash_memory = benchmark(
        lambda: flash(x, None), device, args.repeat, args.warmup)

    print('full_module_check:')
    print(f'  forward_max_abs_error: {abs_err:.6e}')
    print(f'  forward_max_rel_error: {rel_err:.6e}')
    print(f'  input_grad_max_abs_error: {grad_abs:.6e}')
    print(f'  input_grad_max_rel_error: {grad_rel:.6e}')
    print(f'  reference_time_ms: {ref_time:.4f}')
    print(f'  selected_backend_time_ms: {flash_time:.4f}')
    if ref_memory is not None and flash_memory is not None:
        print(f'  reference_peak_memory_mb: {ref_memory:.2f}')
        print(f'  selected_backend_peak_memory_mb: {flash_memory:.2f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float64'])
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--height', type=int, default=14)
    parser.add_argument('--width', type=int, default=14)
    parser.add_argument('--n-win', type=int, default=7)
    parser.add_argument('--qk-dim', type=int, default=16)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--topk', type=int, default=16)
    parser.add_argument('--kv-len', type=int, default=4)
    parser.add_argument('--block-windows', type=int, default=16)
    parser.add_argument(
        '--backend',
        default='torch_block',
        choices=['torch_block', 'block', 'torch', 'cuda', 'cuda_forward'])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=50)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--full-module', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    if args.height % args.n_win != 0 or args.width % args.n_win != 0:
        raise ValueError('height and width must be divisible by n_win.')
    if args.qk_dim % args.num_heads != 0 or args.dim % args.num_heads != 0:
        raise ValueError('qk_dim and dim must be divisible by num_heads.')

    kernel = load_kernel_module()
    run_direct_check(args, kernel, device, dtype)
    if args.full_module:
        run_full_module_check(args, device)


if __name__ == '__main__':
    main()
