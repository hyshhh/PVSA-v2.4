# Copyright (c) OpenMMLab. All rights reserved.
"""Optional block attention backend for PVSA ToppAttention.

The default backend in this file is a PyTorch block implementation with a
custom autograd wrapper. It is intended for correctness and training checks
before replacing the inner block with a hand-written CUDA kernel.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.cpp_extension import CUDA_HOME, load


_CUDA_BACKENDS = {'cuda', 'cuda_forward'}
_CUDA_EXTENSION = None
_CUDA_EXTENSION_ERROR = None
_CUDA_FALLBACK_WARNED = False


def _normalize_backend(backend: Optional[str] = None) -> str:
    backend = backend or os.getenv('PVSA_TOPP_FLASH_BACKEND', 'cuda')
    return backend.strip().lower()


def is_topp_flash_available(backend: Optional[str] = None) -> bool:
    """Return whether the requested optional backend is available."""
    backend = _normalize_backend(backend)
    if backend in _CUDA_BACKENDS:
        return _can_build_cuda_extension()
    return False


def topp_attention_reference(q_pix: Tensor,
                             kv_pix: Tensor,
                             r_weight: Tensor,
                             r_idx: Tensor,
                             r_mask: Tensor,
                             num_heads: int,
                             qk_dim: int,
                             dim: int,
                             scale: float,
                             n_win: int,
                             H: int,
                             W: int) -> Tensor:
    """Reference implementation that matches the original kv_gather path."""
    _validate_inputs(q_pix, kv_pix, r_weight, r_idx, r_mask, num_heads,
                     qk_dim, dim, n_win, H, W)
    n, p2, kv_len, c_kv = kv_pix.shape
    topk = r_idx.size(-1)

    kv_pix_sel = torch.gather(
        kv_pix.view(n, 1, p2, kv_len, c_kv).expand(-1, p2, -1, -1, -1),
        dim=2,
        index=r_idx.long().view(n, p2, topk, 1, 1).expand(
            -1, -1, -1, kv_len, c_kv))
    kv_pix_sel = r_weight.view(n, p2, topk, 1, 1).to(kv_pix_sel.dtype) * kv_pix_sel
    k_pix_sel, v_pix_sel = kv_pix_sel.split([qk_dim, dim], dim=-1)

    head_q = qk_dim // num_heads
    head_v = dim // num_heads
    k_pix_sel = k_pix_sel.view(
        n, p2, topk, kv_len, num_heads, head_q).permute(
            0, 1, 4, 5, 2, 3).reshape(n * p2, num_heads, head_q,
                                      topk * kv_len)
    v_pix_sel = v_pix_sel.view(
        n, p2, topk, kv_len, num_heads, head_v).permute(
            0, 1, 4, 2, 3, 5).reshape(n * p2, num_heads, topk * kv_len,
                                      head_v)
    q = q_pix.view(n * p2, -1, num_heads, head_q).permute(0, 2, 1, 3)

    attn_weight = (q * scale) @ k_pix_sel
    route_mask = r_mask[..., None].expand(-1, -1, -1, kv_len)
    route_mask = route_mask.reshape(n * p2, 1, 1, topk * kv_len)
    attn_weight = attn_weight.masked_fill(
        ~route_mask, torch.finfo(attn_weight.dtype).min)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    out = attn_weight @ v_pix_sel
    out = out.permute(0, 2, 1, 3).reshape(n * p2, -1, dim)
    return _unflatten_windows(out, n, n_win, H, W, dim)


def topp_flash_attention(q_pix: Tensor,
                         kv_pix: Tensor,
                         r_weight: Tensor,
                         r_idx: Tensor,
                         r_mask: Tensor,
                         num_heads: int,
                         qk_dim: int,
                         dim: int,
                         scale: float,
                         n_win: int,
                         H: int,
                         W: int,
                         block_windows: int = 64,
                         backend: Optional[str] = None) -> Tensor:
    """Compute routed attention via CUDA kernel only."""
    backend = _normalize_backend(backend)
    if backend not in _CUDA_BACKENDS:
        raise RuntimeError(
            f'topp flash backend {backend!r} is unavailable. '
            f'Only CUDA backends are supported: {_CUDA_BACKENDS}')
    if _can_run_cuda_forward(q_pix, kv_pix, r_weight, r_idx, r_mask):
        try:
            return _ToppCudaForwardFunction.apply(
                q_pix, kv_pix, r_weight, r_idx, r_mask, num_heads,
                qk_dim, dim, float(scale), n_win, H, W,
                int(block_windows))
        except Exception as exc:
            if _strict_cuda_backend():
                raise
            _warn_cuda_fallback(str(exc))
    if _strict_cuda_backend():
        raise RuntimeError(
            'PVSA CUDA backend is requested but cannot run with the '
            'current device, dtype, or build environment.')
    else:
        _warn_cuda_fallback()
    raise RuntimeError(
        'PVSA CUDA topp attention kernel is unavailable and no fallback '
        'backend is configured.')


class _ToppCudaForwardFunction(torch.autograd.Function):
    """CUDA forward with the existing recompute backward path."""

    @staticmethod
    def forward(ctx, q_pix: Tensor, kv_pix: Tensor, r_weight: Tensor,
                r_idx: Tensor, r_mask: Tensor, num_heads: int, qk_dim: int,
                dim: int, scale: float, n_win: int, H: int, W: int,
                block_windows: int) -> Tensor:
        q_pix = q_pix.contiguous()
        kv_pix = kv_pix.contiguous()
        r_weight = r_weight.contiguous()
        r_idx = r_idx.contiguous().long()
        r_mask = r_mask.contiguous().bool()
        keep_len = r_mask.sum(dim=-1).contiguous().long()
        ctx.save_for_backward(q_pix, kv_pix, r_weight, r_idx, r_mask)
        ctx.params = (num_heads, qk_dim, dim, scale, n_win, H, W,
                      block_windows, keep_len)
        extension = _load_cuda_extension()
        return extension.forward(q_pix, kv_pix, r_weight, r_idx, keep_len,
                                 num_heads, qk_dim, dim, float(scale),
                                 n_win, H, W)

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tuple[Optional[Tensor], ...]:
        q_pix, kv_pix, r_weight, r_idx, r_mask = ctx.saved_tensors
        num_heads, qk_dim, dim, scale, n_win, H, W, block_windows, keep_len = ctx.params

        needs = ctx.needs_input_grad
        with torch.enable_grad():
            q = q_pix.detach().requires_grad_(needs[0])
            kv = kv_pix.detach().requires_grad_(needs[1])
            rw = r_weight.detach().requires_grad_(needs[2])
            out = _topp_attention_block_impl(
                q, kv, rw, r_idx, r_mask, num_heads, qk_dim, dim, scale,
                n_win, H, W, block_windows, keep_len)

        targets = []
        positions = []
        if needs[0]:
            targets.append(q)
            positions.append(0)
        if needs[1]:
            targets.append(kv)
            positions.append(1)
        if needs[2]:
            targets.append(rw)
            positions.append(2)

        grads = [None, None, None]
        if targets:
            computed = torch.autograd.grad(
                out, targets, grad_out.contiguous(), allow_unused=True)
            for pos, grad in zip(positions, computed):
                grads[pos] = grad

        return (grads[0], grads[1], grads[2], None, None, None, None, None,
                None, None, None, None, None)


def _unflatten_windows(flat_out: Tensor, n: int, n_win: int, H: int, W: int,
                       dim: int) -> Tensor:
    q_h = H // n_win
    q_w = W // n_win
    return flat_out.view(n, n_win, n_win, q_h, q_w, dim).permute(
        0, 1, 3, 2, 4, 5).reshape(n, H, W, dim).contiguous()


def _can_build_cuda_extension() -> bool:
    if not torch.cuda.is_available() or CUDA_HOME is None:
        return False
    cpp_path, cu_path = _cuda_source_paths()
    return cpp_path.exists() and cu_path.exists()


def _can_run_cuda_forward(q_pix: Tensor, kv_pix: Tensor, r_weight: Tensor,
                          r_idx: Tensor, r_mask: Tensor) -> bool:
    if not _can_build_cuda_extension():
        return False
    tensors = (q_pix, kv_pix, r_weight, r_idx, r_mask)
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    # 支持 float32, float16, bfloat16
    if q_pix.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if kv_pix.dtype != q_pix.dtype or r_weight.dtype != q_pix.dtype:
        return False
    return r_idx.dtype == torch.long and r_mask.dtype == torch.bool


def _cuda_source_paths() -> Tuple[Path, Path]:
    root = Path(__file__).resolve().parents[3]
    op_dir = root / 'mmseg' / 'ops' / 'topp_flash'
    return op_dir / 'topp_flash.cpp', op_dir / 'topp_flash_cuda.cu'


def _load_cuda_extension():
    global _CUDA_EXTENSION, _CUDA_EXTENSION_ERROR   #用于缓存已加载的拓展
    if _CUDA_EXTENSION is not None:  #如果已经加载过，直接返回缓存实例
        return _CUDA_EXTENSION
    if not _can_build_cuda_extension():
        raise RuntimeError('PVSA CUDA extension build environment is missing.')

    cpp_path, cu_path = _cuda_source_paths()  #获取cuda源码文件路径.cpp 绑定文件 + .cu 内核文件
    extra_cuda_cflags = ['-O3']    #编译优化级别为最高
    arch_list = os.getenv('PVSA_TOPP_FLASH_ARCH')   #通过环境指定目标GPU架构
    if arch_list:
        os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list
    try:
        _CUDA_EXTENSION = load(  #核心调用：torch.utils.cpp_extension.load() 是 PyTorch 的 JIT 编译器。把 C++/CUDA 源码编译成 Python 可调用的模块。
            name='pvsa_topp_flash_cuda',   # 编译产物的名字Linux——共享库（多个程序可以共享的代码库）——pvsa_topp_flash_cuda.so
            sources=[str(cpp_path), str(cu_path)], # 要编译的源文件
            extra_cuda_cflags=extra_cuda_cflags,  # nvcc 编译参数
            verbose=os.getenv('PVSA_TOPP_FLASH_VERBOSE', '0') == '1') # 是否打印编译日志
    # 最终存到全局变量中，全局变量不会随之函数的结束而销毁//程序结束文件还在
    except Exception as exc:
        _CUDA_EXTENSION_ERROR = exc
        raise
    return _CUDA_EXTENSION


def _strict_cuda_backend() -> bool:
    return os.getenv('PVSA_TOPP_FLASH_STRICT_CUDA', '0') == '1'


def _warn_cuda_fallback(reason: Optional[str] = None) -> None:
    global _CUDA_FALLBACK_WARNED
    if _CUDA_FALLBACK_WARNED:
        return
    message = (
        'PVSA CUDA topp attention kernel is unavailable; fallback to '
        'PVSA CUDA topp attention kernel is unavailable.')
    if reason:
        message = f'{message} Reason: {reason}'
    warnings.warn(message)
    _CUDA_FALLBACK_WARNED = True


def _validate_inputs(q_pix: Tensor, kv_pix: Tensor, r_weight: Tensor,
                     r_idx: Tensor, r_mask: Tensor, num_heads: int,
                     qk_dim: int, dim: int, n_win: int, H: int, W: int) -> None:
    if q_pix.dim() != 4 or kv_pix.dim() != 4:
        raise ValueError('q_pix and kv_pix must be 4D tensors.')
    n, p2, q_len, qk = q_pix.shape
    n_kv, p2_kv, _, c_kv = kv_pix.shape
    if (n_kv, p2_kv) != (n, p2):
        raise ValueError('q_pix and kv_pix must share n and p2 dimensions.')
    if r_idx.shape != r_weight.shape or r_idx.shape != r_mask.shape:
        raise ValueError('r_weight, r_idx and r_mask must share one shape.')
    if r_idx.shape[:2] != (n, p2):
        raise ValueError('routing tensors must match q_pix n and p2.')
    if p2 != n_win * n_win:
        raise ValueError('p2 must equal n_win * n_win.')
    if H % n_win != 0 or W % n_win != 0:
        raise ValueError('H and W must be divisible by n_win.')
    if q_len != (H // n_win) * (W // n_win):
        raise ValueError('q_len must match the window area.')
    if qk != qk_dim or c_kv != qk_dim + dim:
        raise ValueError('channel dimensions do not match qk_dim and dim.')
    if qk_dim % num_heads != 0 or dim % num_heads != 0:
        raise ValueError('qk_dim and dim must be divisible by num_heads.')
