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
_TORCH_BACKENDS = {'torch', 'torch_block', 'block'}
_CUDA_EXTENSION = None
_CUDA_EXTENSION_ERROR = None
_CUDA_FALLBACK_WARNED = False
_CUDA_DEBUG_LOGGED = set()
_CUDA_TIMING_LOGGED = set()
_ROUTE_CUDA_FALLBACK_WARNED = False


def _normalize_backend(backend: Optional[str] = None) -> str:
    backend = backend or os.getenv('PVSA_TOPP_FLASH_BACKEND', 'torch_block')
    return backend.strip().lower()


def is_topp_flash_available(backend: Optional[str] = None) -> bool:
    """Return whether the requested optional backend is available."""
    backend = _normalize_backend(backend)
    if backend in _TORCH_BACKENDS:
        return True
    if backend in _CUDA_BACKENDS:
        return _can_build_cuda_extension()
    return False


def topp_route_cuda(query: Tensor,
                    topk: int,
                    p: float,
                    temperature: float,
                    energy: float,
                    scale: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Fused CUDA inference route for fixed 7x7 windows."""
    extension = _load_cuda_extension()
    return tuple(extension.route_forward(
        query.contiguous(), int(topk), float(p), float(temperature),
        float(energy), float(scale)))


def topp_flash_fused_attention(route_query: Tensor,
                               q_pix: Tensor,
                               kv_pix: Tensor,
                               topk: int,
                               p: float,
                               temperature: float,
                               energy: float,
                               route_scale: float,
                               attn_scale: float,
                               num_heads: int,
                               qk_dim: int,
                               dim: int,
                               n_win: int,
                               H: int,
                               W: int,
                               debug: bool = False) -> Tensor:
    """Inference-only fused route and attention CUDA path."""
    debug_key = None
    debug_path = 'cuda_fused_route'
    if debug:
        debug_key = _log_topp_fused_debug(
            route_query, q_pix, kv_pix, topk, num_heads, qk_dim, dim,
            n_win, H, W)

    def run_fused():
        extension = _load_cuda_extension()
        return extension.fused_forward(
            route_query.contiguous(), q_pix.contiguous(), kv_pix.contiguous(),
            int(topk), float(p), float(temperature), float(energy),
            float(route_scale), float(attn_scale), int(num_heads),
            int(qk_dim), int(dim), int(n_win), int(H), int(W))

    return _maybe_time_debug(debug, debug_key, debug_path, q_pix, run_fused)


def can_run_topp_route_cuda(query: Tensor, topk: int) -> bool:
    if os.getenv('PVSA_TOPP_ROUTE_CUDA', '1') != '1':
        return False
    if not _can_build_cuda_extension():
        return False
    if query.requires_grad:
        return False
    return (query.is_cuda and query.dtype == torch.float32 and
            query.dim() == 3 and query.size(1) == 49 and
            0 < int(topk) <= 49)


def can_run_topp_fused_cuda(route_query: Tensor,
                            q_pix: Tensor,
                            kv_pix: Tensor,
                            topk: int,
                            num_heads: int,
                            qk_dim: int,
                            dim: int,
                            n_win: int,
                            H: int,
                            W: int) -> bool:
    if os.getenv('PVSA_TOPP_FUSED_CUDA', '1') != '1':
        return False
    if not _can_build_cuda_extension():
        return False
    if route_query.requires_grad or q_pix.requires_grad or kv_pix.requires_grad:
        return False
    if not (route_query.is_cuda and q_pix.is_cuda and kv_pix.is_cuda):
        return False
    if route_query.dtype != torch.float32 or q_pix.dtype != torch.float32:
        return False
    if kv_pix.dtype != torch.float32:
        return False
    if route_query.dim() != 3 or q_pix.dim() != 4 or kv_pix.dim() != 4:
        return False
    if n_win != 7 or route_query.size(1) != 49 or q_pix.size(1) != 49:
        return False
    if kv_pix.size(1) != 49 or not (0 < int(topk) <= 49):
        return False
    if H % 7 != 0 or W % 7 != 0:
        return False
    if num_heads not in (2, 4, 8, 16):
        return False
    if qk_dim != dim or qk_dim % num_heads != 0:
        return False
    if qk_dim // num_heads != 32:
        return False
    if route_query.size(0) != q_pix.size(0) or kv_pix.size(0) != q_pix.size(0):
        return False
    if route_query.size(2) != qk_dim or q_pix.size(3) != qk_dim:
        return False
    if kv_pix.size(3) != qk_dim + dim:
        return False
    return q_pix.size(2) == (H // n_win) * (W // n_win)


def warn_topp_route_cuda_fallback(reason: str) -> None:
    global _ROUTE_CUDA_FALLBACK_WARNED
    if _ROUTE_CUDA_FALLBACK_WARNED:
        return
    warnings.warn(
        f'topp route CUDA path failed; fallback to torch route. {reason}')
    _ROUTE_CUDA_FALLBACK_WARNED = True


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
                         backend: Optional[str] = None,
                         debug: bool = False) -> Tensor:
    """Compute routed attention via CUDA kernel or torch_block backend."""
    backend = _normalize_backend(backend)
    debug_key = None
    debug_path = None
    if debug:
        debug_path, debug_key = _log_topp_flash_debug(
            q_pix, kv_pix, r_weight, r_idx, r_mask, num_heads, qk_dim, dim,
            n_win, H, W, backend)
    if backend in _TORCH_BACKENDS:
        def run_torch_block():
            return _ToppBlockAttentionFunction.apply(
                q_pix, kv_pix, r_weight, r_idx, r_mask, num_heads, qk_dim,
                dim, float(scale), n_win, H, W, int(block_windows))

        return _maybe_time_debug(debug, debug_key, debug_path, q_pix,
                                 run_torch_block)
    if backend in _CUDA_BACKENDS:
        if _can_run_cuda_forward(q_pix, kv_pix, r_weight, r_idx, r_mask):
            try:
                def run_cuda():
                    return _ToppCudaForwardFunction.apply(
                        q_pix, kv_pix, r_weight, r_idx, r_mask, num_heads,
                        qk_dim, dim, float(scale), n_win, H, W,
                        int(block_windows))

                return _maybe_time_debug(debug, debug_key, debug_path, q_pix,
                                         run_cuda)
            except Exception as exc:
                if debug:
                    print(f'[PVSA TopP Flash] CUDA fallback reason: {exc}')
                if _strict_cuda_backend():
                    raise
                _warn_cuda_fallback(str(exc))
        if _strict_cuda_backend():
            raise RuntimeError(
                'PVSA CUDA backend is requested but cannot run with the '
                'current device, dtype, or build environment.')
        else:
            _warn_cuda_fallback()
        return _ToppBlockAttentionFunction.apply(
            q_pix, kv_pix, r_weight, r_idx, r_mask, num_heads, qk_dim, dim,
            float(scale), n_win, H, W, int(block_windows))
    else:
        raise RuntimeError(
            f'topp flash backend {backend!r} is unavailable in this build.')


def _can_use_specialized_cuda_kernel(q_pix: Tensor, r_idx: Tensor,
                                     num_heads: int, qk_dim: int, dim: int,
                                     n_win: int, H: int, W: int) -> bool:
    p2 = q_pix.size(1)
    topk = r_idx.size(2)
    if n_win != 7 or p2 != 49 or topk > 49:
        return False
    if H % 7 != 0 or W % 7 != 0:
        return False
    if num_heads not in (2, 4, 8, 16):
        return False
    if qk_dim != dim or qk_dim % num_heads != 0 or dim % num_heads != 0:
        return False
    return qk_dim // num_heads == 32 and dim // num_heads == 32


def _specialized_cuda_reject_reasons(q_pix: Tensor, r_idx: Tensor,
                                     num_heads: int, qk_dim: int, dim: int,
                                     n_win: int, H: int, W: int) -> str:
    reasons = []
    p2 = q_pix.size(1)
    topk = r_idx.size(2)
    if n_win != 7:
        reasons.append(f'n_win={n_win}')
    if p2 != 49:
        reasons.append(f'p2={p2}')
    if topk > 49:
        reasons.append(f'topk={topk}')
    if H % 7 != 0 or W % 7 != 0:
        reasons.append(f'HW={H}x{W}')
    if num_heads not in (2, 4, 8, 16):
        reasons.append(f'heads={num_heads}')
    if qk_dim != dim:
        reasons.append(f'qk_dim!=dim({qk_dim}!={dim})')
    if qk_dim % num_heads != 0 or dim % num_heads != 0:
        reasons.append('head_dim_divisible=False')
    elif qk_dim // num_heads != 32 or dim // num_heads != 32:
        reasons.append(
            f'head_dim={qk_dim // num_heads}/{dim // num_heads}')
    return ','.join(reasons) if reasons else 'none'


def _log_topp_flash_debug(q_pix: Tensor, kv_pix: Tensor, r_weight: Tensor,
                          r_idx: Tensor, r_mask: Tensor, num_heads: int,
                          qk_dim: int, dim: int, n_win: int, H: int, W: int,
                          backend: str) -> Tuple[str, tuple]:
    can_build = _can_build_cuda_extension()
    can_run = _can_run_cuda_forward(q_pix, kv_pix, r_weight, r_idx, r_mask)
    specialized = (
        backend in _CUDA_BACKENDS and can_run and
        _can_use_specialized_cuda_kernel(q_pix, r_idx, num_heads, qk_dim,
                                         dim, n_win, H, W))
    if backend in _TORCH_BACKENDS:
        path = 'torch_block'
    elif specialized:
        path = 'cuda_specialized'
    elif can_run:
        path = 'cuda_generic'
    else:
        path = 'fallback'

    key = (
        backend, path, str(q_pix.dtype), tuple(q_pix.shape),
        tuple(kv_pix.shape), tuple(r_idx.shape), num_heads, qk_dim, dim,
        n_win, H, W)
    if key in _CUDA_DEBUG_LOGGED:
        return path, key
    _CUDA_DEBUG_LOGGED.add(key)

    keep_len = r_mask.sum(dim=-1)
    keep_min = int(keep_len.min().item()) if keep_len.numel() else 0
    keep_max = int(keep_len.max().item()) if keep_len.numel() else 0
    keep_mean = float(keep_len.float().mean().item()) if keep_len.numel() else 0.0
    q_len = q_pix.size(2)
    kv_len = kv_pix.size(2)
    topk = r_idx.size(2)
    reject = _specialized_cuda_reject_reasons(
        q_pix, r_idx, num_heads, qk_dim, dim, n_win, H, W)
    print(
        '[PVSA TopP Flash] '
        f'backend={backend} path={path} build={can_build} can_run={can_run} '
        f'specialized={specialized} dtype={q_pix.dtype} '
        f'q={tuple(q_pix.shape)} kv={tuple(kv_pix.shape)} '
        f'route={tuple(r_idx.shape)} keep=min/mean/max '
        f'{keep_min}/{keep_mean:.2f}/{keep_max} '
        f'heads={num_heads} qk_dim={qk_dim} dim={dim} n_win={n_win} '
        f'H={H} W={W} q_len={q_len} kv_len={kv_len} topk={topk} '
        f'special_reject={reject}')
    return path, key


def _log_topp_fused_debug(route_query: Tensor, q_pix: Tensor, kv_pix: Tensor,
                          topk: int, num_heads: int, qk_dim: int, dim: int,
                          n_win: int, H: int, W: int) -> tuple:
    can_build = _can_build_cuda_extension()
    can_run = can_run_topp_fused_cuda(
        route_query, q_pix, kv_pix, topk, num_heads, qk_dim, dim, n_win, H, W)
    key = (
        'cuda_fused_route', str(q_pix.dtype), tuple(route_query.shape),
        tuple(q_pix.shape), tuple(kv_pix.shape), int(topk), num_heads,
        qk_dim, dim, n_win, H, W)
    if key in _CUDA_DEBUG_LOGGED:
        return key
    _CUDA_DEBUG_LOGGED.add(key)
    print(
        '[PVSA TopP Flash] '
        f'backend=cuda path=cuda_fused_route build={can_build} '
        f'can_run={can_run} specialized=True dtype={q_pix.dtype} '
        f'route_query={tuple(route_query.shape)} q={tuple(q_pix.shape)} '
        f'kv={tuple(kv_pix.shape)} heads={num_heads} qk_dim={qk_dim} '
        f'dim={dim} n_win={n_win} H={H} W={W} '
        f'q_len={q_pix.size(2)} kv_len={kv_pix.size(2)} topk={topk}')
    return key


def _maybe_time_debug(debug: bool, debug_key: Optional[tuple],
                      debug_path: Optional[str], timing_tensor: Tensor,
                      runner):
    if (not debug or not _profile_topp_flash() or debug_key is None or
            debug_key in _CUDA_TIMING_LOGGED):
        return runner()
    if not torch.cuda.is_available() or not timing_tensor.is_cuda:
        return runner()
    _CUDA_TIMING_LOGGED.add(debug_key)
    if debug_path in ('cuda_specialized', 'cuda_generic'):
        _load_cuda_extension()
    repeat = max(1, int(os.getenv('PVSA_TOPP_FLASH_TIMING_REPEAT', '5')))
    warmup = max(0, int(os.getenv('PVSA_TOPP_FLASH_TIMING_WARMUP', '2')))
    out = None
    with torch.cuda.device(timing_tensor.device):
        for _ in range(warmup):
            out = runner()
        torch.cuda.synchronize(timing_tensor.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            out = runner()
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end) / repeat
    print(
        '[PVSA TopP Flash] '
        f'timing path={debug_path} elapsed_ms={elapsed_ms:.4f} '
        f'warmup={warmup} repeat={repeat}')
    return out


def _profile_topp_flash() -> bool:
    return os.getenv('PVSA_TOPP_FLASH_PROFILE', '0') == '1'


class _ToppBlockAttentionFunction(torch.autograd.Function):
    """Autograd wrapper that recomputes the block attention in backward."""

    @staticmethod
    def forward(ctx, q_pix: Tensor, kv_pix: Tensor, r_weight: Tensor,
                r_idx: Tensor, r_mask: Tensor, num_heads: int, qk_dim: int,
                dim: int, scale: float, n_win: int, H: int, W: int,
                block_windows: int) -> Tensor:
        q_pix = q_pix.contiguous()
        kv_pix = kv_pix.contiguous()
        r_weight = r_weight.contiguous()
        r_idx = r_idx.contiguous()
        r_mask = r_mask.contiguous().bool()
        keep_len = r_mask.sum(dim=-1).contiguous().long()
        ctx.save_for_backward(q_pix, kv_pix, r_weight, r_idx, r_mask)
        ctx.params = (num_heads, qk_dim, dim, scale, n_win, H, W,
                      block_windows, keep_len)
        with torch.no_grad():
            return _topp_attention_block_impl(
                q_pix, kv_pix, r_weight, r_idx, r_mask, num_heads, qk_dim,
                dim, scale, n_win, H, W, block_windows, keep_len)

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


def _topp_attention_block_impl(q_pix: Tensor,
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
                               keep_len: Optional[Tensor] = None) -> Tensor:
    _validate_inputs(q_pix, kv_pix, r_weight, r_idx, r_mask, num_heads,
                     qk_dim, dim, n_win, H, W)
    n, p2, q_len, _ = q_pix.shape
    _, _, kv_len, c_kv = kv_pix.shape
    topk = r_idx.size(-1)
    head_q = qk_dim // num_heads
    head_v = dim // num_heads
    flat_size = n * p2
    block_windows = flat_size if block_windows <= 0 else min(block_windows,
                                                             flat_size)

    if keep_len is None:
        keep_len = r_mask.sum(dim=-1).long()
    keep_flat = keep_len.reshape(flat_size)

    q_flat = q_pix.reshape(flat_size, q_len, qk_dim)
    idx_flat = r_idx.reshape(flat_size, topk).long()
    weight_flat = r_weight.reshape(flat_size, topk).to(kv_pix.dtype)
    flat_out = []

    for start in range(0, flat_size, block_windows):
        end = min(start + block_windows, flat_size)
        batch = end - start
        flat_ids = torch.arange(start, end, device=q_pix.device)
        n_ids = torch.div(flat_ids, p2, rounding_mode='floor')

        kv_batch = kv_pix.index_select(0, n_ids)
        keep = keep_flat[start:end]
        max_keep = int(keep.max().item())
        if max_keep <= 0:
            flat_out.append(torch.zeros(batch, q_len, dim, device=q_pix.device, dtype=kv_pix.dtype))
            continue
        max_keep = min(max_keep, topk)

        idx = idx_flat[start:end, :max_keep]
        weight = weight_flat[start:end, :max_keep]
        kv_sel = torch.gather(
            kv_batch,
            dim=1,
            index=idx.view(batch, max_keep, 1, 1).expand(
                -1, -1, kv_len, c_kv))
        kv_sel = weight.view(batch, max_keep, 1, 1) * kv_sel
        k_sel, v_sel = kv_sel.split([qk_dim, dim], dim=-1)

        k_sel = k_sel.view(batch, max_keep, kv_len, num_heads,
                           head_q).permute(0, 3, 4, 1, 2).reshape(
                               batch, num_heads, head_q, max_keep * kv_len)
        v_sel = v_sel.view(batch, max_keep, kv_len, num_heads,
                           head_v).permute(0, 3, 1, 2, 4).reshape(
                               batch, num_heads, max_keep * kv_len, head_v)
        q = q_flat[start:end].view(batch, q_len, num_heads,
                                   head_q).permute(0, 2, 1, 3)

        scores = (q * scale) @ k_sel
        pos = torch.arange(max_keep, device=q_pix.device)
        valid_mask = pos[None, :] < keep[:, None]
        route_mask = valid_mask[:, :, None].expand(-1, -1, kv_len).reshape(batch, 1, 1, max_keep * kv_len)
        scores = scores.masked_fill(~route_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v_sel
        flat_out.append(out.permute(0, 2, 1, 3).reshape(batch, q_len, dim))

    flat_out = torch.cat(flat_out, dim=0)
    return _unflatten_windows(flat_out, n, n_win, H, W, dim)


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
        'torch_block backend.')
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
