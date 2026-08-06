"""CUDA Graph 图内计时事件兼容层。

较旧版本的 PyTorch 没有暴露 ``torch.cuda.Event(external=True)``。
当 CUDA 运行时提供 ``cudaEventRecordWithFlags`` 时，使用轻量的运行时接口
创建外部事件，保持真实 CUDA Graph 重放计时；优先使用 PyTorch 原生接口。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
from pathlib import Path
from typing import Iterable, Optional

import torch


# CUDA Runtime API 中 cudaEventRecordExternal 的标志值。
_CUDA_EVENT_RECORD_EXTERNAL = 1
_CUDART = None
_CUDART_LOAD_ERROR = None


def _candidate_cudart_paths() -> Iterable[str]:
    seen = set()

    def add(path: Optional[str]):
        if not path:
            return
        path = str(path)
        if path not in seen:
            seen.add(path)
            yield path

    found = ctypes.util.find_library("cudart")
    yield from add(found)

    for variable in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(variable)
        if root:
            for name in ("lib64/libcudart.so", "lib/libcudart.so",
                         "bin/cudart64_*.dll"):
                yield from add(str(Path(root) / name))

    for root in ("/usr/local/cuda", "/usr/local/cuda-12",
                 "/usr/local/cuda-11"):
        for pattern in ("lib64/libcudart.so*", "lib/libcudart.so*"):
            for path in sorted(glob.glob(str(Path(root) / pattern))):
                yield from add(path)

    torch_root = Path(torch.__file__).resolve().parent
    for root in (torch_root / "lib",
                 torch_root.parent / "nvidia" / "cuda_runtime" / "lib"):
        for pattern in ("libcudart.so*", "cudart64_*.dll"):
            for path in sorted(root.glob(pattern)):
                yield from add(str(path))


def _load_cudart():
    global _CUDART, _CUDART_LOAD_ERROR
    if _CUDART is not None:
        return _CUDART
    if _CUDART_LOAD_ERROR is not None:
        raise RuntimeError(_CUDART_LOAD_ERROR)

    errors = []
    for path in _candidate_cudart_paths():
        try:
            library = ctypes.CDLL(path)
            required = ("cudaEventCreateWithFlags",
                        "cudaEventRecordWithFlags",
                        "cudaEventSynchronize",
                        "cudaEventElapsedTime",
                        "cudaEventDestroy")
            missing = [name for name in required
                       if not hasattr(library, name)]
            if missing:
                errors.append(f"{path}: 缺少 {', '.join(missing)}")
                continue

            void_p = ctypes.c_void_p
            uint = ctypes.c_uint
            int_type = ctypes.c_int
            float_type = ctypes.c_float
            library.cudaEventCreateWithFlags.argtypes = [
                ctypes.POINTER(void_p), uint]
            library.cudaEventCreateWithFlags.restype = int_type
            library.cudaEventRecordWithFlags.argtypes = [
                void_p, void_p, uint]
            library.cudaEventRecordWithFlags.restype = int_type
            library.cudaEventSynchronize.argtypes = [void_p]
            library.cudaEventSynchronize.restype = int_type
            library.cudaEventElapsedTime.argtypes = [
                ctypes.POINTER(float_type), void_p, void_p]
            library.cudaEventElapsedTime.restype = int_type
            library.cudaEventDestroy.argtypes = [void_p]
            library.cudaEventDestroy.restype = int_type
            if hasattr(library, "cudaGetErrorString"):
                library.cudaGetErrorString.argtypes = [int_type]
                library.cudaGetErrorString.restype = ctypes.c_char_p
            _CUDART = library
            return library
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    _CUDART_LOAD_ERROR = (
        "当前 PyTorch 不支持 torch.cuda.Event(external=True)，且无法加载 "
        "CUDA 运行时的外部事件接口 cudaEventRecordWithFlags。请升级 "
        "PyTorch，或确认 CUDA 运行时库已加入 LD_LIBRARY_PATH。"
        + (" 尝试路径：" + "；".join(errors) if errors else ""))
    raise RuntimeError(_CUDART_LOAD_ERROR)


def _check_cuda_result(library, result: int, function_name: str) -> None:
    if int(result) == 0:
        return
    detail = "未知 CUDA 错误"
    if hasattr(library, "cudaGetErrorString"):
        try:
            value = library.cudaGetErrorString(int(result))
            if value:
                detail = value.decode("utf-8", errors="replace")
        except Exception:
            pass
    raise RuntimeError(
        f"{function_name} 调用失败，错误码={int(result)}，{detail}")


def _stream_pointer(stream) -> ctypes.c_void_p:
    value = getattr(stream, "cuda_stream", None)
    if value is None:
        raise RuntimeError("无法取得 CUDA 流句柄，不能记录 CUDA Graph 事件")
    return ctypes.c_void_p(int(value))


def _is_stream_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


class CUDARTGraphEvent:
    """使用 CUDA Runtime API 记录可被 Graph 重放的外部计时事件。"""

    backend = "cudart_external"

    def __init__(self) -> None:
        self._library = _load_cudart()
        self._handle = ctypes.c_void_p()
        _check_cuda_result(
            self._library,
            self._library.cudaEventCreateWithFlags(
                ctypes.byref(self._handle), 0),
            "cudaEventCreateWithFlags")

    def record(self, stream) -> None:
        flags = (_CUDA_EVENT_RECORD_EXTERNAL
                 if _is_stream_capturing() else 0)
        _check_cuda_result(
            self._library,
            self._library.cudaEventRecordWithFlags(
                self._handle, _stream_pointer(stream), flags),
            "cudaEventRecordWithFlags")

    def synchronize(self) -> None:
        _check_cuda_result(
            self._library,
            self._library.cudaEventSynchronize(self._handle),
            "cudaEventSynchronize")

    def elapsed_time(self, end_event) -> float:
        if not isinstance(end_event, CUDARTGraphEvent):
            raise TypeError("CUDARTGraphEvent 只能与同类型事件计算耗时")
        elapsed = ctypes.c_float()
        _check_cuda_result(
            self._library,
            self._library.cudaEventElapsedTime(
                ctypes.byref(elapsed), self._handle, end_event._handle),
            "cudaEventElapsedTime")
        return float(elapsed.value)

    def __del__(self):
        handle = getattr(self, "_handle", None)
        library = getattr(self, "_library", None)
        if handle is None or library is None or not handle.value:
            return
        try:
            library.cudaEventDestroy(handle)
        except Exception:
            pass
        self._handle = ctypes.c_void_p()


def new_graph_event():
    """优先返回 PyTorch 原生外部事件，不支持时使用 CUDA Runtime 回退。"""
    try:
        event = torch.cuda.Event(enable_timing=True, external=True)
        try:
            event._compare_graph_backend = "pytorch_external"
        except Exception:
            pass
        return event
    except TypeError:
        return CUDARTGraphEvent()
