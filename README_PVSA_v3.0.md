# PVSA v3.0 服务器使用说明

## 分支定位

`pvsa-v3.0` 是带自定义显卡核源码的分支。当前已经加入前向核源码：

```text
mmseg/ops/topp_flash/topp_flash.cpp
mmseg/ops/topp_flash/topp_flash_cuda.cu
```

默认行为仍然安全：

```text
use_topp_flash=False
```

只有显式设置下面参数时才会尝试使用新后端：

```text
use_topp_flash=True
topp_flash_backend='cuda'
```

导入仓库不会编译核，不会修改系统 `CUDA`，不会重装 `torch`，不会改驱动。显卡核通过 `torch.utils.cpp_extension.load` 在运行时按需编译。

## 当前能力

当前 `pvsa-v3.0` 的显卡核只实现前向：

```text
q_pix + kv_pix + r_idx + r_weight + r_mask -> out
```

反向传播暂时仍然使用 `pvsa-v2.0` 的重算式 `PyTorch` 后端，因此训练链路可以跑，但不是完整纯显卡核训练。下一步性能优化需要继续实现 `topp_flash_backward_cuda.cu`。

## 服务器准备

进入服务器上的仓库：

```bash
cd /path/to/PVSA-Net
git fetch pvsa-v1
git checkout pvsa-v3.0
git pull pvsa-v1 pvsa-v3.0
```

确认当前分支：

```bash
git branch --show-current
```

应该输出：

```text
pvsa-v3.0
```

## 环境检查

先确认 `torch` 能看到显卡：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
PY
```

确认服务器有 `nvcc`：

```bash
nvcc --version
```

如果没有 `nvcc`，`topp_flash_backend='cuda'` 会回退到 `torch_block` 后端，不会使用真正显卡核。

安装运行时编译需要的 `ninja`：

```bash
pip install ninja
```

确认 `ninja` 可用：

```bash
ninja --version
```

如果服务器没有完整依赖，还需要按你的原环境安装：

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install einops timm fairscale
pip install -e .
```

注意：不要随便重装 `torch`。`torch`、驱动、服务器 `CUDA` 版本需要和你原训练环境保持一致。

## 先跑安全后端

先不要启用显卡核，确认数学路径和数据集能跑：

```bash
python tools/analysis_tools/check_topp_flash_attention.py \
  --device cuda \
  --backend torch_block \
  --repeat 10 \
  --warmup 3 \
  --block-windows 8
```

这个命令不编译自定义核，只检查普通块式后端。输出里应看到：

```text
forward_max_abs_error
grad_q_pix_max_abs_error
grad_kv_pix_max_abs_error
grad_r_weight_max_abs_error
selected_backend_time_ms
selected_backend_peak_memory_mb
```

## 编译并检查显卡核

确认安全后端没问题后，再检查显卡核：

```bash
export PVSA_TOPP_FLASH_BACKEND=cuda
python tools/analysis_tools/check_topp_flash_attention.py \
  --device cuda \
  --backend cuda \
  --repeat 10 \
  --warmup 3 \
  --block-windows 8
```

第一次运行会触发编译，时间会比较长。编译产物默认放在 `torch` 扩展缓存目录里，不会写进仓库源码。

如果你想确认显卡核必须被使用，不允许回退：

```bash
export PVSA_TOPP_FLASH_BACKEND=cuda
export PVSA_TOPP_FLASH_STRICT_CUDA=1
python tools/analysis_tools/check_topp_flash_attention.py \
  --device cuda \
  --backend cuda \
  --repeat 10 \
  --warmup 3 \
  --block-windows 8
```

严格模式下，如果缺 `nvcc`、缺 `ninja`、编译失败、输入不是 `float32`，程序会直接报错，不会回退。

## 训练启动

建议先用原始路径训练，确认数据和配置没问题：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --work-dir work_dirs/pvsa_v3_gqy_base \
  --cfg-options \
  _base_="[../_base_/models/VTFormer-s.py,../_base_/datasets/gqy.py,../_base_/default_runtime.py,../_base_/schedules/schedule_20k.py]"
```

然后启用安全块式后端：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --work-dir work_dirs/pvsa_v3_gqy_torch_block \
  --cfg-options \
  _base_="[../_base_/models/VTFormer-s.py,../_base_/datasets/gqy.py,../_base_/default_runtime.py,../_base_/schedules/schedule_20k.py]" \
  model.backbone.use_topp_flash=True \
  model.backbone.topp_flash_backend="'torch_block'" \
  model.backbone.topp_flash_block_windows=64
```

最后启用显卡核前向：

```bash
export PVSA_TOPP_FLASH_BACKEND=cuda
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --work-dir work_dirs/pvsa_v3_gqy_cuda \
  --cfg-options \
  _base_="[../_base_/models/VTFormer-s.py,../_base_/datasets/gqy.py,../_base_/default_runtime.py,../_base_/schedules/schedule_20k.py]" \
  model.backbone.use_topp_flash=True \
  model.backbone.topp_flash_backend="'cuda'" \
  model.backbone.topp_flash_block_windows=64
```

如果你要强制确认训练中一定用了显卡核：

```bash
export PVSA_TOPP_FLASH_BACKEND=cuda
export PVSA_TOPP_FLASH_STRICT_CUDA=1
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --work-dir work_dirs/pvsa_v3_gqy_cuda_strict \
  --cfg-options \
  _base_="[../_base_/models/VTFormer-s.py,../_base_/datasets/gqy.py,../_base_/default_runtime.py,../_base_/schedules/schedule_20k.py]" \
  model.backbone.use_topp_flash=True \
  model.backbone.topp_flash_backend="'cuda'" \
  model.backbone.topp_flash_block_windows=64
```

## 测试启动

普通测试：

```bash
python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  work_dirs/pvsa_v3_gqy_cuda/latest.pth \
  --cfg-options \
  _base_="[../_base_/models/VTFormer-s.py,../_base_/datasets/gqy.py,../_base_/default_runtime.py,../_base_/schedules/schedule_20k.py]" \
  model.backbone.use_topp_flash=True \
  model.backbone.topp_flash_backend="'cuda'" \
  model.backbone.topp_flash_block_windows=64
```

保存可视化结果：

```bash
python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  work_dirs/pvsa_v3_gqy_cuda/latest.pth \
  --show-dir work_dirs/pvsa_v3_gqy_cuda/show \
  --cfg-options \
  _base_="[../_base_/models/VTFormer-s.py,../_base_/datasets/gqy.py,../_base_/default_runtime.py,../_base_/schedules/schedule_20k.py]" \
  model.backbone.use_topp_flash=True \
  model.backbone.topp_flash_backend="'cuda'" \
  model.backbone.topp_flash_block_windows=64
```

## 数据路径

当前 `gqy.py` 使用：

```text
data_root = data/gqyyz
```

服务器目录需要至少包含：

```text
data/gqyyz/image/train
data/gqyyz/annotation/train
data/gqyyz/image/val
data/gqyyz/annotation/val
data/gqyyz/image/test1
data/gqyyz/annotation/test1
```

## 常见问题

如果报错 `Ninja is required to load C++ extensions`：

```bash
pip install ninja
```

如果报错找不到 `nvcc`：

```bash
nvcc --version
```

没有 `nvcc` 就不能编译显卡核，只能用 `torch_block` 或原始路径。

如果报错 `float32 only`，说明当前输入可能是半精度。先不要开混合精度：

```bash
不要加 --amp
```

如果你必须使用混合精度，当前显卡核会回退到 `torch_block`，后续需要补半精度核。

如果显卡核编译失败但你想继续训练，取消严格模式：

```bash
unset PVSA_TOPP_FLASH_STRICT_CUDA
```

如果你想清理扩展编译缓存：

```bash
python - <<'PY'
import torch
from torch.utils.cpp_extension import _get_build_directory
print(_get_build_directory('pvsa_topp_flash_cuda', verbose=False))
PY
```

删除上面输出的目录后，下次运行会重新编译。

## 推荐顺序

1. 先跑 `use_topp_flash=False`，确认原模型和数据集能训练。
2. 再跑 `topp_flash_backend='torch_block'`，确认新入口不影响训练。
3. 再跑 `topp_flash_backend='cuda'`，检查前向核是否编译和对齐。
4. 严格模式只用于调试，不建议第一次训练就打开。
5. 当前显卡核是前向核，速度是否提升要以服务器实测为准。
