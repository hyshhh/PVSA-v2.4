# FPS 测速（benchmark.py）
## 1. 原始路径推理 fps（torch）打印 attention 各阶段耗时（与 CUDA 核对比）：
```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth \
  --cfg-options model.backbone.topp_flash_backend=None \
  model.backbone.topp_flash_debug=true \
  --input-size 224 224 \
  --cudnn-benchmark
```
## 2. 自定义 CUDA 核推理 fps
```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=false \
  --input-size 224 224 \
  --cudnn-benchmark
```

## 3. CUDA Graph 推理（最高吞吐）
```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=false \
  --input-size 224 224 \
  --cudnn-benchmark \
  --cuda-graph
```

## 4. 复杂度统计

```bash
python tools/analysis_tools/get_flops.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py --shape 224 224
```