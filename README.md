# PVSA-Net

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.topp_flash_backend=None model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=16 \
  --work-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA
```

## 推理并保存分割结果

显示推理结果（分割结果图保存到 `--show-dir` 指定目录）：

### 原始路径推理（torch）

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --show-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/vis_results/gqy \
  --cfg-options model.backbone.topp_flash_backend=None \
  --input-size 224 224 \
  --cudnn-benchmark
```

### 自定义 CUDA 核推理

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --show-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/vis_results/2 \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=false \
  --input-size 224 224 \
  --cudnn-benchmark
```

## FPS 测速

fps 测速、阶段耗时分析、CUDA Graph 高吞吐推理见 **`FPS.md`**。
