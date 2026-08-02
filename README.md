# PVSA-Net
## 训练
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.topp_flash_backend=None model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=16 \
  --work-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA
```
## 原始路径推理 打印 attention 各阶段耗时（与 CUDA 核对比）：
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
## 自定义 CUDA 核推理
推理模板：最后一层默认使用 49 窗口全连接路由，不再需要额外开关。
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
## 复杂度统计
```bash
python tools/analysis_tools/get_flops.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py --shape 224 224
```
## 推理并保存分割结果
```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth  \
  --show-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/vis_results/gqy \
  --input-size 224 224 \
  --cudnn-benchmark
```