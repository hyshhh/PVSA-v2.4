# PVSA-Net
## 训练
只使用原始路径训练：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.topp_flash_backend=None model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=16 \
  --work-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA
```
## 原始路径推理
```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth \
  --cfg-options model.backbone.topp_flash_backend=None \
  --input-size 224 224 \
  --cudnn-benchmark
```
打印 attention 各阶段耗时（与 CUDA 核对比）：
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
## 推理参数说明
- `--cudnn-benchmark`：开启 cuDNN 自动调优。固定输入尺寸时通常快 5-15%；首次推理有 autotune 预热。**推理命令默认已加**。
- `--input-size H W`：调整测试输入分辨率（同时覆盖 Resize pipeline 与 data_preprocessor.size）。尺寸需能被 `n_win=7` 整除（如 224/256/512），CUDA 核路径要求整除否则自动回退 torch。
- `--batch-size N`：默认 **1**（真实环境单图延迟 fps）。增大 batch 可测吞吐，但单图延迟不变。benchmark.py 与 test.py 均支持。
```bash
# 真实环境 fps 示例（batch=1，默认）：
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  --input-size 224 224 --cudnn-benchmark
```
## 正确性说明（自定义核 vs 原始路径）
- 自定义 CUDA 核的 `use_route_weight` 自动跟随配置的 `soft_routing`：
  `soft_routing=False` 时核内不乘 route_weight，与原始路径 KVGather 行为一致。
- 修改 `soft_routing` 等路由行为需改配置文件并重训。