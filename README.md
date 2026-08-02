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
  --cfg-options model.backbone.topp_flash_backend=None
```
打印 attention 各阶段耗时（与 CUDA 核对比）：
```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA-v2.4:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth \
  --cfg-options model.backbone.topp_flash_backend=None \
  model.backbone.topp_flash_debug=true
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
  --show-dir /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/vis_results/gqy
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
  model.backbone.topp_flash_debug=false
```
## 推理参数（benchmark.py 与 test.py 通用）
```bash
# 开启 cuDNN 自动调优（固定输入尺寸时通常更快，首次推理有 autotune 预热）
--cudnn-benchmark
# 吞吐测试：增大 batch，fps 提升但单图延迟不变
--batch-size 4
# 调整测试输入尺寸（同时覆盖 Resize pipeline 和 data_preprocessor size）
--input-size 512 512
# 组合示例：CUDA 核 + cudnn benchmark + batch 4
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  --cudnn-benchmark --batch-size 4
```
## 输入尺寸参数 `--input-size H W`
推理时直接通过命令行调整输入分辨率，无需改配置文件。

### 用法
```bash
# 推理前指定 512x512（H=512, W=512）
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth \
  --input-size 512 512

# 保存分割结果时同样支持（tools/test.py）
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v2.4/work_dirs/PVSA/epoch_10.pth \
  --input-size 1024 1024 --show-dir vis_results/
```

### 作用范围
- `test_pipeline` 中的 `Resize` 被覆盖为目标尺寸（`scale=(W, H)`）
- `data_preprocessor.size` 同步更新，避免输入被 padding 回旧尺寸
- `RandomCrop` 的 `crop_size` 一并同步（如需裁剪）

### 注意事项
- **H/W 顺序**：`--input-size H W`，内部转成 mmseg 的 `(W, H)` 约定
- **n_win=7 整除性**：模型窗口路由要求尺寸能被 7 整除（如 224/256/512/1024），否则 attention 的 padding 路径生效
- **CUDA 核约束**：`topp_flash_backend=cuda` 要求 H、W 能被 `n_win` 整除（`_can_use_specialized_kernel` 检查），不满足会自动回退 torch 路径（不影响正确性，只影响速度）
- 输入尺寸变化会改变各 stage 特征尺寸，**推理速度随之变化**（尺寸越大越慢）

## 正确性说明（自定义核 vs 原始路径）
- 自定义 CUDA 核的 `use_route_weight` 自动跟随配置的 `soft_routing`：
  `soft_routing=False` 时核内不乘 route_weight，与原始路径 KVGather 行为一致。
- 如需修改 `soft_routing` 等路由行为，请改配置文件后重训。