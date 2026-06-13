# PVSA-Net: Top-P Voting Sparse Attention Network

> 分支备注：`pvsa-v3.0` 是当前主线和仓库默认分支；`main`、`pvsa-v2.0`、`backup/before-topp-mask-20260611` 仅作为历史备份保留。后续开发、训练修复和结果复现请优先基于 `pvsa-v3.0`。

基于 MMSegmentation 的语义分割框架，核心创新是 **Top-P 投票稀疏注意力机制**（ToppAttention）。

## 核心特性

### Top-P 注意力机制
传统 Top-K 注意力固定选择 K 个最相关的窗口，而 Top-P 注意力通过**累积概率阈值**动态确定参与计算的窗口数量：
- 对窗口级注意力分数做 Softmax（带温度缩放）
- 按累积概率 `cumsum <= P` 进行截断
- 保留概率质量集中的窗口，自动过滤噪声

### 三种计算后端
| 后端 | 配置 | 显存 | 速度 | 适用场景 |
|------|------|------|------|----------|
| **kv_gather** | `use_topp_flash=False` | 高 | 快 | 显存充足时使用 |
| **kv_gather + fast** | `use_fast_attention=True` | 中 | 最快 | 推荐，Top-P 剪枝前移 |
| **pruned_kv_gather** | `use_pruned_kv_gather=True` | 高 | 中 | 按 keep_len 裁剪无效路由 |
| **cuda** | `backend='cuda'` | 低 | 中 | 极致显存优化，需编译环境 |

### Top-P 参数配置
| 原 topk | 实际 topk | P 阈值 | 温度 | 能量补偿 |
|---------|----------|--------|------|----------|
| 16 | 25 | 0.2 | 0.0175 | 4 |
| 12 | 18 | 0.4 | 0.025 | 1.5 |
| 8 | 36 | 0.6 | 0.05 | 0.75 |
| 6 | 49 | 0.8 | 0.15 | 0.4 |

## 项目结构
```
PVSA-Net/
├── mmseg/
│   ├── models/
│   │   ├── backbones/
│   │   │   ├── bi_topp_vote.py      # VTFormer 骨干网络
│   │   │   └── biformer_fusion.py   # 双路融合骨干
│   │   ├── utils/
│   │   │   ├── top_p_bra.py         # ToppAttention 实现
│   │   │   ├── topp_flash_kernel.py # 分块/CUDA 后端
│   │   │   └── common.py            # 基础注意力模块
│   │   └── decode_heads/            # 解码头（SegformerHead 等）
│   └── ops/
│       └── topp_flash/              # CUDA 内核源码
├── configs-h/                       # 当前主线配置
└── tools/                           # 训练/推理工具
```

## 快速开始

### 安装
```bash
git clone -b pvsa-v3.0 https://github.com/hyshhh/PVSA-v1.git
cd PVSA-v1
pip install -r requirements/mminstall.txt
pip install -r requirements/runtime.txt
```

### 训练


四种注意力后端对应的训练命令如下。

1. `kv_gather` 模式：原始注意力路径，速度较快，但最占显存。
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=False model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=4
```

2. `pruned_kv_gather` 模式：按 keep_len 裁剪无效路由窗口。
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=False model.backbone.use_pruned_kv_gather=True model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=4
```

3. `cuda` 模式：自定义 CUDA 后端，显存最低，但依赖服务器具备可用的 CUDA 编译环境。
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=True model.backbone.topp_flash_backend=cuda model.backbone.topp_flash_block_windows=16 model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=4
```

4. `torch_block` 模式：PyTorch 分块实现，速度最快，无需编译 CUDA 扩展。
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=True model.backbone.topp_flash_backend=torch_block model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=4
```

5. `kv_gather + fast` 模式：在 kv_gather 基础上启用 Top-P 剪枝前移，减少无效 gather 和 matmul，速度更快。
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=False model.backbone.use_fast_attention=True model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=4
```

### 测试方法
使用 `tools/analysis_tools/benchmark.py` 测试推理速度（FPS），该脚本会自动运行足够轮次并计算平均 FPS。

> **前提**：确保 Python 加载的是本项目代码，而非旧版 mmseg。如果之前在其他路径安装过 mmseg（`pip install -e .`），需要在每次运行前设置 `PYTHONPATH`：
> ```bash
> export PYTHONPATH=/path/to/PVSA-Net:$PYTHONPATH
> ```
> 或者一次性重新注册当前路径：`pip install -e . --force-reinstall --no-deps`

1. `kv_gather` 模式（原始注意力路径）：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth \
  --cfg-options model.backbone.use_topp_flash=False \
  model.backbone.feature_vis_config.enabled=False \
  model.backbone.attn_vis_config.enabled=False
```

2. `pruned_kv_gather` 模式（按 keep_len 裁剪无效路由）：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth \
  --cfg-options model.backbone.use_topp_flash=False \
  model.backbone.use_pruned_kv_gather=True \
  model.backbone.feature_vis_config.enabled=False \
  model.backbone.attn_vis_config.enabled=False
```

3. `cuda` 模式（自定义 CUDA 后端，显存最低）：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth \
  --cfg-options model.backbone.use_topp_flash=True \
  model.backbone.topp_flash_backend=cuda \
  model.backbone.feature_vis_config.enabled=False \
  model.backbone.attn_vis_config.enabled=False
```

4. `torch_block` 模式（PyTorch 分块实现，速度最快）：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth \
  --cfg-options model.backbone.use_topp_flash=True \
  model.backbone.topp_flash_backend=torch_block \
  model.backbone.feature_vis_config.enabled=False \
  model.backbone.attn_vis_config.enabled=False
```

5. `kv_gather + fast` 模式（Top-P 剪枝前移，速度最快）：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth \
  --cfg-options model.backbone.use_topp_flash=False \
  model.backbone.use_fast_attention=True \
  model.backbone.feature_vis_config.enabled=False \
  model.backbone.attn_vis_config.enabled=False
```

如果需要强制确认 CUDA 后端可用（不可用则报错），测试前设置：
```bash
export PVSA_TOPP_FLASH_STRICT_CUDA=1
```



### 编译方法
项目通过 `torch.utils.cpp_extension.load()` 在第一次使用 `cuda` 后端时自动 JIT 编译 CUDA 扩展，不需要手动写 `setup.py`。

推荐先在服务器上打开编译日志：
```bash
export PVSA_TOPP_FLASH_VERBOSE=1
export PVSA_TOPP_FLASH_STRICT_CUDA=1
```

如果服务器 GPU 架构自动检测失败，可以手动指定：
```bash
export PVSA_TOPP_FLASH_ARCH="8.6"
```

首次训练或测试时会触发编译：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=True model.backbone.topp_flash_backend=cuda train_dataloader.batch_size=4
```

### 编译步骤与风险
1. 拉取最新代码
```bash
git pull origin pvsa-v3.0
```
风险：如果服务器本地改过同名文件，`git pull` 可能产生冲突，需要先处理冲突再训练。

2. 确认 CUDA 编译环境
```bash
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```
风险：如果没有 `nvcc`，或者 PyTorch 的 CUDA 版本和系统编译环境不匹配，CUDA 扩展会编译失败。

3. 首次触发 JIT 编译
```bash
export PVSA_TOPP_FLASH_VERBOSE=1
export PVSA_TOPP_FLASH_STRICT_CUDA=1
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=True model.backbone.topp_flash_backend=cuda train_dataloader.batch_size=4
```
风险：第一次运行会比平时慢，因为需要编译扩展；如果开启 `--amp`，张量可能变成 `float16`，当前 CUDA forward 只支持 `float32`，会回退或报错。

4. 清理旧编译缓存后重编译
如果修改过 `mmseg/ops/topp_flash/topp_flash_cuda.cu`，建议清理 PyTorch 扩展缓存后重新触发编译：
```bash
rm -rf ~/.cache/torch_extensions/py*/pvsa_topp_flash_cuda
```
风险：删除缓存后下一次启动会重新编译；如果路径写错可能误删其他缓存，请只删除 `pvsa_topp_flash_cuda` 对应目录。

### 正确性检查清单
本地没有服务器训练环境，因此不在本地做编译、训练或数值验证。当前代码层面的检查重点如下：
- 只允许 `cuda` 后端源文件和 README 变化，其他模式代码不变
- `topp_flash_forward_kernel` 中 `blockIdx.x` 必须对应 `coarse = batch * p2 + p`
- launch 的 grid 数量必须为 `n * p2`
- `flat_out` 写入布局必须保持 `{n, p2, q_len, dim}`
- `unflatten_windows_kernel` 的输入输出布局保持不变
- C++ 绑定 `topp_flash_forward(...)` 和 Python 侧 `extension.forward(...)` 调用签名保持不变
- 严格模式下应通过 `PVSA_TOPP_FLASH_STRICT_CUDA=1` 暴露编译或 dtype 问题，避免静默回退

## 引用
如果本项目对您的研究有帮助，请考虑引用：
```bibtex
@misc{pvsa2024,
    title={PVSA-Net: Top-P Voting Sparse Attention for Semantic Segmentation},
    author={PVSA-Net Contributors},
    year={2024}
}
```

## 致谢
本项目基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 构建，感谢 OpenMMLab 团队的优秀工作。

## 许可证
当前精简分支未保留独立许可证文件；如需正式发布或复用，请从备份分支恢复许可证文件或补充新的许可证说明。
