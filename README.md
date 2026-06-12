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
| **kv_gather** | `use_topp_flash=False` | 高 | 快（小 topk） | 显存充足时使用 |
| **torch_block** | `use_topp_flash=True, backend='torch_block'` | 中 | 中 | 默认推荐，显存受限 |
| **cuda** | `use_topp_flash=True, backend='cuda'` | 低 | 慢 | 极致显存优化 |

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
├── configs-h/                       # 高分辨率配置
├── configs_l/                       # 低分辨率配置
└── tools/                           # 训练/推理工具
```

## 快速开始

### 安装

```bash
# 克隆仓库
git clone -b pvsa-v3.0 https://github.com/hyshhh/PVSA-v1.git
cd PVSA-v1

# 安装依赖
pip install -r requirements/mminstall.txt
pip install -r requirements/runtime.txt
```

### 训练

```bash
# 单卡训练
python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py

# 多卡训练
bash tools/dist_train.sh configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py ${GPU_NUM}
```

显式指定低显存 Top-P Flash 参数：

```bash
python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options \
  model.backbone.use_topp_flash=True \
  model.backbone.topp_flash_backend=torch_block \
  model.backbone.topp_flash_block_windows=16 \
  train_dataloader.batch_size=4
```

### 训练配置

| 配置项 | 值 | 说明 |
|--------|----|------|
| 训练轮数 | 200 epochs | 基于 `EpochBasedTrainLoop` |
| 批量大小 | 4 | 单卡训练默认值 |
| 学习率 | 6e-4 | AdamW 优化器 |
| 验证间隔 | 10 epochs | 每 10 个 epoch 验证一次 |
| 检查点保存间隔 | 10 epochs | 每 10 个 epoch 保存一次 |

### 推荐硬件

- **GPU**: NVIDIA RTX 3090 或更高（24GB 显存）
- **显存需求**: 约 8-10GB（使用 `torch_block` 后端）

### 训练环境

| 项目 | 值 |
|------|-----|
| GPU | *待填写* |
| CUDA | *待填写* |
| PyTorch | *待填写* |
| 显存占用 | *待填写* |

注意：`configs-h/_base_/models/VTFormer-s.py` 只是模型片段配置，缺少数据集、训练循环、优化器、运行时作用域等内容，不能直接用于训练。直接使用它会导致运行器配置不完整，或触发 `EncoderDecoder` 注册表查找错误。

### 推理

```bash
python tools/test.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py ${CHECKPOINT_FILE}
```

## 配置说明

### 模型配置

```python
backbone=dict(
    type='BiFormer_fusion',
    embed_dim=[64, 128, 256, 512],
    depth=[3, 4, 6, 3],
    topks=[1, 4, 16, -2],           # 每个 stage 的 topk 设置
    n_win=7,                         # 窗口数量
    use_topp_flash=True,             # 是否启用分块后端
    topp_flash_backend='torch_block', # 'torch_block' 或 'cuda'
    topp_flash_block_windows=16      # 分块大小
)
```

### topk 参数说明

- `topk > 0`：使用 ToppAttention（Top-P 稀疏注意力）
- `topk == -1`：使用标准全局注意力
- `topk == -2`：使用带局部位置编码的全局注意力（AttentionLePE）

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `PVSA_TOPP_FLASH_BACKEND` | 强制指定后端 | `torch_block` |
| `PVSA_TOPP_FLASH_STRICT_CUDA` | CUDA 失败时是否报错 | `0` |
| `PVSA_TOPP_FLASH_VERBOSE` | 打印编译日志 | `0` |
| `PVSA_TOPP_FLASH_ARCH` | 目标 GPU 架构 | 自动检测 |

## 性能对比

三种后端的显存占用对比（相对值）：

| 后端 | 显存峰值 | 推理速度 |
|------|---------|---------|
| kv_gather | 100% | 最快（小 topk） |
| torch_block | ~13% | 中等 |
| cuda | ~0% | 最慢 |

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
