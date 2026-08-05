"""对比实验专用模型，单独归档，不改动原始 PVSA-Net 主干。"""

from .biformer_compare import CompareBiFormer
from .swin_compare import CompareSwin
from .vit_compare import CompareViT

__all__ = ["CompareBiFormer", "CompareSwin", "CompareViT"]
