"""对比实验中的分层 ViT 主干，四个阶段均使用全局自注意力。"""

import torch.nn as nn

from mmseg.registry import MODELS

from .attention_timing import CompareAttentionBlock, CompareBackboneBase
from .biformer_compare import GlobalSelfAttention


_VIT_VARIANTS = {
    "tiny": dict(
        embed_dims=(48, 96, 192, 384),
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24)),
    "small": dict(
        embed_dims=(64, 128, 256, 512),
        depths=(2, 2, 8, 2),
        num_heads=(4, 8, 16, 32)),
    "base": dict(
        embed_dims=(96, 192, 384, 768),
        depths=(2, 2, 8, 2),
        num_heads=(6, 12, 24, 48)),
}


@MODELS.register_module()
class CompareViT(CompareBackboneBase):
    """ViT-T、ViT-S、ViT-B 的统一实现。"""

    def __init__(self,
                 variant: str = "tiny",
                 mlp_ratio: float = 4.0,
                 **kwargs) -> None:
        variant = str(variant).lower()
        aliases = {"t": "tiny", "s": "small", "b": "base"}
        variant = aliases.get(variant, variant)
        if variant not in _VIT_VARIANTS:
            raise KeyError(
                f"未知 ViT 版本 {variant}，可选值为 {tuple(_VIT_VARIANTS)}")
        spec = dict(_VIT_VARIANTS[variant])
        embed_dims = kwargs.pop("embed_dims", spec["embed_dims"])
        depths = kwargs.pop("depths", spec["depths"])
        num_heads = kwargs.pop("num_heads", spec["num_heads"])
        self.variant = variant
        model_name = kwargs.pop("model_name", f"vit_{variant}")
        in_chans = kwargs.pop("in_chans", 3)
        super().__init__(
            in_chans=in_chans,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            model_name=model_name,
            **kwargs)

    def _make_block(self, stage_idx: int, block_idx: int,
                    drop_path: float, mlp_ratio: float) -> nn.Module:
        attention = GlobalSelfAttention(
            dim=self.embed_dims[stage_idx],
            num_heads=self.num_heads[stage_idx])
        return CompareAttentionBlock(
            dim=self.embed_dims[stage_idx],
            attention=attention,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path)


__all__ = ["CompareViT"]
