_base_ = './VTFormer-s.py'

# MBConv版额外骨干：
# - 全局切换为MBConv，便于一键对比。
# - 深度保持和原始DWConv版一致，只切换块类型，方便公平对比。
# - Transformer分支额外层保持0，避免路由注意力前再堆卷积导致推理变慢。
model = dict(
    backbone=dict(
        extra_block_type='mbconv',
        stage_archs=[
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=2, expansion=3, kernel_size=3,
                    layer_scale=1e-6)),
            dict(
                blocks=4,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=1, expansion=3, kernel_size=3,
                    layer_scale=1e-6)),
            dict(
                blocks=6,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=2, expansion=3, kernel_size=3,
                    layer_scale=1e-6)),
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=1, expansion=3, kernel_size=3,
                    layer_scale=1e-6)),
        ]))
