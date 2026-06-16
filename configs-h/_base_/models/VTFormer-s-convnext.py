_base_ = './VTFormer-s.py'

# ConvNeXt版额外骨干：
# - 全局切换为ConvNeXt块，增强局部建模。
# - 深度保持和原始DWConv版一致，只切换块类型，方便公平对比。
# - 使用很小的layer_scale，让新增块初始更接近恒等映射，降低训练塌缩风险。
model = dict(
    backbone=dict(
        extra_block_type='convnext',
        stage_archs=[
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=2, expansion=2, kernel_size=7,
                    layer_scale=1e-6)),
            dict(
                blocks=4,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=1, expansion=2, kernel_size=7,
                    layer_scale=1e-6)),
            dict(
                blocks=6,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=2, expansion=2, kernel_size=7,
                    layer_scale=1e-6)),
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=1, expansion=2, kernel_size=7,
                    layer_scale=1e-6)),
        ]))
