# 对比实验：ViT-T
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(256, 256))
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='CompareViT',
        variant='tiny',
        model_name='vit_tiny',
        timing=dict(enabled=False, interval=100)),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[48, 96, 192, 384],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
