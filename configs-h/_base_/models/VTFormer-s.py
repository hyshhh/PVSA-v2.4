# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='BiFormer_fusion',
        embed_dim=[64, 128, 256, 512],
        depth=[3, 4, 6, 3],
        mlp_ratios=[3, 3, 3, 3],
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[16, 12, 8, 6],
        # Top-P v3 路由参数表
        topp_route_configs={
            16: dict(maxk=25, p=0.2, temperature=0.0175, energy=4.0),
            12: dict(maxk=18, p=0.4, temperature=0.025, energy=1.5),
            8: dict(maxk=36, p=0.6, temperature=0.05, energy=0.75),
            6: dict(maxk=49, p=0.8, temperature=0.15, energy=0.4),
        },
        debug_route=False,
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        auto_pad=True,
        # CUDA 推理后端
        topp_flash_backend=None,
        topp_flash_block_windows=64,
        topp_flash_debug=False,
        # 特征图保存开关
        feature_vis_config=dict(
            enabled=False,
            save_dir='cam/features_imgs4',
            out_size=512,
            channel_reduce='mean'),
        # 注意力图保存开关
        attn_vis_config=dict(
            enabled=False,
            save_heatmap=False,
            save_topk=True,
            query_index=32,
            trigger_maxk=25,
            image_path='',
            heatmap_save_path='',
            topk_save_path='',
            dark_ratio=0.3,
            once=True)
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
