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
        # Top-P v3 路由参数表（对齐 vision-topp-cnn.py）
        topp_route_configs={
            16: dict(maxk=5, mink=1, p=0.2, temperature=0.5, energy=3.0),
            12: dict(maxk=10, mink=3, p=0.6, temperature=4, energy=6.0),
            8: dict(maxk=25, mink=5, p=0.6, temperature=8, energy=12.0),
        },
        debug_route=False,
        # 自注意力类型: 'topp'=Top-P路由(PVSA) / 'bra'=标准Bi-Level Routing Attention
        attention_type='topp',
        # 路由 mask 硬屏蔽
        use_route_mask=True,
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        auto_pad=True,
        # FAM 空间注意力降维比例（1=无降维，4=压缩到1/4通道）
        fam_reduction=4,
        # FAM 内部 CA/SA 开关：关了就不建对应参数
        fam_use_channel=True,
        fam_use_spatial=True,
        # CNN 分支各层 block 数量：[stem, stage1, stage2, stage3]
        cnn_block_layers=[2, 1, 2, 1],
        # CNN block 类型：
        # 'dwconv' | 'dwconv_act' | 'mbconv' | 'mbconv_no_se'
        # 'c2f' | 'c3k2' | 'convnext'
        cnn_block_type='mbconv_no_se',
        # CUDA 推理后端
        topp_flash_backend=None,
        topp_flash_block_windows=64,
        topp_flash_debug=False,
        # 路由 token 池化方式: 'avg' | 'max' | 'avgmax'
        # （注：v2.4 ToppAttention 固定使用 avgmax，此配置保留以对齐 vision-topp-cnn）
        route_pooling='avg',
        # 最后一层 stage 用普通 self-attention 替代 ToppAttention
        use_plain_attn_last_stage=True,
        # 特征图保存开关
        feature_vis_config=dict(
            enabled=False,              # True 开启保存
            save_dir='cam/features_imgs4',  # 保存目录
            out_size=512,               # 上采样目标尺寸
            channel_reduce='mean'),     # 通道聚合方式：'mean' | 'max'
        # 同层融合消融: 'conv1x1' | 'conv1x1_bn_gelu' | 'conv1x1_bn_gelu_dwconv'
        fusion_type='conv1x1_bn_gelu_dwconv',
        # 跨层融合消融:
        # 'none' | 'gate' | 'concat' | 'gate_concat' | 'cross_gate' | 'cross_concat'
        # 'mask'（v2.4 旧版固定 mask 调制）
        cross_stage_fusion_mode='cross_concat',
        # 特征图保存开关
        feature_vis_config=dict(
            enabled=False,              # True 开启保存
            save_dir='cam/features_imgs4',  # 保存目录
            out_size=512,               # 上采样目标尺寸
            channel_reduce='mean'),     # 通道聚合方式：'mean' | 'max'
        # 注意力图保存开关
        attn_vis_config=dict(
            enabled=False,              # True 开启保存
            save_heatmap=False,         # 是否保存叠加热力图
            save_topk=True,             # 是否保存 top-k 窗口选择图
            query_index=32,             # 可视化的 query 窗口索引
            trigger_maxk=25,            # 只在 topk==25 时触发（None=始终触发）
            image_path='',              # 叠加热力图的背景图片路径（必须配置）
            heatmap_save_path='cam/attn_vis/heatmap.png',    # 热力图保存路径
            topk_save_path='cam/attn_vis/topk_select.png',   # top-k 选择图保存路径
            dark_ratio=0.3,             # 背景暗化比例
            once=True)                  # True=只保存第一张图
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
