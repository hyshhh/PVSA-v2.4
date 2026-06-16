# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# data_preprocessor = dict(
#     type='SegDataPreProcessor',
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     bgr_to_rgb=True,
#     pad_val=0,
#     seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='BiFormer_fusion',
        embed_dim=[64, 128, 256, 512],     # BiFormer的通道配置
        # 一键切换所有额外骨干模块，支持'dwconv'、'mbconv'、'convnext'。
        # 只改这里即可全局切换，stage_archs里只保留每层depth。
        extra_block_type='dwconv',
        # 每个stage的结构：
        # blocks是Transformer Block数量；
        # trans_extra/cnn_extra分别控制两条分支下采样后的额外骨干模块数量。
        stage_archs=[
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(depth=2)),
            dict(
                blocks=4,
                trans_extra=dict(depth=0),
                cnn_extra=dict(depth=1)),
            dict(
                blocks=6,
                trans_extra=dict(depth=0),
                cnn_extra=dict(depth=2)),
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(depth=1)),
        ],
        mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        # 四层网络的 Top-P 路由标志位。每个标志位会到
        # topp_route_configs 中查出真实 maxk、P 阈值、温度和能量补偿。
        # 当前默认值等价于重构前 bi_topp_vote.py 中硬编码的 [16, 12, 8, 6]。
        topks=[16, 12, 8, 6],
        # Top-P v3 路由参数表：
        # maxk 表示 torch.topk 的最大候选窗口数；
        # p 表示累计概率裁剪阈值；
        # temperature 表示路由得分 softmax 温度；
        # energy 表示裁剪后的能量补偿系数。
        # 测试不同 maxk/p 时，直接修改这里的值，不要用 --cfg-options（无法覆盖整数键）
        topp_route_configs={
            16: dict(maxk=25, p=0.2, temperature=0.0175, energy=4.0),
            12: dict(maxk=18, p=0.4, temperature=0.025, energy=1.5),
            8: dict(maxk=36, p=0.6, temperature=0.05, energy=0.75),
            6: dict(maxk=49, p=0.8, temperature=0.15, energy=0.4),
        },
        # 调试开关：打印每层路由的 maxk、p、max_len、keep_len 统计
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
        # Top-P 推理后端切换：
        # - topp_flash_backend=None：训练和普通推理路径。
        # - topp_flash_backend='cuda'：自定义 CUDA 路由/注意力推理路径。
        # 最后一层默认使用 49 个窗口全连接路由，不再暴露额外开关。
        topp_flash_backend=None,
        topp_flash_block_windows=16,
        # 打开后打印 PVSA TopP Stage，包括 Transformer、Router kernel、
        # Flash kernel、CNN 分支和融合模块耗时。
        topp_flash_debug=False,
        use_pruned_kv_gather=False,
        # pruned_kv_gather 粗分桶数量。按 keep_len 范围切成 N 个桶，
        # 例如 topk=8 且 N=2 时，桶范围为 1-4 和 5-8。
        # 桶数越少，循环和索引开销越低；桶数越多，padding 浪费越小。
        pruned_kv_num_groups=3,
        use_fast_attention=False,
        # 特征图保存开关。训练默认关闭；打开后会把 FAM 前后和融合后的
        # 特征图保存到 save_dir，频繁写图会明显降低训练速度。
        feature_vis_config=dict(
            enabled=False,
            save_dir='cam/features_imgs4',
            out_size=512,
            channel_reduce='mean'),
        # 注意力图保存开关。训练默认关闭；打开后可保存指定 query 的
        # 热力图或 Top-K 聚光灯图。trigger_maxk=25 对应原来只在
        # 第一层标志位 16 映射到 maxk=25 时保存。
        attn_vis_config=dict(
            enabled=False,
            save_heatmap=False,
            save_topk=True,
            query_index=32,
            trigger_maxk=25,
            image_path=(
                '/media/ddc/新加卷/hys/ljf/mmsegmentation-main/'
                'mmsegmentation-main/data/gqyyz/image/test1/'
                '2024_08_19_195748172_cropped_1.jpg'),
            heatmap_save_path=(
                '/media/ddc/新加卷/hys/ljf/mmsegmentation-main/'
                'mmsegmentation-main/cam/attn/attn_stage.png'),
            topk_save_path=(
                '/media/ddc/新加卷/hys/ljf/mmsegmentation-main/'
                'mmsegmentation-main/cam/attn/attn_stage_topk.png'),
            dark_ratio=0.3,
            once=True)
    ),
    # decode_head=dict(
    #     type='UPerHead',
    #     in_channels=[64, 128, 256, 512],   # 对应backbone输出通道
    #     in_index=[0, 1, 2, 3],
    #     pool_scales=(1, 2, 3, 6),
    #     channels=512,
    #     dropout_ratio=0.1,
    #     num_classes=2,                    # 类别数（根据数据集修改）
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    # ),
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
    # 模型训练与推理配置
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
