_base_ = [
    '../_base_/models/VTFormer-s.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
#tmux attach -t hys7  unset CUDA_LAUNCH_BLOCKING(解除锁定)
# python tools/test.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py 
#/media/ddc/新加卷/hys/ljf/mmsegmentation-main/mmsegmentation-main/qmy/city/3.11PVSA-Net-s/last_checkpoint --show-dir  qmy/test
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py --work-dir qmy/gqy/3.11
# python tools/analysis_tools/get_flops.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py --shape 224 224
# --------------------------
# 数据预处理配置
# --------------------------
crop_size = (256, 256)
# crop_size = (224, 224)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)
# --------------------------
# 训练配置
# --------------------------
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True)
)
val_dataloader = dict(batch_size=4, num_workers=2)
test_dataloader = dict(batch_size=1, num_workers=1)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,  # 不要太小
        by_epoch=True,
        begin=0,
        end=10
    ),
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=1.0,
        by_epoch=True,
        begin=10,
        end=200  # 总训练 50 epoch
    )
]
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=200,      # 总训练轮数
    val_interval=10      # 每训练 1 个 epoch 验证一次
)
#配置2，camvid
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.001,  # 不要太小
#         by_epoch=True,
#         begin=0,
#         end=10
#     ),
#     dict(
#         type='PolyLR',
#         eta_min=1e-6,
#         power=1.0,
#         by_epoch=True,
#         begin=10,
#         end=1000  # 总训练 50 epoch
#     )
# ]
# train_cfg = dict(
#     type='EpochBasedTrainLoop',
#     max_epochs=1000,      # 总训练轮数
#     val_interval=2      # 每训练 1 个 epoch 验证一次
# )


# --------------------------
# 优化器与学习率Ddc134567ddc.
# --------------------------
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0)
        })
)


# --------------------------
# 运行时设置
# --------------------------

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# # 评估配置
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    ignore_index=255,  # 避免把填充值算进指标
    classwise=True
)
test_evaluator = val_evaluator

# --------------------------
# 模型额外配置（仅修改部分）
# --------------------------
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='whole')
)
checkpoint_config = dict(by_epoch=True, interval=10)  # 每个 epoch 保存一次
