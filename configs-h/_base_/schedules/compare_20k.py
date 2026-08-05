# 对比实验统一训练策略：同一数据集、同一轮数、同一优化器。
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.001,
         begin=0,
         end=5,
         by_epoch=True),
    dict(type='PolyLR',
         eta_min=1e-6,
         power=1.0,
         begin=5,
         end=80,
         by_epoch=True)]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=5,
                    save_best='auto', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
