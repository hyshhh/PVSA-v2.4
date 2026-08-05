# BiFormer-T 在 YZ 数据集上的对比实验配置
_base_ = [
    '../_base_/models/compare/biformer_t.py',
    '../_base_/datasets/compare_gqy.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/compare_20k.py'
]

custom_imports = dict(
    imports=['mmseg.models.backbones.compare'],
    allow_failed_imports=False)

# 固定验证和测速输入尺寸；测速脚本也可以通过 --input-size 覆盖。
model = dict(
    data_preprocessor=dict(size=(256, 256)),
    test_cfg=dict(mode='whole'))

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    ignore_index=255,
    classwise=True)
test_evaluator = val_evaluator
