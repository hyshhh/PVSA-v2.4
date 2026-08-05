# 对比实验统一使用的数据集设置；继承原有 YZ 数据集，不改动原文件。
_base_ = ['gqy.py']

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = dict(batch_size=1, num_workers=2)
