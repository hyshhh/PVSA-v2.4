"""全链路正确性验证：CUDA Graph 模式 vs 普通 predict 模式输出对比"""
import argparse
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--input-size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--num-images', type=int, default=5)
    args = parser.parse_args()

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda()
    model.eval()

    # 用 dataloader 取真实图片
    from mmengine.runner import Runner as R
    dl = R.build_dataloader(cfg.test_dataloader)
    data_iter = iter(dl)

    print('=== 对比 CUDA Graph 捕获的 _forward vs 普通 predict ===')
    for idx in range(args.num_images):
        data = next(data_iter)
        data = model.data_preprocessor(data, False)
        inputs = data['inputs']
        samples = data['data_samples']

        with torch.no_grad():
            # 普通 predict（完整流程）
            out_predict = model(inputs, samples, mode='predict')
            # CUDA Graph 捕获的 _forward（纯 GPU forward）
            logits_graph = model._forward(inputs, samples)
            # 补 predict_by_feat + postprocess（复刻 benchmark 的 cuda-graph 路径）
            metas = [s.metainfo for s in samples]
            seg_logits = model.decode_head.predict_by_feat(logits_graph, metas)
            out_graph = model.postprocess_result(seg_logits, samples)

        # 对比 pred_sem_seg
        pred_p = out_predict[0].pred_sem_seg.data
        pred_g = out_graph[0].pred_sem_seg.data
        same = torch.equal(pred_p, pred_g)
        # 数值差异
        diff = (pred_p.float() - pred_g.float()).abs()
        print(f'img[{idx}] shape={tuple(pred_p.shape)} '
              f'完全一致={same} '
              f'max_diff={diff.max().item():.2f} '
              f'不同像素数={(diff > 0).sum().item()}')

    print('=== 验证完成 ===')


if __name__ == '__main__':
    main()
