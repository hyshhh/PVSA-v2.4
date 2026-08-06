from pathlib import Path

import torch
import torch.nn as nn

from tools.analysis_tools.compare.pvsa_fair_timer import PVSAFairStageTimer


class _Attention(nn.Module):
    def forward(self, x):
        return x + 0.0


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.PA = _Attention()

    def forward(self, x):
        return self.PA(x)


class _FAM(nn.Module):
    def forward(self, x, cnn):
        return x + 0.0, cnn + 0.0


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample_layers2 = nn.ModuleList(
            [nn.Identity() for _ in range(4)])
        self.downsample_layers = nn.ModuleList(
            [nn.Identity() for _ in range(4)])
        self.stages = nn.ModuleList([
            nn.Sequential(_Block()) for _ in range(4)])
        self.FAM = nn.ModuleList([_FAM() for _ in range(4)])

    def forward(self, x):
        cnn = x
        for index in range(4):
            cnn = self.downsample_layers2[index](cnn)
            x = self.downsample_layers[index](x)
            x = self.stages[index](x)
            x, cnn = self.FAM[index](x, cnn)
        return x


def test_pvsa_fair_timer_reports_attention_and_outer_stage_total():
    root = Path(__file__).resolve().parents[3]
    source = (root / 'tools' / 'analysis_tools' / 'compare' /
              'pvsa_fair_timer.py').read_text(encoding='utf-8')
    assert 'S1_total' in source
    assert '_make_total_pre_hook' in source
    assert '_make_total_post_hook' in source
    assert 'downsample_layers2' in source
    assert 'consume_graph_replay' in source
    assert 'external=True' in source

    benchmark_source = (root / 'tools' / 'analysis_tools' / 'compare' /
                        'benchmark_compare.py').read_text(encoding='utf-8')
    assert 'capture_timing=True' in benchmark_source
    assert 'graph_attention_profile' in benchmark_source

    backbone = _Backbone().eval()
    timer = PVSAFairStageTimer(backbone, model_name='test_pvsa', interval=1)
    timer.configure(enabled=True, interval=1)

    with torch.inference_mode():
        backbone(torch.randn(2, 1, 4, 4))
    report = timer.flush()
    timer.close()

    assert report is not None
    assert report['images'] == 2
    for stage in ('S1', 'S2', 'S3', 'S4'):
        assert report[stage] > 0.0
        assert report[f'{stage}_total'] > 0.0
