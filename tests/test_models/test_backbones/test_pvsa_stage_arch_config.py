from pathlib import Path


def test_vtformer_exposes_stage_arch_config():
    root = Path(__file__).resolve().parents[3]
    source = (
        root / 'mmseg' / 'models' / 'backbones' / 'bi_topp_vote.py'
    ).read_text(encoding='utf-8')
    config = (
        root / 'configs-h' / '_base_' / 'models' / 'VTFormer-s.py'
    ).read_text(encoding='utf-8')

    assert 'stage_archs=None' in source
    assert 'extra_block_type=None' in source
    assert "cfg['blocks']" in source
    assert "cfg.get('trans_extra')" in source
    assert "cfg.get('cnn_extra')" in source
    assert "self.stage_archs[0]['trans_extra']" in source
    assert "self.stage_archs[0]['cnn_extra']" in source
    assert 'class MBConvModule' in source
    assert 'class ConvNeXtBlock' in source
    assert 'stage_archs=[' in config
    assert "extra_block_type='dwconv'" in config
    assert "trans_extra=dict(depth=0)" in config
    assert "cnn_extra=dict(depth=2)" in config
    assert "'dwconv', 'mbconv', 'convnext'" in source


def test_vtformer_keeps_legacy_depth_configs_as_fallback():
    root = Path(__file__).resolve().parents[3]
    source = (
        root / 'mmseg' / 'models' / 'backbones' / 'bi_topp_vote.py'
    ).read_text(encoding='utf-8')

    assert 'transformer_branch_depth=None' in source
    assert 'cnn_branch_depth=None' in source
    assert '_normalize_stage_archs(' in source
