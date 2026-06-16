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
    assert "cfg['blocks']" in source
    assert "cfg['trans_dwconv']" in source
    assert "cfg['cnn_dwconv']" in source
    assert 'stage_archs=[' in config
    assert 'dict(blocks=3, trans_dwconv=0, cnn_dwconv=2)' in config


def test_vtformer_keeps_legacy_depth_configs_as_fallback():
    root = Path(__file__).resolve().parents[3]
    source = (
        root / 'mmseg' / 'models' / 'backbones' / 'bi_topp_vote.py'
    ).read_text(encoding='utf-8')

    assert 'transformer_branch_depth=None' in source
    assert 'cnn_branch_depth=None' in source
    assert '_normalize_stage_archs(' in source
