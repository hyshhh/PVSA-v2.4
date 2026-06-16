from pathlib import Path


def test_topp_attention_stage_log_requires_stage_debug():
    root = Path(__file__).resolve().parents[3]
    source = (
        root / 'mmseg' / 'models' / 'utils' / 'top_p_bra.py'
    ).read_text(encoding='utf-8')
    log_call = source.index('_log_topp_stage_debug(', source.index('log_path ='))
    guard = source.rfind('if stage_debug:', 0, log_call)

    assert guard != -1
    assert log_call - guard < 200
