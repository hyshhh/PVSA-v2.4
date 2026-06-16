from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[3]


def test_block_supports_non_topp_attention_paths():
    source = (
        _repo_root() / 'mmseg' / 'models' / 'backbones' / 'bi_topp_vote.py'
    ).read_text(encoding='utf-8')

    assert 'self.use_topp_attention = True' in source
    assert 'self.use_topp_attention = False' in source
    assert 'attn_fn = lambda tensor: self.PA(tensor, None)' in source
    assert 'attn_fn = self.attn' in source


def test_topp_routing_uses_query_and_key_config():
    source = (
        _repo_root() / 'mmseg' / 'models' / 'utils' / 'top_p_bra.py'
    ).read_text(encoding='utf-8')

    assert 'q = F.normalize(self.emb(query), dim=-1)' in source
    assert 'k = F.normalize(self.emb(key), dim=-1)' in source
    assert 'self.soft_routing = soft_routing' in source
    assert 'self.soft_routing = True' not in source


def test_cuda_route_kernel_accepts_query_and_key():
    root = _repo_root()
    kernel = (
        root / 'mmseg' / 'models' / 'utils' / 'topp_flash_kernel.py'
    ).read_text(encoding='utf-8')
    cpp = (
        root / 'mmseg' / 'ops' / 'topp_flash' / 'topp_flash.cpp'
    ).read_text(encoding='utf-8')
    cuda = (
        root / 'mmseg' / 'ops' / 'topp_flash' / 'topp_flash_cuda.cu'
    ).read_text(encoding='utf-8')

    assert 'def topp_route_cuda(query: Tensor,' in kernel
    assert 'key: Tensor,' in kernel
    assert 'extension.route_forward(' in kernel
    assert 'query.contiguous(), key.contiguous()' in kernel
    assert 'std::vector<torch::Tensor> topp_route_forward(torch::Tensor query,' in cpp
    assert 'torch::Tensor key,' in cpp
    assert 'const float *__restrict__ key,' in cuda
    assert 'const float kv = key[k_base + d];' in cuda


def test_fusion_head_does_not_allocate_unused_mask_stage():
    source = (
        _repo_root() / 'mmseg' / 'models' / 'backbones' / 'biformer_fusion.py'
    ).read_text(encoding='utf-8')

    assert 'for i in range(3):' in source
    assert 'self.extra_norms.append(LayerNorm2d(self.embed_dim[3]))' in source
    assert 'def _time_cuda_stage' not in source
