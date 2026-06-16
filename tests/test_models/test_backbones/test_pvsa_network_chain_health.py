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


def test_fusion_and_route_mechanism_options_are_configurable():
    root = _repo_root()
    backbone = (
        root / 'mmseg' / 'models' / 'backbones' / 'bi_topp_vote.py'
    ).read_text(encoding='utf-8')
    fusion = (
        root / 'mmseg' / 'models' / 'backbones' / 'biformer_fusion.py'
    ).read_text(encoding='utf-8')
    route = (
        root / 'mmseg' / 'models' / 'utils' / 'top_p_bra.py'
    ).read_text(encoding='utf-8')
    model_cfg = (
        root / 'configs-h' / '_base_' / 'models' / 'VTFormer-s.py'
    ).read_text(encoding='utf-8')
    dataset_cfg = (
        root / 'configs-h' / '_base_' / 'datasets' / 'gqy.py'
    ).read_text(encoding='utf-8')

    assert 'fam_stages=(0, 1, 2, 3)' in backbone
    assert "mask_source='branch_low'" in backbone
    assert "mask_source must be one of 'branch_low' or 'fused_low'." in backbone
    assert 'route_pooling=' in backbone
    assert 'if i in self.fam_stages:' in fusion
    assert 'self.bn11 = nn.ModuleList()' in fusion
    assert 'self.bn12 = nn.ModuleList()' in fusion
    assert "if self.mask_source == 'branch_low':" in fusion
    assert 'mask_source1 = channel1[i]' in fusion
    assert 'mask_source2 = channel2[i]' in fusion
    assert 'mask_source1 = channel3[i]' in fusion
    assert 'mask_source2 = channel3[i]' in fusion
    assert "route_pooling must be one of 'avg', 'max', or 'avgmax'." in route
    assert 'q_route = 0.5 * (q.mean([2, 3]) + q.amax(dim=(2, 3)))' in route
    assert 'fam_stages=[0, 1, 2, 3]' in model_cfg
    assert "mask_source='branch_low'" in model_cfg
    assert "route_pooling='avgmax'" in model_cfg
    assert 'img_scale = (224, 224)' in dataset_cfg
    assert 'crop_size = (224, 224)' in dataset_cfg
