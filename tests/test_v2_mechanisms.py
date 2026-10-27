import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.backbones.vit_pytorch import TransReID
from model.visa_adapt import ETGM, DLFM, make_retrieval_feature
from loss.local_cm import local_cross_modality_loss
from processor.processor import PrototypeBank, rebuild_prototype_bank


def test_specific_mask_preserves_uad_cls_and_patch11():
    torch.manual_seed(3)
    arguments = dict(img_size=(32, 32), patch_size=16, stride_size=16,
                     embed_dim=48, depth=2, num_heads=12, drop_path_rate=0)
    original = TransReID(**arguments, specific_tokens=False).eval()
    adapted = TransReID(**arguments, specific_tokens=True).eval()
    adapted.load_state_dict(original.state_dict(), strict=False)
    intermediate = []
    handle = original.blocks[0].register_forward_hook(
        lambda _, __, output: intermediate.append(output[:, 1:].detach()))
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        global_original = original(images)
        global_v2, specific, patch11 = adapted(images, modal_label=torch.tensor([0, 2]))
    handle.remove()
    torch.testing.assert_close(global_v2, global_original, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(patch11, intermediate[0], atol=1e-5, rtol=1e-5)
    assert specific.shape == (2, 48)
    assert not torch.equal(adapted.specific_tokens[0], adapted.specific_tokens[2])


def test_patch_read_and_gcn_gradient_boundaries():
    torch.manual_seed(4)
    experts = ETGM(48, experts=4, top_k=2, queries=8, heads=12)
    local_branch = DLFM(48, heads=12, graph_k=40)
    global_feature = torch.randn(6, 48, requires_grad=True)
    specific_feature = torch.randn(6, 48, requires_grad=True)
    patch11 = torch.randn(6, 128, 48, requires_grad=True)
    modality = torch.tensor([0, 1, 2, 0, 1, 2])
    shared, specific, balance = experts(global_feature, specific_feature, modality)
    local = local_branch(global_feature, specific_feature, patch11, shared, specific)
    (local.square().sum() + balance).backward()
    assert shared.shape == specific.shape == (6, 8, 48)
    assert local.shape == (6, 48)
    assert global_feature.grad is None  # CLS condition and shared pool inputs detach.
    assert specific_feature.grad is not None and specific_feature.grad.abs().sum() > 0
    assert patch11.grad is not None and patch11.grad.abs().sum() > 0
    assert local_branch.shared_read.residual_scale.grad is not None
    assert local_branch.shared_gcn.weight.grad is not None


def test_fusion_scale_and_stop_gradient():
    global_feature = torch.randn(3, 48, requires_grad=True)
    local_feature = torch.randn(3, 48, requires_grad=True)
    fusion = make_retrieval_feature(global_feature, local_feature,
                                    detach_global_for_loss=True)
    torch.testing.assert_close(fusion.norm(dim=1), torch.ones(3), atol=1e-6, rtol=1e-6)
    fusion[:, 48:].sum().backward()
    assert global_feature.grad is None
    assert local_feature.grad is not None


def test_cross_modality_camera_filter_and_safe_zero():
    features = torch.randn(6, 48, requires_grad=True)
    pids = torch.tensor([0, 0, 0, 1, 1, 1])
    modalities = torch.tensor([0, 1, 2, 0, 1, 2])
    cameras = torch.tensor([0, 1, 2, 1, 2, 3])
    loss, coverage = local_cross_modality_loss(features, pids, cameras, modalities)
    assert len(coverage) == 6 and any(valid for valid, _ in coverage.values())
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    zero, coverage = local_cross_modality_loss(features, pids,
                                                torch.zeros_like(cameras), modalities)
    assert zero.item() == 0 and all(valid == 0 for valid, _ in coverage.values())


def test_gpd_bank_rebuild_uses_equal_modality_centers():
    class MockBase(torch.nn.Module):
        def forward(self, images, modal_label=None):
            return images, None, None

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = MockBase()

    features = torch.tensor([[2., 0.], [0., 2.], [1., 1.],
                             [0., 2.], [2., 0.], [1., -1.]])
    pids = (0, 0, 0, 1, 1, 1)
    modalities = (1, 2, 3, 1, 2, 3)
    loader = [(features, pids, None, None, modalities, None)]
    bank = PrototypeBank(2, 2, device='cpu')
    rebuild_prototype_bank(MockModel(), loader, bank, 'cpu')
    assert torch.isfinite(bank.prototypes).all()
    torch.testing.assert_close(bank.prototypes.norm(dim=1), torch.ones(2))
