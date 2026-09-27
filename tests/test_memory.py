import torch

from three_dllm_mem.memory import EpisodicMemoryBank, MemoryFusion


def test_scene_revisit_replaces_entry() -> None:
    bank = EpisodicMemoryBank()
    bank.write("room", torch.ones(2, 4), 1)
    bank.write("room", torch.full((3, 4), 2.0), 5)
    state = bank.as_state()
    assert len(bank) == 1
    assert state.tokens.shape == (1, 1, 3, 4)
    assert state.timesteps.item() == 5


def test_fusion_masks_padded_memory() -> None:
    bank = EpisodicMemoryBank()
    bank.write("a", torch.randn(2, 4), 1)
    bank.write("b", torch.randn(1, 4), 2)
    fused, attention = MemoryFusion(4, 6)(torch.randn(1, 3, 4), bank.as_state())
    assert fused.shape == (1, 3, 12)
    assert attention.shape == (1, 3, 4)
    assert torch.allclose(attention[:, :, 3], torch.zeros(1, 3))
