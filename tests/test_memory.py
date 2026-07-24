import torch

from three_dllm_mem.memory import EpisodicMemoryBank, MemoryFusion


def test_revisit_replaces_memory_entry() -> None:
    bank = EpisodicMemoryBank()
    bank.write("kitchen", torch.ones(2, 4), timestep=1)
    bank.write("kitchen", torch.full((3, 4), 2.0), timestep=5)
    assert len(bank) == 1
    state = bank.as_state()
    assert state.tokens.shape == (1, 1, 3, 4)
    assert state.timesteps.item() == 5
    assert torch.all(state.tokens[0, 0] == 2.0)


def test_fusion_masks_padding_and_produces_expected_shapes() -> None:
    torch.manual_seed(0)
    bank = EpisodicMemoryBank()
    bank.write("bedroom", torch.randn(2, 4), timestep=1)
    bank.write("kitchen", torch.randn(1, 4), timestep=3)
    fusion = MemoryFusion(input_dim=4, memory_dim=6)
    output, attention = fusion(torch.randn(1, 3, 4), bank.as_state())
    assert output.shape == (1, 3, 12)
    assert attention.shape == (1, 3, 4)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(1, 3))
    # The fourth flattened token is padding for the one-token kitchen entry.
    assert torch.all(attention[..., 3] == 0)


def test_temporal_embeddings_change_across_timesteps() -> None:
    fusion = MemoryFusion(input_dim=4, memory_dim=6)
    embedding = fusion.temporal_embedding(torch.tensor([[1, 2]]))
    assert not torch.allclose(embedding[:, 0], embedding[:, 1])
