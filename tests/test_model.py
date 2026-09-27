import torch

from three_dllm_mem.model import SpatialTemporalMemory


def test_controller_observe_commit_and_retrieve() -> None:
    model = SpatialTemporalMemory(feature_dim=4, memory_dim=6, num_tokens=2)
    patches = torch.randn(1, 1, 2, 2, 4)
    depth = torch.ones(1, 1, 2, 2)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    pose = torch.eye(4).reshape(1, 1, 4, 4)
    first, first_attention = model.observe(patches, depth, intrinsics, pose, scene_id="a", timestep=0)
    assert first.shape == (1, 2, 4)
    assert first_attention.shape[-1] == 0
    model.commit()
    second, attention = model.observe(patches, depth, intrinsics, pose, scene_id="b", timestep=1)
    assert second.shape == (1, 2, 12)
    assert attention.shape == (1, 2, 2)
