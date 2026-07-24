import torch

from three_dllm_mem.model import SpatialTemporalMemory


def test_first_observation_has_no_history_then_fuses_after_commit() -> None:
    model = SpatialTemporalMemory(feature_dim=4, memory_dim=6, num_tokens=2)
    features = torch.randn(1, 1, 2, 2, 4)
    depth = torch.ones(1, 1, 2, 2)
    intrinsic = torch.eye(3).view(1, 1, 3, 3)
    pose = torch.eye(4).view(1, 1, 4, 4)
    first, no_history = model.observe(features, depth, intrinsic, pose, scene_id="room-a", timestep=1)
    assert first.shape == (1, 2, 4)
    assert no_history.shape == (1, 2, 0)
    model.commit()
    fused, attention = model.observe(features, depth, intrinsic, pose, scene_id="room-b", timestep=2)
    assert fused.shape == (1, 2, 12)
    assert attention.shape == (1, 2, 2)
