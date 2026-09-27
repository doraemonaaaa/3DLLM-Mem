import torch

from three_dllm_mem.geometry import farthest_point_sample, unproject_depth


def test_unproject_identity_camera() -> None:
    depth = torch.ones(1, 1, 2, 2)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    pose = torch.eye(4).reshape(1, 1, 4, 4)
    points = unproject_depth(depth, intrinsics, pose)
    assert torch.equal(points[0, 0, 1, 1], torch.tensor([1.0, 1.0, 1.0]))


def test_fps_uses_only_valid_points() -> None:
    points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0]]])
    indices = farthest_point_sample(points, torch.tensor([[True, False, True]]), 3)
    assert indices.tolist() == [[0, 2, 0]]
