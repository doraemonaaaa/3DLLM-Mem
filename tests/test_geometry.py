import torch

from three_dllm_mem.geometry import SpatialTokenEncoder, farthest_point_sample, unproject_depth


def test_unproject_identity_camera() -> None:
    depth = torch.ones(1, 1, 2, 2)
    camera = torch.eye(3).view(1, 1, 3, 3)
    pose = torch.eye(4).view(1, 1, 4, 4)
    points = unproject_depth(depth, camera, pose)
    assert torch.allclose(points[0, 0, 1, 1], torch.tensor([1.0, 1.0, 1.0]))


def test_fps_returns_spatially_separated_endpoints() -> None:
    points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    indices = farthest_point_sample(points, torch.ones(1, 3, dtype=torch.bool), 2)
    assert indices.tolist() == [[0, 2]]


def test_spatial_encoder_outputs_fixed_token_count() -> None:
    encoder = SpatialTokenEncoder(feature_dim=4, num_tokens=3)
    features = torch.randn(1, 1, 2, 2, 4)
    depth = torch.ones(1, 1, 2, 2)
    intrinsic = torch.eye(3).view(1, 1, 3, 3)
    pose = torch.eye(4).view(1, 1, 4, 4)
    tokens, points = encoder(features, depth, intrinsic, pose)
    assert tokens.shape == (1, 3, 4)
    assert points.shape == (1, 3, 3)
