"""RGB-D geometry utilities for constructing spatially grounded 3D tokens."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def unproject_depth(depth: Tensor, intrinsics: Tensor, extrinsics: Tensor) -> Tensor:
    """Back-project depth maps into world coordinates.

    Args:
        depth: `[B, V, H, W]` metric depth maps.
        intrinsics: `[B, V, 3, 3]` pinhole camera matrices.
        extrinsics: `[B, V, 4, 4]` camera-to-world transformations.
    Returns:
        World points `[B, V, H, W, 3]`.
    """
    if depth.ndim != 4:
        raise ValueError("depth must have shape [B, V, H, W]")
    batch, views, height, width = depth.shape
    if intrinsics.shape != (batch, views, 3, 3) or extrinsics.shape != (batch, views, 4, 4):
        raise ValueError("camera matrix shapes do not match depth")
    rows, cols = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pixels = torch.stack((cols, rows, torch.ones_like(rows)), dim=-1)
    pixels = pixels.view(1, 1, height * width, 3).expand(batch, views, -1, -1)
    camera_rays = torch.matmul(pixels, torch.linalg.inv(intrinsics).transpose(-1, -2))
    camera_points = camera_rays * depth.reshape(batch, views, -1, 1)
    homogeneous = torch.cat((camera_points, torch.ones_like(camera_points[..., :1])), dim=-1)
    world = torch.matmul(homogeneous, extrinsics.transpose(-1, -2))[..., :3]
    return world.view(batch, views, height, width, 3)


class FourierPositionEmbedding(nn.Module):
    """Learnable projection of fixed Fourier features for 3D world coordinates."""

    def __init__(self, output_dim: int, num_frequencies: int = 8) -> None:
        super().__init__()
        if output_dim <= 0 or num_frequencies <= 0:
            raise ValueError("output_dim and num_frequencies must be positive")
        self.num_frequencies = num_frequencies
        self.project = nn.Linear(3 * 2 * num_frequencies, output_dim)

    def forward(self, points: Tensor) -> Tensor:
        if points.shape[-1] != 3:
            raise ValueError("points must end in xyz coordinates")
        frequencies = (2.0 ** torch.arange(self.num_frequencies, device=points.device, dtype=points.dtype))
        angles = points.unsqueeze(-1) * frequencies
        features = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return self.project(features)


def farthest_point_sample(points: Tensor, valid: Tensor, count: int) -> Tensor:
    """Return indices of spatially spread valid points, padding short inputs by repetition."""
    if points.ndim != 3 or points.shape[-1] != 3 or valid.shape != points.shape[:2]:
        raise ValueError("expected points [B,P,3] and valid [B,P]")
    if count <= 0:
        raise ValueError("count must be positive")
    result = []
    for batch_index in range(points.shape[0]):
        candidates = valid[batch_index].nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError("every batch item must contain a valid 3D point")
        selected = [candidates[0]]
        distances = torch.full((candidates.numel(),), float("inf"), device=points.device, dtype=points.dtype)
        for _ in range(1, min(count, candidates.numel())):
            previous = points[batch_index, selected[-1]]
            distances = torch.minimum(distances, ((points[batch_index, candidates] - previous) ** 2).sum(-1))
            selected.append(candidates[distances.argmax()])
        selected_tensor = torch.stack(selected)
        if selected_tensor.numel() < count:
            selected_tensor = selected_tensor.repeat((count + selected_tensor.numel() - 1) // selected_tensor.numel())[:count]
        result.append(selected_tensor)
    return torch.stack(result)


class SpatialTokenEncoder(nn.Module):
    """Fuse supplied 2D patch features with reconstructed 3D positions and FPS."""

    def __init__(self, feature_dim: int, num_tokens: int, num_frequencies: int = 8) -> None:
        super().__init__()
        self.position = FourierPositionEmbedding(feature_dim, num_frequencies)
        self.num_tokens = num_tokens

    def forward(self, patch_features: Tensor, depth: Tensor, intrinsics: Tensor, extrinsics: Tensor) -> tuple[Tensor, Tensor]:
        """Return sampled `[B, N, D]` features and their `[B, N, 3]` world points."""
        if patch_features.ndim != 5:
            raise ValueError("patch_features must have shape [B, V, H, W, D]")
        if patch_features.shape[:-1] != depth.shape:
            raise ValueError("patch feature grid and depth shape differ")
        points = unproject_depth(depth, intrinsics, extrinsics)
        valid = torch.isfinite(depth) & (depth > 0)
        fused = patch_features + self.position(points)
        batch = patch_features.shape[0]
        flat_points = points.reshape(batch, -1, 3)
        flat_features = fused.reshape(batch, -1, fused.shape[-1])
        indices = farthest_point_sample(flat_points, valid.reshape(batch, -1), self.num_tokens)
        gather_features = indices.unsqueeze(-1).expand(-1, -1, flat_features.shape[-1])
        gather_points = indices.unsqueeze(-1).expand(-1, -1, 3)
        return flat_features.gather(1, gather_features), flat_points.gather(1, gather_points)
