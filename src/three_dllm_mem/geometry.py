"""RGB-D geometry and 3D token construction."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def unproject_depth(depth: Tensor, intrinsics: Tensor, camera_to_world: Tensor) -> Tensor:
    """Map `[B,V,H,W]` metric depth to world points `[B,V,H,W,3]`."""
    if depth.ndim != 4:
        raise ValueError("depth must be [B, V, H, W]")
    batch, views, height, width = depth.shape
    if intrinsics.shape != (batch, views, 3, 3):
        raise ValueError("intrinsics must be [B, V, 3, 3]")
    if camera_to_world.shape != (batch, views, 4, 4):
        raise ValueError("camera_to_world must be [B, V, 4, 4]")

    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1).reshape(1, 1, -1, 3)
    pixels = pixels.expand(batch, views, -1, -1)
    rays = pixels @ torch.linalg.inv(intrinsics).transpose(-1, -2)
    camera_points = rays * depth.reshape(batch, views, -1, 1)
    homogeneous = torch.cat((camera_points, torch.ones_like(camera_points[..., :1])), dim=-1)
    world = homogeneous @ camera_to_world.transpose(-1, -2)
    return world[..., :3].reshape(batch, views, height, width, 3)


def farthest_point_sample(points: Tensor, valid: Tensor, count: int) -> Tensor:
    """Select spatially separated indices, repeating selections if `count` is larger."""
    if points.ndim != 3 or points.shape[-1] != 3 or valid.shape != points.shape[:2]:
        raise ValueError("points must be [B,P,3] and valid must be [B,P]")
    if count < 1:
        raise ValueError("count must be positive")
    result: list[Tensor] = []
    for b in range(points.shape[0]):
        candidates = valid[b].nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError("each item requires at least one positive finite depth")
        selected = [candidates[0]]
        nearest = torch.full((candidates.numel(),), torch.inf, device=points.device, dtype=points.dtype)
        for _ in range(1, min(count, candidates.numel())):
            distance = (points[b, candidates] - points[b, selected[-1]]).square().sum(-1)
            nearest = torch.minimum(nearest, distance)
            selected.append(candidates[nearest.argmax()])
        indices = torch.stack(selected)
        result.append(indices.repeat((count + indices.numel() - 1) // indices.numel())[:count])
    return torch.stack(result)


class FourierPositionEmbedding(nn.Module):
    """Fixed Fourier xyz encoding projected to the visual feature dimension."""

    def __init__(self, output_dim: int, frequencies: int = 8) -> None:
        super().__init__()
        if output_dim < 1 or frequencies < 1:
            raise ValueError("output_dim and frequencies must be positive")
        self.frequencies = frequencies
        self.projection = nn.Linear(6 * frequencies, output_dim)

    def forward(self, points: Tensor) -> Tensor:
        if points.shape[-1] != 3:
            raise ValueError("points must end with xyz")
        bands = 2.0 ** torch.arange(self.frequencies, device=points.device, dtype=points.dtype)
        angles = points.unsqueeze(-1) * bands
        features = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return self.projection(features)


class SpatialTokenEncoder(nn.Module):
    """Adds 3D positional features to LLaVA-3D-compatible patch features then FPS samples."""

    def __init__(self, feature_dim: int, num_tokens: int, frequencies: int = 8) -> None:
        super().__init__()
        self.position = FourierPositionEmbedding(feature_dim, frequencies)
        self.num_tokens = num_tokens

    def forward(self, patch_features: Tensor, depth: Tensor, intrinsics: Tensor, camera_to_world: Tensor) -> tuple[Tensor, Tensor]:
        if patch_features.ndim != 5 or patch_features.shape[:-1] != depth.shape:
            raise ValueError("patch_features must be [B,V,H,W,D] aligned with depth")
        points = unproject_depth(depth, intrinsics, camera_to_world)
        fused = patch_features + self.position(points)
        batch = depth.shape[0]
        flat_points = points.reshape(batch, -1, 3)
        flat_features = fused.reshape(batch, -1, fused.shape[-1])
        valid = (depth.isfinite() & (depth > 0)).reshape(batch, -1)
        indices = farthest_point_sample(flat_points, valid, self.num_tokens)
        return (
            flat_features.gather(1, indices.unsqueeze(-1).expand(-1, -1, flat_features.shape[-1])),
            flat_points.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3)),
        )
