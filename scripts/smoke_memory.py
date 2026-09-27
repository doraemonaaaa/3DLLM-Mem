"""Run a synthetic online trajectory through the complete memory core."""

import torch

from three_dllm_mem import SpatialTemporalMemory


def main() -> None:
    model = SpatialTemporalMemory(feature_dim=16, memory_dim=32, num_tokens=8)
    patches = torch.randn(1, 2, 4, 4, 16)
    depth = torch.ones(1, 2, 4, 4)
    intrinsics = torch.eye(3).repeat(1, 2, 1, 1)
    poses = torch.eye(4).repeat(1, 2, 1, 1)
    tokens, _ = model.observe(patches, depth, intrinsics, poses, scene_id="kitchen", timestep=0)
    model.commit()
    fused, attention = model.observe(patches, depth, intrinsics, poses, scene_id="hall", timestep=1)
    print(f"working tokens: {tokens.shape}; fused tokens: {fused.shape}; attention: {attention.shape}")


if __name__ == "__main__":
    main()
