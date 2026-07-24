"""Run the paper's RGB-D -> 3D token -> episodic fusion path on synthetic data."""

import torch

from three_dllm_mem.model import SpatialTemporalMemory


def main() -> None:
    torch.manual_seed(7)
    feature_dim, memory_dim, token_count = 32, 16, 8
    model = SpatialTemporalMemory(feature_dim, memory_dim, token_count)
    intrinsics = torch.eye(3).view(1, 1, 3, 3)
    extrinsics = torch.eye(4).view(1, 1, 4, 4)

    for timestep, scene_id in enumerate(("kitchen", "bedroom"), start=1):
        features = torch.randn(1, 1, 4, 4, feature_dim)
        depth = torch.rand(1, 1, 4, 4) + 0.5
        model.observe(features, depth, intrinsics, extrinsics, scene_id=scene_id, timestep=timestep)
        model.commit()

    current_features = torch.randn(1, 1, 4, 4, feature_dim)
    current_depth = torch.rand(1, 1, 4, 4) + 0.5
    enhanced, attention = model.observe(
        current_features, current_depth, intrinsics, extrinsics, scene_id="living-room", timestep=3
    )
    print(f"memory enhanced: {tuple(enhanced.shape)}")
    print(f"attention: {tuple(attention.shape)}")


if __name__ == "__main__":
    main()
