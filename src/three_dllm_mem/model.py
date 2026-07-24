"""Online memory controller joining spatial tokens and episodic fusion."""

from __future__ import annotations

from torch import Tensor, nn

from .geometry import SpatialTokenEncoder
from .memory import EpisodicMemoryBank, MemoryFusion


class SpatialTemporalMemory(nn.Module):
    """L0 online controller for a single embodied trajectory."""

    def __init__(self, feature_dim: int, memory_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.encoder = SpatialTokenEncoder(feature_dim, num_tokens)
        self.fusion = MemoryFusion(feature_dim, memory_dim)
        self.bank = EpisodicMemoryBank()
        self._working_memory: Tensor | None = None
        self._scene_id: str | None = None
        self._timestep: int | None = None

    def observe(self, patch_features: Tensor, depth: Tensor, intrinsics: Tensor, extrinsics: Tensor, *, scene_id: str, timestep: int) -> tuple[Tensor, Tensor]:
        """Encode the current scene and fuse it with committed past scenes."""
        working, _ = self.encoder(patch_features, depth, intrinsics, extrinsics)
        if working.shape[0] != 1:
            raise ValueError("L0 controller supports one trajectory per instance")
        self._working_memory, self._scene_id, self._timestep = working, scene_id, timestep
        if not self.bank:
            return working, working.new_zeros((1, working.shape[1], 0))
        return self.fusion(working, self.bank.as_state(device=working.device))

    def commit(self) -> None:
        """Persist the most recently observed scene, replacing it on revisit."""
        if self._working_memory is None or self._scene_id is None or self._timestep is None:
            raise RuntimeError("observe a scene before committing it")
        self.bank.write(self._scene_id, self._working_memory[0], self._timestep)
