"""Online controller and a thin adapter boundary for an LLaVA-3D decoder."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from .geometry import SpatialTokenEncoder
from .memory import EpisodicMemoryBank, MemoryFusion


class SpatialTemporalMemory(nn.Module):
    """One-trajectory controller implementing observe -> fuse -> commit."""

    def __init__(self, feature_dim: int, memory_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.encoder = SpatialTokenEncoder(feature_dim, num_tokens)
        self.fusion = MemoryFusion(feature_dim, memory_dim)
        self.bank = EpisodicMemoryBank()
        self._working_memory: Tensor | None = None
        self._scene_id: str | None = None
        self._timestep: int | None = None

    def observe(self, patch_features: Tensor, depth: Tensor, intrinsics: Tensor, camera_to_world: Tensor, *, scene_id: str, timestep: int) -> tuple[Tensor, Tensor]:
        """Encode an observation and retrieve past committed scene states."""
        working, _ = self.encoder(patch_features, depth, intrinsics, camera_to_world)
        if working.shape[0] != 1:
            raise ValueError("one SpatialTemporalMemory instance handles one trajectory")
        self._working_memory, self._scene_id, self._timestep = working, scene_id, timestep
        if not self.bank:
            return working, working.new_zeros((1, working.shape[1], 0))
        return self.fusion(working, self.bank.as_state(working.device))

    def commit(self) -> None:
        """Write the current state, replacing the entry when the scene is revisited."""
        if self._working_memory is None or self._scene_id is None or self._timestep is None:
            raise RuntimeError("observe before commit")
        self.bank.write(self._scene_id, self._working_memory[0], self._timestep)


def decode_with_llava3d(decoder: Callable[..., object], fused_tokens: Tensor, instruction: object, **kwargs: object) -> object:
    """Call a supplied LLaVA-3D-compatible decoder without prescribing its API.

    The unavailable LLaVA-3D implementation determines projection, prompt
    packing and generation arguments. This boundary makes that dependency clear.
    """
    return decoder(visual_tokens=fused_tokens, instruction=instruction, **kwargs)
