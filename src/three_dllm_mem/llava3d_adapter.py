"""Non-invasive runtime adapter for an unmodified LLaVA-3D checkout.

The adapter uses module forward hooks only. It does not patch or write files in
the LLaVA-3D repository: the video-tower hook captures voxel offsets and the
projector hook replaces its projected RGB-D tokens with memory-enhanced tokens.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor, nn

from .memory import EpisodicMemoryBank, MemoryFusion


@dataclass(frozen=True)
class _Observation:
    scene_id: str
    timestep: int
    trajectory_id: str
    commit: bool


class LLaVA3DEpisodicAdapter(nn.Module):
    """Inject 3DLLM-Mem into LLaVA-3D at runtime without source modification.

    A context applies to exactly one RGB-D video item. This mirrors embodied
    inference, where each model invocation observes one environment at a time.
    Do not share an adapter between concurrent requests; create one per worker.
    """

    def __init__(self, hidden_size: int, memory_dim: int) -> None:
        super().__init__()
        self.fusion = MemoryFusion(hidden_size, memory_dim)
        self.to_hidden = nn.Linear(memory_dim * 2, hidden_size, bias=False)
        self._banks: dict[str, EpisodicMemoryBank] = {}
        self._active: _Observation | None = None
        self._offsets: Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def install(self, model: nn.Module) -> None:
        """Install hooks on an unmodified LLaVA-3D Llama model exactly once."""
        if self._handles:
            raise RuntimeError("adapter is already installed")
        core = model.get_model()
        video_tower = core.get_video_tower()
        if video_tower is None or not hasattr(core, "mm_projector"):
            raise ValueError("model must be an initialized LLaVA-3D RGB-D model")
        self._handles = [
            video_tower.register_forward_hook(self._capture_offsets),
            core.mm_projector.register_forward_hook(self._fuse_projected_tokens),
        ]

    def uninstall(self) -> None:
        """Remove hooks before disposing the wrapped LLaVA-3D model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._offsets = None

    def clear(self, trajectory_id: str | None = None) -> None:
        """Clear episode-local state; call at the end of every environment episode."""
        if trajectory_id is None:
            self._banks.clear()
        else:
            self._banks.pop(trajectory_id, None)

    @contextmanager
    def observation(
        self, *, scene_id: str, timestep: int, trajectory_id: str = "default", commit: bool = True
    ) -> Iterator[None]:
        """Activate memory for one standard LLaVA-3D `forward` or `generate` call."""
        if self._active is not None:
            raise RuntimeError("nested observations are not supported")
        self._active = _Observation(scene_id, int(timestep), trajectory_id, commit)
        self._offsets = None
        try:
            yield
        finally:
            self._active = None
            self._offsets = None

    def _capture_offsets(self, _module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        if self._active is None:
            return
        if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[1], Tensor):
            raise RuntimeError("unexpected LLaVA-3D video tower output; expected (features, offsets)")
        self._offsets = output[1]

    def _fuse_projected_tokens(self, _module: nn.Module, _inputs: tuple[object, ...], output: Tensor) -> Tensor:
        active = self._active
        if active is None:
            return output
        if self._offsets is None:
            raise RuntimeError("LLaVA-3D projector ran without RGB-D video tower offsets")
        offsets = self._offsets.reshape(-1)
        if offsets.numel() != 1:
            raise ValueError("one adapter observation must contain exactly one RGB-D video")
        end = int(offsets.item())
        if output.ndim != 2 or end != output.shape[0]:
            raise RuntimeError("projected LLaVA-3D token shape does not match video offsets")

        bank = self._banks.setdefault(active.trajectory_id, EpisodicMemoryBank())
        working = output.unsqueeze(0)
        if bank:
            fused, _ = self.fusion(working, bank.as_state(device=output.device))
            enhanced = self.to_hidden(fused).squeeze(0)
        else:
            enhanced = output
        if active.commit:
            # Keep raw projected tokens; retrieval must not recursively store fused output.
            bank.write(active.scene_id, output, active.timestep)
        return enhanced
