"""Token-level episodic memory and fusion used by 3DLLM-Mem.

This module intentionally models only the paper's memory core. RGB-D encoding,
scene association, and the LLM interface are separate concerns.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MemoryState:
    """Padded episodic tokens, their validity mask, and per-entry timesteps."""

    tokens: Tensor  # [batch, entries, tokens_per_entry, feature_dim]
    mask: Tensor  # [batch, entries, tokens_per_entry], True for valid tokens
    timesteps: Tensor  # [batch, entries]


class EpisodicMemoryBank:
    """Small, explicit memory bank with append and scene-keyed update semantics."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[Tensor, int]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def write(self, scene_id: str, tokens: Tensor, timestep: int) -> None:
        """Append a new scene or replace its latest state on revisit.

        `scene_id` is an oracle association key for L0. A later simulator adapter
        will derive it from room IDs or pose-based scene association.
        """
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [num_tokens, feature_dim]")
        if not scene_id:
            raise ValueError("scene_id must be non-empty")
        self._entries[scene_id] = (tokens.detach().clone(), int(timestep))

    def as_state(self, *, device: torch.device | None = None) -> MemoryState:
        """Return a one-item padded batch suitable for memory fusion."""
        if not self._entries:
            raise RuntimeError("cannot fuse an empty episodic memory bank")

        values = list(self._entries.values())
        feature_dim = values[0][0].shape[-1]
        if any(item[0].shape[-1] != feature_dim for item in values):
            raise ValueError("all memory entries must have the same feature dimension")
        entry_count = len(values)
        max_tokens = max(item[0].shape[0] for item in values)
        target_device = device or values[0][0].device
        dtype = values[0][0].dtype
        tokens = torch.zeros((1, entry_count, max_tokens, feature_dim), device=target_device, dtype=dtype)
        mask = torch.zeros((1, entry_count, max_tokens), device=target_device, dtype=torch.bool)
        timesteps = torch.empty((1, entry_count), device=target_device, dtype=torch.long)
        for index, (entry_tokens, timestep) in enumerate(values):
            count = entry_tokens.shape[0]
            tokens[0, index, :count] = entry_tokens.to(target_device)
            mask[0, index, :count] = True
            timesteps[0, index] = timestep
        return MemoryState(tokens=tokens, mask=mask, timesteps=timesteps)


class MemoryFusion(nn.Module):
    """Working-memory-query cross attention over temporally encoded 3D memories."""

    def __init__(self, input_dim: int, memory_dim: int, *, temporal_scale: float = 10_000.0) -> None:
        super().__init__()
        if memory_dim <= 0 or memory_dim % 2:
            raise ValueError("memory_dim must be a positive even integer")
        self.query = nn.Linear(input_dim, memory_dim, bias=False)
        # The paper first maps stored observations into a memory-specific space.
        self.memory_projector = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.key = nn.Linear(memory_dim, memory_dim, bias=False)
        self.value = nn.Linear(memory_dim, memory_dim, bias=False)
        self.memory_dim = memory_dim
        self.temporal_scale = temporal_scale

    def temporal_embedding(self, timesteps: Tensor) -> Tensor:
        """Standard sinusoidal embedding for timesteps shaped [...]."""
        half_dim = self.memory_dim // 2
        frequencies = torch.exp(
            -torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            * (torch.log(torch.tensor(self.temporal_scale, device=timesteps.device)) / max(half_dim - 1, 1))
        )
        angles = timesteps.to(torch.float32).unsqueeze(-1) * frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def forward(self, working_memory: Tensor, episodic: MemoryState) -> tuple[Tensor, Tensor]:
        """Fuse episodic values into current tokens.

        Args:
            working_memory: Current 3D tokens `[batch, current_tokens, input_dim]`.
            episodic: Historical entries with shape `[batch, entries, tokens, input_dim]`.

        Returns:
            `memory_enhanced`: `[batch, current_tokens, 2 * memory_dim]`.
            `attention`: `[batch, current_tokens, entries * tokens]` for inspection.
        """
        if working_memory.ndim != 3 or episodic.tokens.ndim != 4:
            raise ValueError("working memory must be rank 3 and episodic tokens rank 4")
        batch, entries, tokens_per_entry, _ = episodic.tokens.shape
        if working_memory.shape[0] != batch:
            raise ValueError("working memory and episodic batch sizes differ")
        if episodic.mask.shape != (batch, entries, tokens_per_entry):
            raise ValueError("episodic mask shape is invalid")

        query = self.query(working_memory)
        memory_features = self.memory_projector(episodic.tokens)
        keys = self.key(memory_features)
        values = self.value(memory_features)
        time = self.temporal_embedding(episodic.timesteps).to(keys.dtype).unsqueeze(2)
        keys = keys + time
        values = values + time
        keys = keys.flatten(1, 2)
        values = values.flatten(1, 2)
        valid = episodic.mask.flatten(1, 2)
        if not valid.any(dim=1).all():
            raise ValueError("each batch item needs at least one valid episodic token")

        scores = torch.matmul(query, keys.transpose(-1, -2)) / (self.memory_dim**0.5)
        scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        fused = torch.matmul(attention, values)
        return torch.cat((fused, query), dim=-1), attention
