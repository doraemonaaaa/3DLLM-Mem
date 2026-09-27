"""Scene-keyed episodic memory and temporal cross-attention fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MemoryState:
    """Padded bank representation consumed by `MemoryFusion`."""

    tokens: Tensor  # [B, T, N, D]
    mask: Tensor  # [B, T, N], True means a real token
    timesteps: Tensor  # [B, T]


class EpisodicMemoryBank:
    """Single-trajectory bank that replaces a scene entry on revisit.

    `scene_id` is the deliberately explicit L0 oracle association method. In a
    simulator it should be a room ID; in a real system it needs localization.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[Tensor, int]] = {}

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def write(self, scene_id: str, tokens: Tensor, timestep: int) -> None:
        if not scene_id:
            raise ValueError("scene_id must be non-empty")
        if tokens.ndim != 2:
            raise ValueError("tokens must be [N, D]")
        self._entries[scene_id] = (tokens.detach().clone(), int(timestep))

    def as_state(self, device: torch.device | None = None) -> MemoryState:
        if not self._entries:
            raise RuntimeError("episodic memory bank is empty")
        entries = list(self._entries.values())
        feature_dim = entries[0][0].shape[-1]
        if any(tokens.ndim != 2 or tokens.shape[-1] != feature_dim for tokens, _ in entries):
            raise ValueError("all memory entries must be [N, D] with the same D")
        target = device or entries[0][0].device
        max_tokens = max(tokens.shape[0] for tokens, _ in entries)
        dtype = entries[0][0].dtype
        tokens = torch.zeros((1, len(entries), max_tokens, feature_dim), device=target, dtype=dtype)
        mask = torch.zeros((1, len(entries), max_tokens), device=target, dtype=torch.bool)
        timesteps = torch.empty((1, len(entries)), device=target, dtype=torch.long)
        for i, (entry, timestep) in enumerate(entries):
            tokens[0, i, : entry.shape[0]] = entry.to(target)
            mask[0, i, : entry.shape[0]] = True
            timesteps[0, i] = timestep
        return MemoryState(tokens=tokens, mask=mask, timesteps=timesteps)


class MemoryFusion(nn.Module):
    """Paper-specified Q/K/V episodic cross-attention with temporal encoding."""

    def __init__(self, input_dim: int, memory_dim: int, temporal_scale: float = 10_000.0) -> None:
        super().__init__()
        if input_dim < 1 or memory_dim < 2 or memory_dim % 2:
            raise ValueError("input_dim must be positive and memory_dim positive/even")
        self.memory_dim = memory_dim
        self.temporal_scale = temporal_scale
        self.memory_projector = nn.Sequential(nn.Linear(input_dim, memory_dim), nn.GELU(), nn.Linear(memory_dim, memory_dim))
        self.query = nn.Linear(input_dim, memory_dim, bias=False)
        self.key = nn.Linear(memory_dim, memory_dim, bias=False)
        self.value = nn.Linear(memory_dim, memory_dim, bias=False)

    def temporal_embedding(self, timesteps: Tensor) -> Tensor:
        half = self.memory_dim // 2
        denominator = max(half - 1, 1)
        exponents = -torch.arange(half, device=timesteps.device, dtype=torch.float32)
        frequencies = torch.exp(exponents * (torch.log(torch.tensor(self.temporal_scale, device=timesteps.device)) / denominator))
        angles = timesteps.float().unsqueeze(-1) * frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def forward(self, working_memory: Tensor, episodic: MemoryState) -> tuple[Tensor, Tensor]:
        if working_memory.ndim != 3 or episodic.tokens.ndim != 4:
            raise ValueError("working_memory must be [B,N,D], episodic tokens [B,T,N,D]")
        batch, entries, tokens, _ = episodic.tokens.shape
        if episodic.mask.shape != (batch, entries, tokens) or episodic.timesteps.shape != (batch, entries):
            raise ValueError("invalid episodic mask or timesteps")
        if working_memory.shape[0] != batch:
            raise ValueError("working and episodic batch dimensions differ")
        valid = episodic.mask.flatten(1, 2)
        if not valid.any(-1).all():
            raise ValueError("every batch item needs a valid memory token")

        query = self.query(working_memory)
        memory = self.memory_projector(episodic.tokens)
        temporal = self.temporal_embedding(episodic.timesteps).to(memory.dtype).unsqueeze(2)
        keys = self.key(memory + temporal).flatten(1, 2)
        values = self.value(memory + temporal).flatten(1, 2)
        scores = query @ keys.transpose(-1, -2) / (self.memory_dim**0.5)
        scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        attention = scores.softmax(-1)
        fused = attention @ values
        # Feature-axis concat preserves one fused context per current spatial token.
        return torch.cat((fused, query), dim=-1), attention
