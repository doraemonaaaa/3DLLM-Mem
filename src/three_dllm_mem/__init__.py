"""3DLLM-Mem independent implementation.

The package implements the paper's spatial-token and episodic-memory core.
It deliberately exposes the LLaVA-3D boundary rather than bundling a proxy
for the unavailable upstream visual encoder and language decoder.
"""

from .geometry import SpatialTokenEncoder, farthest_point_sample, unproject_depth
from .llava3d_adapter import LLaVA3DEpisodicAdapter
from .memory import EpisodicMemoryBank, MemoryFusion, MemoryState
from .model import SpatialTemporalMemory

__all__ = [
    "EpisodicMemoryBank",
    "MemoryFusion",
    "MemoryState",
    "LLaVA3DEpisodicAdapter",
    "SpatialTemporalMemory",
    "SpatialTokenEncoder",
    "farthest_point_sample",
    "unproject_depth",
]
