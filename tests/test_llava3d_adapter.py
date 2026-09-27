import torch
from torch import nn

from three_dllm_mem.llava3d_adapter import LLaVA3DEpisodicAdapter


class _VideoTower(nn.Module):
    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tokens, torch.tensor([tokens.shape[0]], device=tokens.device)


class _Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_tower = _VideoTower()
        self.mm_projector = nn.Identity()

    def get_video_tower(self) -> nn.Module:
        return self.video_tower


class _LLaVAStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = _Core()

    def get_model(self) -> _Core:
        return self.core

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.core.mm_projector(self.core.video_tower(tokens)[0])


def test_adapter_is_inactive_without_observation_context() -> None:
    model = _LLaVAStub()
    adapter = LLaVA3DEpisodicAdapter(hidden_size=4, memory_dim=6)
    adapter.install(model)
    tokens = torch.randn(2, 4)
    assert model(tokens).data_ptr() == tokens.data_ptr()


def test_adapter_retrieves_and_replaces_scene_state() -> None:
    model = _LLaVAStub()
    adapter = LLaVA3DEpisodicAdapter(hidden_size=4, memory_dim=6)
    adapter.install(model)
    with adapter.observation(scene_id="room", timestep=0):
        first = model(torch.randn(2, 4))
    assert len(adapter._banks["default"]) == 1
    with adapter.observation(scene_id="hall", timestep=1):
        second = model(torch.randn(2, 4))
    assert first.shape == second.shape == (2, 4)
    adapter.uninstall()
