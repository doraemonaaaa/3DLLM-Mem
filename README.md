# 3DLLM-Mem

An independent, runnable implementation of the memory mechanisms described in
the 3DLLM-Mem paper. The original implementation is not public, so this project
implements only mechanisms confirmed by the paper and labels integration choices
explicitly in [Scheme.md](Scheme.md).

## Included

- RGB-D unprojection using intrinsics and camera-to-world poses.
- Learned Fourier 3D position embeddings added to externally supplied 2D patch features.
- Farthest Point Sampling to produce a fixed-size working memory.
- Scene-keyed episodic memory with overwrite-on-revisit semantics.
- Sinusoidal temporal encoding and masked scaled dot-product cross-attention.
- A non-invasive LLaVA-3D runtime adapter: hooks replace RGB-D tokens after
  projection into LLM hidden space without modifying the cloned repository.

## Install and check

```bash
python -m pip install -e '.[dev]'
pytest -q
python scripts/smoke_memory.py
```

`SpatialTemporalMemory` serves a single online trajectory. Call `observe(...)`
for an RGB-D observation and `commit()` when leaving a scene. Supplying an
existing `scene_id` replaces its stored state, reflecting the paper's latest
environment-state requirement.

## LLaVA-3D adapter

Keep LLaVA-3D as an untouched sibling checkout, install it normally, and load
its RGB-D checkpoint with its own `load_pretrained_model` API. The external
adapter supports the original Llama 3D path, whose video tower returns voxelized
tokens and offsets.

```python
from three_dllm_mem import LLaVA3DEpisodicAdapter

adapter = LLaVA3DEpisodicAdapter(hidden_size=model.config.hidden_size, memory_dim=4096)
adapter.to(model.device, dtype=model.dtype)
adapter.install(model)

with adapter.observation(scene_id="scene0356_00", timestep=0, trajectory_id="episode-42"):
    answer_ids = model.generate(
        inputs=input_ids, images=rgbd_views, depths=depths,
        poses=poses, intrinsics=intrinsics, lengths=lengths,
    )

adapter.clear("episode-42")
adapter.uninstall()
```

The first observation is stored after generation; later observations retrieve
before storing. Do not share one adapter across concurrent requests.
