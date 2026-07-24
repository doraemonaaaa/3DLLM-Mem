# 3DLLM-Mem Reproduction

This is an independent reimplementation from the paper. The official repository currently contains no usable model implementation, so every non-explicit design choice is documented separately rather than attributed to the authors.

## Current scope

- `src/three_dllm_mem/geometry.py`: RGB-D back-projection, learnable 3D Fourier position encoding, and FPS token sampling.
- `src/three_dllm_mem/memory.py`: memory update, sinusoidal temporal encoding, masked Q/K/V memory fusion.
- `src/three_dllm_mem/model.py`: online single-trajectory controller for encode, commit, and fuse.
- `tests/`: synthetic tests for geometry, update, padding mask, shapes, and temporal encoding.
- `TODOList.md`: staged roadmap from L0 module tests to benchmark-scale reproduction.

## Install and test

Use a Python 3.10-3.14 environment with a suitable PyTorch build, then run:

```bash
python -m pip install -e '.[dev]'
pytest -q
python scripts/smoke_memory.py
```

The current encoder receives patch features from an external vision encoder. CLIP/LLaVA-3D compatibility, Habitat data pipeline, and LLM integration are the next stages.
