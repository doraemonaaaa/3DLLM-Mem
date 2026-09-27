# 3DLLM-Mem TODO

## Current status

- [x] RGB-D unprojection from intrinsics and camera-to-world poses.
- [x] 3D positional feature addition and farthest point sampling.
- [x] Scene-keyed episodic bank with overwrite-on-revisit behavior.
- [x] Temporal encoding, masked Q/K/V cross-attention, and feature-axis fusion.
- [x] Unit tests and a synthetic online-trajectory smoke example.
- [ ] Install the declared PyTorch and pytest dependencies, then run the tests.

## LLaVA-3D integration

- [x] Implement an external forward-hook adapter after the LLaVA-3D video tower
  and `mm_projector`, without modifying the upstream checkout.
- [x] Add a trainable output adapter from fused `[B, N, 2*M]` to LLaVA hidden
  size, plus trajectory-scoped scene overwrite and explicit clearing.
- [ ] Pin a tested LLaVA-3D revision, install its runtime and checkpoint, then
  run an end-to-end generation test through `LLaVA3DEpisodicAdapter`.
- [ ] Preserve the paper's current working tokens alongside retrieved memory;
  compare feature-axis fusion with token-axis concatenation as an ablation.
- [x] Scope memory-bank state by trajectory ID. Deploy one adapter per worker;
  batching multiple RGB-D videos remains an explicit future extension.
- [ ] Define the non-oracle scene association mechanism using simulator room IDs
  first, then localization or pose/scene matching for real deployments.
- [ ] Add an inference wrapper that performs observe -> retrieve -> generate ->
  commit, with an explicit environment-switch policy.

## Training and evaluation

- [ ] Implement a dataset/collator that returns sequential RGB-D observations,
  camera data, instruction/answer labels, trajectory ID, and scene ID.
- [ ] Train the memory module and LLaVA-3D decoder using causal-LM loss; freeze
  the visual encoder initially as specified in `Scheme.md`.
- [ ] Configure Adam with LR `2e-5`, zero weight decay, 3% warmup from `1e-8`,
  cosine decay, and context length 8192.
- [ ] Add resume-safe checkpointing for model, optimizer, scheduler, and memory
  configuration; episodic state should not leak across examples.
- [ ] Evaluate no-memory, fixed-memory, overwrite-on-revisit, and temporal
  encoding ablations on the target embodied navigation/task benchmark.

## Engineering

- [ ] Add CPU and CUDA integration tests using a tiny local LLaVA-3D fixture.
- [ ] Add shape, dtype, device, and all-masked-memory checks at the adapter
  boundary; test multiple trajectory batch elements.
- [ ] Decide an eviction/compression strategy before scaling beyond the LLM's
  8192-token context limit.
