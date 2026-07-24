# 3DLLM-Mem 方法实现规格

本文件只记录论文可直接确认的机制；未明确的实现选择列在 `unknowns.md`。

## 输入到 3D token

对每个时刻的 RGB-D 多视角观测，LLaVA-3D 的流程为：CLIP 将图像编码为 2D patch features；由深度、相机内参和外参得到对应的 3D 世界坐标并编码为 3D position embeddings；两者相加后，经 Farthest Point Sampling（FPS）保留固定数目的 3D tokens。

```text
RGB-D + camera poses
  -> CLIP patch features Xp [V, d, w, h]
  + 3D position embeddings P [V, d, w, h]
  -> X3D [V, d, w, h]
  -> FPS
  -> X_t [N, d]
```

`X_t` 是时刻 `t` 的 Working Memory。历史 observation 组成 Episodic Memory，概念形状为 `[T, N, d]`。

## Episodic Memory

每个历史 observation 经 MLP 投影到 memory feature space，并叠加其时间步的 sinusoidal positional embedding。对当前时刻之前的历史 token，memory bank 提供 key/value：

```text
Q_t: [B, N_current, M]
K,V: [B, T * N_history, M]
mask: [B, T * N_history]
```

本仓库 L0 代码额外为每条 entry 保存 `scene_id`。这是一种明确、可测试的 oracle 场景关联方式；真实系统将改由 simulator room ID 或 pose/scene matching 产生该 key。

## Memory Fusion

论文的融合是标准 scaled dot-product cross-attention：

```text
A = softmax(Q_t K^T / sqrt(M))
F = A V
f_memory = concat(F, Q_t)
```

`concat` 的确切轴论文没有文字说明。当前 L0 默认沿 feature 轴拼接，因此输出为 `[B, N_current, 2M]`；这是让每个当前 token 对齐获得一个历史上下文向量的最直接实现。接入 LLaVA-3D 时需要以官方实现或消融确认是否应改为 token 轴拼接。

## 在线更新

```text
当前 observation -> Working Memory
离开/切换环境     -> 写入 Episodic Memory Bank
重访且环境变化     -> 更新相应 memory entry
```

论文说 memory bank 反映已探索环境的最新状态，但未规定场景匹配算法、是否保存旧状态版本，或对局部可见变化的融合方式。

## LLM 接口

论文将 fused episodic memory 与 working memory 一起作为 memory-enhanced representation 交给 LLM，并与语言指令共同用于自回归生成推理和高层动作。L0 仅实现该接口前的表示，不包含 LLaVA-3D 或动作执行器。
