# 3DLLM-Mem 方法实现规格

本文件只记录论文可直接确认的机制；未明确的实现选择列在 `unknowns.md`。

## 输入到 3D token

对每个时刻的 RGB-D 多视角观测，论文以 **LLaVA-3D** 为基座：CLIP 将图像编码为 2D patch features，并投影到 LLM hidden space；由深度、相机内参和外参得到对应的 3D 世界坐标并编码为 LLaVA-3D 的 3D position embeddings；两者相加后，经 Farthest Point Sampling（FPS）保留固定数目的 3D tokens。

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

论文将 fused episodic memory 与 working memory 一起作为 memory-enhanced representation 交给 **LLaVA-3D 的 LLM decoder**，并与语言指令共同用于自回归生成推理和高层动作。当前 L0 只实现该接口前的表示；它必须被替换/接入到 LLaVA-3D 的实际视觉 token 管线后，才属于论文复现。


# 论文未明确的实现细节

下列内容在论文正文、附录 D 和 Figure 3 中没有给出足以无歧义复现的规范。官方仓库目前没有可用实现；因此这些默认值只代表本仓库的独立选择，而不是论文事实。

| 项目 | 论文可确认内容 | 当前计划 |
| --- | --- | --- |
| `N`，每时刻 token 数 | 通过 FPS 下采样为固定数目 | 暂不固定，配置化 |
| `M/C`，memory 维度 | 公式中定义为 memory feature space | 暂不固定，配置化 |
| MLP/QKV 结构 | observation 先经 MLP；存在 Q/K/V features | L0 使用独立无 bias Linear Q/K/V |
| attention heads | 未说明 | L0 为单头；后续提供 multi-head 版本 |
| temporal encoding | sinusoidal positional embedding | L0 使用标准 sin/cos |
| concat 轴 | 写作 `Concat[f_fuse^Q; f_t^Q]` | L0 沿 feature 轴；作为独立实现选择 |
| scene entry 匹配 | 重访已存在环境时更新 | L0 使用 `scene_id` oracle key |
| 旧状态策略 | bank 反映最新状态 | L0 覆盖旧 entry；可增加版本历史消融 |
| instruction 参与 retrieval | 文中称 task-relevant，但 Q 来自 working memory | L0 不让 instruction 进入 Q；另做扩展实验 |
| history 上限/淘汰 | episodic memory 被称为 expandable | L0 不淘汰；长序列增加可选策略 |
| 视觉 encoder 是否训练 | 训练 memory module 和 LLM decoder | 默认冻结视觉 encoder；后续通过消融决定 |

# 已确认训练参数

- 基座：LLaVA-3D。
- 上下文长度：8192。
- 微调：memory module 和 LLM decoder。
- loss：标准自回归 language modeling loss。
- optimizer：Adam；learning rate `2e-5`；weight decay 0。
- schedule：前 3% steps 从 `1e-8` 线性 warmup，之后 cosine decay。
- 论文完整训练：8 TPU v5p cores，global batch 256，1000 steps，约 1 天。
