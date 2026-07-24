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

## 已确认训练参数

- 基座：LLaVA-3D。
- 上下文长度：8192。
- 微调：memory module 和 LLM decoder。
- loss：标准自回归 language modeling loss。
- optimizer：Adam；learning rate `2e-5`；weight decay 0。
- schedule：前 3% steps 从 `1e-8` 线性 warmup，之后 cosine decay。
- 论文完整训练：8 TPU v5p cores，global batch 256，1000 steps，约 1 天。
