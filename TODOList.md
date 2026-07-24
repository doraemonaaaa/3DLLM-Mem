# 3DLLM-Mem 复现 TODO List

目标：在本目录中复现论文 **3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D Large Language Model**。优先完成可运行的模型与最小实验，再逐步复现 3DMem-Bench 数据、完整训练和论文指标。

论文：[`3DLLM-MemLong-TermSpatial-TemporalMemory.pdf`](./3DLLM-MemLong-TermSpatial-TemporalMemory.pdf)

## 0. 复现范围与验收层级

- [ ] **L0：模块级复现**：Memory Bank、时间编码、Memory Fusion、在线更新已实现；待安装 PyTorch 后运行单元测试。
- [ ] **L1：最小端到端复现**：在少量 HM3D 场景和短轨迹上完成 RGB-D → 3D token → memory fusion → LLM 动作文本的训练和推理。
- [ ] **L2：主要结果复现**：在 3DMem-Bench 或等价重建数据上复现 embodied task、EQA、captioning 三类评测，趋势与论文一致。
- [ ] **L3：论文数值对齐**：尽量对齐论文 Table 2/3；记录无法严格对齐的私有数据、模型权重、模拟器版本和算力差异。

建议先完成 L0 和 L1，不要一开始就投入完整数据生成和大规模 TPU/GPU 训练。

## 1. 项目初始化与资源核实

- [ ] 建立项目目录结构：

  ```text
  3DLLM-Mem/
  ├── README.md
  ├── TODOList.md
  ├── configs/
  ├── docs/
  ├── scripts/
  ├── src/
  │   ├── data/
  │   ├── models/
  │   ├── memory/
  │   ├── simulator/
  │   └── evaluation/
  ├── tests/
  ├── third_party/
  ├── data/          # 不提交大文件
  ├── checkpoints/   # 不提交权重
  └── outputs/       # 日志、预测、指标
  ```

- [ ] 核实论文项目页、作者仓库、3DMem-Bench 数据和预训练权重是否公开。
- [ ] 记录所有资源的 URL、commit hash、发布日期、许可证和下载日期到 `docs/resources.md`。
- [x] 官方仓库状态已核实：仅包含占位 README，无模型实现；本项目采用论文驱动的独立复现。
- [ ] 获取并固定 LLaVA-3D 基线代码及预训练权重版本。
- [ ] 核实 LLaVA-3D 所依赖的 CLIP、LLM、tokenizer、视觉投影层和 3D position embedding 实现。
- [ ] 核实 HM3D、HM3D Semantics、Habitat-Sim、Habitat-Lab 和 Objaverse 的许可证与访问条件。
- [ ] 写 `.gitignore`，排除数据、模型权重、缓存、渲染结果和实验输出。
- [ ] 创建环境文件 `environment.yml` 或 `pyproject.toml`，固定 Python、PyTorch、CUDA、Transformers、Habitat 版本。
- [ ] 记录机器信息和可用算力：GPU 型号/数量、显存、CPU、内存、磁盘空间。

**验收**：新环境能安装；可打印 PyTorch/CUDA/Habitat/LLaVA-3D 版本；资源来源可追踪。

## 2. 论文规格整理

- [x] 将论文方法整理为 `docs/method_spec.md`，明确以下张量：
  - 多视角 patch：`Xp [V, d, w, h]`
  - 3D patch：`X3D [V, d, w, h]`
  - FPS 后 token：`X_3d_feat [N, d]`
  - Working Memory：`X_t [N, d]`
  - Episodic Memory：`X_hist [T, N, d]`
  - Query：`Q [N, M]`
  - Key/Value：`K,V [T*N, M]`
  - 融合输出：`f_fuse [N, M]`
  - 最终表示：`f_memory [2N, M]`，或记录接入 LLM 前的实际投影维度。
- [ ] 整理论文明确给出的训练配置：
  - context length：8192
  - batch size：256
  - training steps：1000
  - optimizer：Adam
  - learning rate：`2e-5`
  - weight decay：0
  - warmup：前 3% steps
  - scheduler：cosine decay
  - 训练模块：memory module + LLM decoder
  - loss：标准自回归 language modeling loss
- [x] 建立 `docs/unknowns.md`，记录论文未说明或有歧义的参数：
  - 每时刻 memory token 数 `N`
  - memory hidden size `M/C`
  - Memory MLP 层数、激活函数、归一化与 dropout
  - Q/K/V 是否使用独立投影
  - 多头还是单头 attention
  - 历史时间步上限和 memory eviction 策略
  - “同一环境”的匹配与覆盖规则
  - 工作记忆进入 LLM 的 token 排列方式
  - instruction 是否显式参与 memory query
  - action history/feedback 的编码和拼接方式
- [x] 对所有未知项给出默认实现与可配置开关，禁止把推测写成论文事实。

**验收**：仅根据 `method_spec.md` 即可实现模型；所有推测均有标记。

## 3. 基础 3D 编码器复现

- [x] 接通 RGB-D、相机内参、外参和世界坐标系。
- [ ] 使用 CLIP 提取多视角 2D patch features。
- [x] 将深度像素/patch 反投影到 3D 世界坐标。
- [x] 实现独立的 3D position embedding（可学习 Fourier position encoding）。
- [ ] 将视觉 patch 与 3D position embedding 对齐并融合。
- [x] 实现 Farthest Point Sampling，将可变数量 patch 下采样为固定 `N` 个 token。
- [ ] 正确处理无效深度、视野重叠、padding 和 batch mask。
- [ ] 保存调试可视化：带颜色的点云、FPS 采样点和坐标轴。
- [ ] 对比 LLaVA-3D 原实现输出，检查数值范围、dtype 和 token shape。

**测试**：

- [ ] 单视角和多视角反投影测试。
- [ ] 相同 3D 点在不同视角下应落到相近世界坐标。
- [ ] FPS 输出恰好为 `N` 个 token，mask 正确。
- [ ] 编码器前向无 NaN/Inf，梯度可回传。

## 4. Episodic Memory Bank

- [ ] 定义 `MemoryEntry`，至少保存：
  - scene/room identifier
  - timestep
  - pose 或空间范围
  - memory tokens
  - token mask
  - 环境版本/交互后状态
- [x] 实现 observation → memory space 的 MLP 投影。
- [x] 实现 sinusoidal temporal embedding，并加到对应时间步的 memory tokens。
- [x] 实现 append：首次访问场景时写入新 entry。
- [x] 实现 update：交互导致场景改变或重访场景时更新对应 entry。
- [ ] 将 scene-ID 精确匹配和 pose-based 近似匹配设计成两种后端。
- [ ] 决定覆盖旧状态还是保留状态历史；两种策略均做成配置项。
- [ ] 支持 batch 内不同轨迹长度以及 padding/mask。
- [ ] 支持 memory bank 的序列化和恢复，便于调试长轨迹。
- [ ] 增加可选的最大容量、FIFO/LRU 淘汰策略，作为长轨迹扩展实验，不冒充论文默认设计。

**测试**：连续写入、重访覆盖、不同 batch 长度、序列化恢复和 mask 均正确。

## 5. 3D Memory Fusion

- [x] 实现 Working Memory 投影：`X_t → Q [B,N,M]`。
- [x] 实现历史 memory 的 K/V 投影：`[B,T,N,M] → [B,T*N,M]`。
- [x] 实现论文公式：

  ```text
  A = softmax(Q K^T / sqrt(M))
  F = A V
  MemoryEnhanced = concat(F, Q)
  ```

- [x] 对 padding 的历史 token 使用 attention mask。
- [ ] 确认 concat 维度是 token 维还是 feature 维；根据架构图优先实现 token 维，并做消融验证。
- [ ] 支持 PyTorch SDPA/Flash Attention，降低长 memory 的显存占用。
- [ ] 输出 attention map，支持按时间步、房间和 token 聚合可视化。
- [ ] 实现论文 Table 3 的三种 Query 初始化：
  - Working Memory（论文方法）
  - Most Recent Episodic Memory
  - Learnable Zero Parameters
- [ ] 增加可选的 instruction-conditioned query 作为扩展实验，与论文实现分开标记。

**测试**：

- [ ] Q/K/V 和输出 shape 正确。
- [ ] mask 后的 token 权重为 0。
- [ ] 构造“当前物体与某个历史 token 相同”的合成样例，正确历史应获得最高注意力。
- [ ] fp32 与 bf16 下无 NaN，反向传播正常。

## 6. 与 LLaVA-3D / LLM 集成

- [ ] 确定融合 memory token 在 LLM 输入序列中的位置。
- [ ] 定义 tokenizer special tokens：图像、3D memory、高层动作、思考/反馈、任务结束。
- [ ] 拼接 language instruction、fused episodic memory、working memory 和必要的 action history。
- [ ] 扩展 context length 到 8192，并正确配置 RoPE/position embedding。
- [ ] 加载 LLaVA-3D 预训练权重，记录 missing/unexpected keys。
- [ ] 默认只训练 memory module 与 LLM decoder；明确视觉编码器是否冻结。
- [ ] 实现标准 causal language modeling loss，并屏蔽非目标/padding token。
- [ ] 实现 autoregressive generation 与结构化动作解析器。
- [ ] 对非法动作、缺失参数和 `Task Complete` 提供确定性的解析/错误处理。

**验收**：给定一段短轨迹，模型能完成前向、loss、反向和动作文本生成。

## 7. 3DMem-Bench 数据复现

### 7.1 环境构建

- [ ] 获取 HM3D Semantics，并核实论文使用的 182 spaces / 2,602 rooms 筛选规则。
- [ ] 预处理房间和物体的 axis-aligned bounding boxes（AABB）。
- [ ] 获取 Objaverse 交互物体子集，并记录资产 ID、类别和缩放规则。
- [ ] 将交互物体放入 HM3D 场景，验证碰撞、可达性和语义标签。
- [ ] 固定 Habitat-Sim 版本、传感器参数、相机高度、分辨率、HFOV 和动作空间。

### 7.2 轨迹生成与验证

- [ ] 实现论文中的 box-demonstration-instruction prompting 数据格式。
- [ ] 若使用 Gemini 生成任务，固定模型版本、prompt、temperature 和调用日期。
- [ ] 保存原始生成结果，避免不可重复的二次调用。
- [ ] 实现高层命令：导航、拾取、放置、交互、完成任务。
- [ ] 实现 trajectory simulator validation：
  - agent 当前房间正确
  - 引用物体存在
  - pick-up/put-down 状态转换正确
  - 高层动作可由模拟器执行
- [ ] 生成 random exploration 的 RGB-D、camera pose 和重建点云。
- [ ] 保存交互前后 observation，支持论文所述的快速训练/推理加载。
- [ ] 按 simple/medium/hard 划分 3/5/10 个多房间设置。
- [ ] 构建 in-domain 和 in-the-wild split，确保 unseen object、unseen memory context 和新挑战无泄漏。

### 7.3 EQA 与 Captioning

- [ ] 构建五类 EQA：spatial、navigation、comparative、layout、count。
- [ ] 构建跨房间、交互前后的 episodic memory captioning 数据。
- [ ] 实现自动验证和人工抽样检查脚本。
- [ ] 输出数据统计，与论文的 26,276 train trajectories、1,860 embodied test tasks、865 EQA、167 captioning 对比。

**验收**：随机抽取的轨迹可在固定 seed 下重放；物体状态和答案与最终环境一致。

## 8. 训练流程

- [ ] 创建 smoke-test 配置：单 GPU、少量场景、几十个样本、10–50 steps。
- [ ] 创建论文配置：8192 context、global batch 256、1000 steps、Adam、LR `2e-5`、无 weight decay、3% warmup、cosine decay。
- [ ] 明确 gradient accumulation，使 global batch 与论文一致。
- [ ] 支持 bf16、gradient checkpointing 和分布式训练。
- [ ] 每次运行保存 config、git commit、seed、环境版本、日志和 checkpoint。
- [ ] 记录 trainable/total parameter 数。
- [ ] 监控 loss、梯度范数、学习率、tokens/s、显存和 memory attention entropy。
- [ ] 支持断点续训，并验证恢复后 optimizer/scheduler 状态一致。
- [ ] 至少运行 3 个随机种子；完整训练成本过高时，明确报告单 seed 限制。

**验收**：smoke test loss 能下降；完整训练没有数据泄漏、OOM 或持续 NaN。

## 9. Baseline 复现

- [ ] **3D-LLM / LLaVA-3D Finetuned**：不使用显式长期记忆。
- [ ] **Everything in Context**：将能放入上下文的全部历史 observation 送入模型。
- [ ] **Most Recent Memory**：只保留最近 observation。
- [ ] **Retrieval-Augmented Memory**：相似度检索 Top-K 历史 observation，再拼接到当前 observation 前。
- [ ] 若资源允许，复现或评估 3D-Mem；明确它不支持 embodied action execution 的差异。
- [ ] 所有 baseline 使用相同的数据 split、视觉编码器、LLM、训练 token 数和评测脚本。
- [ ] 记录每种方法的 memory token 数、上下文长度、推理显存和延迟，避免只比较准确率。

## 10. 评测与论文表格对齐

- [ ] 实现 embodied task Success Rate（SR）。
- [ ] 实现 Sub-Success Rate（Sub-SR），并核实子目标定义。
- [ ] 分别报告 simple/medium/hard、in-domain/in-the-wild。
- [ ] 实现五类 EQA 的开放式回答评测。
- [ ] 若采用 LLM-as-judge，固定 judge 模型、prompt 和缓存结果，同时增加 exact/normalized match 等可重复指标。
- [ ] 实现 captioning 的 BLEU-1、BLEU-4、METEOR。
- [ ] 复现 Table 2a、Table 2b、Table 3 的输出脚本。
- [ ] 为每项指标保存逐样本预测、失败原因和 bootstrap confidence interval。
- [ ] 对齐重点：
  - 3DLLM-Mem in-domain 平均 SR：论文 37.6
  - 3DLLM-Mem in-the-wild 平均 SR：论文 32.1
  - hard in-the-wild SR：论文 27.8
  - Working Memory Query 应优于另外两种初始化
- [ ] 数值未对齐时，依次排查数据 split、base checkpoint、token 数、动作解析、上下文截断、memory mask 和 simulator 状态。

## 11. 分析与消融

- [ ] Query 初始化消融：working / recent / learnable zero。
- [ ] 去掉 temporal embedding。
- [ ] 去掉 3D position embedding，仅使用 2D visual token。
- [ ] 不更新旧场景，仅追加 memory。
- [ ] 覆盖旧状态 vs 保留完整状态历史。
- [ ] 不同 FPS token 数 `N`。
- [ ] 不同历史长度 `T`。
- [ ] soft fusion vs Top-K observation retrieval。
- [ ] 当前观测 Query vs instruction-conditioned Query。
- [ ] 绘制性能—显存—延迟随 `T`/`N` 的曲线。
- [ ] 可视化成功与失败案例的跨时间 attention，判断是否真的召回正确房间和物体。

## 12. 工程质量与最终交付

- [ ] 使用配置文件管理所有路径和超参数，代码中不写死本机绝对路径。
- [ ] 为数据、memory、fusion、动作解析和指标增加单元测试。
- [ ] 添加最小 CI：lint + unit tests + CPU synthetic forward。
- [ ] README 给出环境安装、数据准备、最小 demo、训练、评测和结果表生成命令。
- [ ] 提供 `scripts/run_smoke_test.sh`。
- [ ] 提供 `scripts/train_3dllm_mem.sh`。
- [ ] 提供 `scripts/eval_all.sh`。
- [ ] 提供 `scripts/reproduce_tables.sh`。
- [ ] 输出最终复现报告 `docs/reproduction_report.md`：
  - 完成范围
  - 与论文一致/不一致的部分
  - 指标对比
  - 资源消耗
  - 已知限制
  - 可复现命令

## 推荐执行顺序

1. 资源核实与 LLaVA-3D 环境固定。
2. 用合成 token 完成 Memory Bank 和 Memory Fusion 单元测试。
3. 接通真实 RGB-D → 3D tokens。
4. 完成少量轨迹的端到端 smoke test。
5. 先跑 Most Recent、RAG 和 3DLLM-Mem 小规模对照。
6. 再投入完整数据构建与正式训练。
7. 最后复现消融、效率分析和论文表格。

## 当前状态

- [x] 论文 PDF 已放入项目目录。
- [x] 已根据论文正文和附录整理第一版复现任务清单。
- [x] 官方代码状态已核实：仓库为空，仅含 README。
- [x] 项目代码已初始化：L0 memory + 3D token encoder 已实现。
