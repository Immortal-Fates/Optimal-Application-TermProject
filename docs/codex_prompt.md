# 给 Codex 的完整项目 Prompt（可直接复制粘贴）

你是 **Codex**。请你为《优化方法及应用》课程大作业生成一个 **可复现、可快速运行** 的端到端项目（代码 + 实验 + 分析 + 图表 + 报告骨架），主题是：

> **4 张 GPU 的 PyTorch DDP 分布式训练中，因 NLP 变长序列导致每卡计算/显存占用动态波动。设计并实现一种“动态 per-GPU micro-batch 调度”策略，通过动态调整每张 GPU 每步处理的样本数，使资源利用更均衡、降低尾步时、提高吞吐。**

我希望你输出一个结构清晰、代码规范、易运行的仓库项目。**默认实验要很快**（几分钟内），但要能产生足够的现象与对比结果（图表 + 指标）。

---

## 0. 重要约束与我提供的条件

- 我有 **4 张 GPU**（同一台机器），希望用 `torchrun --nproc_per_node=4` 运行。
- 主数据集：**SST-2**（GLUE 子任务），要求数据易获取、规模小。
- 任务要求：训练实验要“轻量快跑”，不追求 SOTA，只要能体现 **step time / GPU 显存** 的不均衡现象，以及调度策略带来的改善。
- 代码要求：**良好的代码风格与规范**（模块化、类型标注、日志、可复现、配置化、合理的错误处理）。
- 产出要求：训练过程要保存必要数据；提供分析脚本与绘图脚本；生成可写进大作业报告的结果与图。

---

## 1. 你要交付的产物（必须全部生成）

请在仓库内生成下列内容（文件名可调整，但功能必须齐全）：

1. **训练代码**：PyTorch DDP + Transformers + SST-2 微调，支持两种实验模式  
   - `static`：每卡固定 micro-batch（baseline）  
   - `dynamic`：每卡动态 micro-batch（你的调度策略）
2. **调度算法模块**：单独文件/类实现（可测试、可复用），并写清楚接口与文档。
3. **数据与指标记录**：每 step 记录并保存到 CSV/JSONL（推荐 CSV）  
   - 每卡 step time、peak memory、local batch size、padding 长度或 token 统计、吞吐等  
   - rank0 合并成一个 `merged_metrics.csv`
4. **分析与绘图脚本**：读取 merged 指标，自动生成对比图表与 summary 表格（CSV/Markdown）  
   - 输出到 `outputs/<run_id>/figures/`
5. **一键运行脚本**：运行 baseline + dynamic + analysis  
   - 如 `scripts/run_all.sh` 或 Makefile
6. **README.md**：包含环境安装、运行命令、输出解释、如何复现图表
7. **报告骨架**：`report/report.md`（或 LaTeX）  
   - 包含：摘要、前言、建模、求解方法、实验、结论、参考文献  
   - 把数学模型、目标函数、约束、算法伪代码写好，并留出插图位置  
   - 结论部分要能直接引用你生成的图和统计表

---

## 2. 研究问题与优化建模要求（写进 report）

你需要在报告中把问题形式化为一个“每步的小规模整数优化/离散优化”。

### 2.1 场景描述
- DDP 同步训练中，每步耗时由最慢 GPU 决定  
- 由于 NLP 变长序列与动态 padding，每个 rank 的 batch “有效 token 数 / padding 长度”不同 → 每步计算与显存峰值会波动且不均衡  
- 静态每卡 batch size 会导致尾步时上升与吞吐降低

### 2.2 决策变量
- 第 t 步，每卡 micro-batch：\( b_{g,t} \in \mathbb{Z}_{\ge 1} \)，g=1..G，G=4  
- 全局 batch 约束：\(\sum_g b_{g,t} = B\)（B 为给定的 global batch target，可配置）

### 2.3 成本/代理目标（必须可计算）
你需要给一个 per-rank 的 cost proxy \(c_g(b)\)，用于调度求解。要求：
- 能从“下一批候选样本的长度信息”快速估算  
- 与该 rank 的计算/显存开销单调相关

推荐做法（实现必须包含）：
- 每个 rank 在本步开始前预取 K 个样本到 buffer（K >= max 可能的 batch 上限）
- 对 k=1..K，计算前 k 个样本的最大长度 `max_len[k]`（token 数），并构造  
  - `padded_tokens[k] = k * max_len[k]`
- 计算成本代理：
  - 计算代理：`c_time[k] = padded_tokens[k] ** 2`（近似 attention 的 L^2）或 `padded_tokens[k]`（更保守简单）
  - 显存代理：`c_mem[k] = padded_tokens[k]`
- 目标：最小化每步的 “最大成本”，并在同等 max 成本下尽量降低方差（更均衡）

### 2.4 优化问题（每步求解）
你需要在每一步求解下面的离散优化（**world_size=4 很小，允许用枚举求 exact**）：

\[
\min_{b_1,\dots,b_G} \ \max_g c_g(b_g) \quad
\text{s.t.}\ \sum_g b_g = B,\ 1 \le b_g \le b^{\max}_g,\ b_g \in \mathbb{Z}
\]

Tie-break（可选但推荐）：
- 在达到同样的 \(\max_g c_g(b_g)\) 的解中，选择 `Var(c_g(b_g))` 更小的解

并在报告里说明：
- 这是一个 min-max 的整数优化问题  
- 因为 G=4、B 不大，你可以用枚举在每步得到 exact 解（这也是“优化求解方法”的亮点）

---

## 3. 关键工程点（实现必须正确）

### 3.1 DDP 下允许 per-rank batch size 不同的正确梯度缩放
DDP 默认会在 allreduce 后按 world_size 平均梯度。若各 rank 的 batch size 不同，为了等价于全局 batch 的平均梯度，你需要 **对每个 rank 的 loss 做权重缩放**：

- 设本步全局 batch \(B = \sum_g b_g\)
- 每个 rank 的 local batch 为 \(b_g\)
- 令 local loss 用 **mean reduction**（每个样本平均）得到 \(L_g\)
- backward 前将其缩放为：

\[
\tilde L_g = L_g \cdot \frac{G \cdot b_g}{B}
\]

这样 allreduce 的平均梯度等价于全局平均梯度（请在 report 里说明这个推导要点）。

实现要求：
- 每步通过 `dist.all_reduce` 汇总 local batch 得到 B（即使理论上固定为目标 B，也要做，保证健壮性）
- 训练必须能稳定跑完整个 steps，不因不一致而 hang

### 3.2 OOM 风险控制
为了保证“最可交版本”稳定，请实现以下两层保护：
1. `max_length` + truncation，限制序列长度（默认 256，可配置）
2. 启动时对每个 GPU 做一个 **快速容量探测**，得到该卡在 worst-case 长度下的 `b_max_g`（例如用合成全长序列做一次 forward/backward，指数增长探测 1/2/4/8/16...，遇到 OOM 回退）

调度求解必须遵守 `b_g <= b_max_g`。

---

## 4. 实验设计（必须跑得快但要有结果）

### 4.1 模型与数据
- 模型：`distilbert-base-uncased` + 分类头（Transformers）
- 数据：SST-2（从 Hugging Face datasets 下载）
- 训练默认步数：`steps=300`（可配置），warmup 少量 steps
- 优先用 AMP（fp16 或 bf16）加速与省显存（自动检测支持情况）

为了更快：
- 允许只用 SST-2 train 子集（例如随机取 10k 或 5k），并在 report 里说明这是为了快速实验
- 验证集可以每隔 N steps 抽样评估（例如 200 条）以证明训练仍合理（非必须每步评估）

### 4.2 对比方法（至少两个）
**Baseline：static**
- 固定每卡 micro-batch = B/4（要求 B 可被 4 整除）
- 其余训练配置与 dynamic 完全一致

**你的方法：dynamic**
- 每步通过“预取 buffer + 求解离散优化”得到 b_g
- 每步 batch 分配不同

### 4.3 评价指标（必须记录）
每步记录（每 rank + rank0 汇总）：
- `step`: step index
- `rank`
- `local_batch`
- `max_seq_len_in_batch`
- `padded_tokens = local_batch * max_seq_len_in_batch`
- `step_time_ms`（forward+backward+optimizer 的 wall time）
- `peak_mem_mb`（本步 peak allocated，建议用 `torch.cuda.max_memory_allocated()` ）
- `throughput_samples_per_s`（全局）
- `throughput_tokens_per_s`（全局）

并额外计算用于对比的 summary（rank0 在 analysis 中生成）：
- mean step time / P95 step time（以 per-step 的 max(rank step time) 为准）
- mean throughput / P95 throughput
- 资源均衡指标：每步 `std(peak_mem)`、`std(step_time)`、`std(padded_tokens)`，再取均值或 P95

---

## 5. 你必须生成的图（analysis 脚本自动输出）

请在 `analysis/analyze_metrics.py` 里生成并保存以下图（matplotlib）：

1. **每步最大步时** 随 step 变化：static vs dynamic（两条曲线）
2. **每步 step time 的跨卡方差/标准差**：static vs dynamic
3. **每步 padded_tokens 的跨卡方差/标准差**：static vs dynamic（证明调度在“长度”层面做到了均衡）
4. **peak memory 的跨卡标准差**：static vs dynamic
5. **step time 分布图**：static vs dynamic（hist 或 boxplot）
6. 汇总表：输出 `summary.md` 或 `summary.csv`，包含两种方法的 mean、P95、提升百分比

所有图保存为 `png`，并在 README/报告里引用路径。

---

## 6. 代码结构与风格要求（必须遵守）

### 6.1 推荐仓库结构
请生成类似结构（可微调）：

```
.
├─ README.md
├─ requirements.txt
├─ pyproject.toml              # 可选：ruff/black 配置
├─ scripts/
│  ├─ run_all.sh
│  ├─ run_static.sh
│  └─ run_dynamic.sh
├─ src/
│  ├─ train_ddp.py
│  ├─ scheduler.py
│  ├─ data.py
│  ├─ metrics.py
│  ├─ distributed.py
│  └─ utils.py
├─ analysis/
│  └─ analyze_metrics.py
├─ report/
│  └─ report.md
└─ outputs/
   └─ ... (运行时生成)
```

### 6.2 代码规范
- 全部 Python 代码加类型标注（`from __future__ import annotations`）
- 关键类/函数写 docstring
- 日志统一用 `logging`，rank0 负责打印关键 summary，其它 rank 少打印
- 训练与调度逻辑分离（scheduler 独立可单测）
- 配置通过 argparse + 可选 YAML（例如 `--config configs/default.yaml`）
- 每次运行输出一个唯一 `run_id` 文件夹，保存：
  - config（yaml/json）
  - 原始 metrics（每 rank）
  - merged metrics
  - figures
  - summary

---

## 7. 运行方式要求（README 必须包含）

必须支持如下命令（示例参数你自行定默认值）：

```bash
# 1) 安装
pip install -r requirements.txt

# 2) baseline
torchrun --nproc_per_node=4 -m src.train_ddp --mode static --dataset sst2 --steps 300 --global_batch 32 --max_length 256

# 3) dynamic
torchrun --nproc_per_node=4 -m src.train_ddp --mode dynamic --dataset sst2 --steps 300 --global_batch 32 --max_length 256

# 4) 分析
python -m analysis.analyze_metrics --run_dir outputs/<run_id_static> --compare_dir outputs/<run_id_dynamic>
```

并提供 `scripts/run_all.sh` 一键跑完。

---

## 8. 实现建议与“最可交版本”的具体策略（请直接实现）

### 8.1 预取 buffer + exact 枚举求解
- 设置 `K=32`（可配置）
- 每个 rank 的 DataLoader 输出单样本（不 padding）
- 维护一个 `buffer: list[dict]`，不足 K 就继续从 iterator 取样本填充
- 对 k=1..min(K, b_max_g) 计算 `max_len[k]` 与 cost
- 使用 `dist.all_gather` 收集各 rank 的 `costs` 数组到 rank0
- rank0 枚举所有满足 `sum b_g = B` 的组合，选择 min-max cost 最小方案（ties 选 var 最小）
- 广播得到 `b_g` 给所有 rank
- 每 rank 从 buffer pop 出 b_g 个样本，使用 `DataCollatorWithPadding` 组 batch 训练

### 8.2 训练循环（固定 steps）
- 不强调 epoch 概念，训练按固定 steps 走（更快更稳定）
- 支持 `--seed`，尽量可复现
- 每步记录本 rank 的 metrics，定期同步到 rank0 合并

### 8.3 结果一定要“可解释”
- 在报告里解释为什么用 `padded_tokens`（或平方）作为 proxy
- 解释 min-max 目标与 DDP 同步慢卡瓶颈的关系
- 用图证明 dynamic 确实降低了跨卡 padded_tokens 不均衡，并带来 step time 的 P95 改善

---

## 9. 额外加分项（可选，但如果不难请做）

1. 在 scheduler 中提供两种 cost：
   - `--cost_type tokens` 使用 padded_tokens
   - `--cost_type tokens_sq` 使用 padded_tokens^2
   并在分析中对比哪种更能改善尾延迟
2. 在小规模 B 下，输出 “exact 解” 与 “贪心解” 的差距（若实现贪心），作为优化方法讨论亮点
3. 输出一次简短的 “失败/限制” 分析：例如 proxy 与真实 step time 的偏差来源、通信开销、序列分布变化等

---

## 10. 输出格式要求（非常重要）
请你直接生成完整仓库的所有文件内容。  
如果你不能真正写文件，请以如下方式输出：

- 先输出项目文件树
- 然后按文件路径逐个给出内容，例如：

```text
# File: README.md
...内容...

# File: src/train_ddp.py
...内容...
```

不要省略关键实现，不要写伪代码替代核心模块。  
代码要能直接复制到本地运行。

---

## 11. 允许使用的依赖（尽量少、稳定）
必须依赖：
- torch
- transformers
- datasets
- numpy
- pandas
- matplotlib
- tqdm
- pyyaml（可选）

不要引入大型复杂框架（如 Ray、Spark），保持轻量。

---

## 12. 你完成后应达到的“验收标准”
- `static` 与 `dynamic` 两种模式都能在 4 GPU 下跑完默认 steps
- `outputs/` 下产生 metrics 与 figures
- `analysis` 能生成对比图与 summary 表
- README 说明清晰，一键脚本可跑
- report 骨架可直接填图写结论，且包含数学建模与求解方法描述

---

**现在开始按以上要求生成整个项目。请务必以“可直接运行”为第一优先级，并保证默认实验耗时短。**
