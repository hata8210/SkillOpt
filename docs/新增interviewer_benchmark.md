# 新增 Interviewer Benchmark 方案

> 目标：基于 `data/interviewer_data/data.csv`（香港物业管理保安员面试数据）新增一个 SkillOpt benchmark。
> 任务定义：给定招聘岗位要求（`jd`）+ 面试对话记录（`context`），由模型按评分表给出总分，据此判断是否录用；监督标签为 `result` 列（`通过` / `不通過`）。
> 本文档为完整实现方案，含任务设计、评分规则、约束机制、文件清单、配置与验证步骤。
> 设计决策：**不改任何通用核心代码**，不占用 `SLOW_UPDATE` / `APPENDIX` 保护区语义；自定义 tag 仅作标识，约束主要靠环境专属 analyst prompt（软约束）。

---

## 目录

0. [背景与数据](#0-背景与数据)
1. [任务与评分设计](#1-任务与评分设计)
2. [「只 train 评分细则」的软约束方案](#2-只-train-评分细则的软约束方案)
3. [数据物化与切分](#3-数据物化与切分)
4. [文件清单](#4-文件清单)
5. [配置要点](#5-配置要点)
6. [运行与验证](#6-运行与验证)
7. [风险提示](#7-风险提示)

---

## 0. 背景与数据

### 0.1 数据源

- 文件：`data/interviewer_data/data.csv`（注意路径是 `interviewer_data`，不是 `interview_data`）
- 有效样本：34 条（已清理 8 个 `Unnamed` 空列；原文件末尾还有 1 行全空行，物化脚本应跳过）
- 列结构：

| 列名 | 类型 | 说明 |
|---|---|---|
| `job` | str | 岗位名称，全部为「物業管理員／保安員」 |
| `jd` | str | 岗位描述，34 条完全相同的固定文案（79 字） |
| `candidate` | str | 候选人姓名 |
| `age` | float | 年龄，范围 23–44 |
| `sex` | str | 性别，全部为「男」 |
| `interview_time` | str | 面试时间，格式 `YYYY/M/D H:M:S` |
| `context` | str | 面试对话全文（粤语为主），长度 291–2848 字符，平均约 1731 字符 |
| `result` | str | 面试结果：`通过` 24 条、`不通過` 10 条 |

### 0.2 任务定义

- 输入：岗位要求（`jd`）+ 面试对话记录（`context`）
- 输出：模型按评分表给出的**总分**（0~8，0.5 粒度）
- 判定：总分 → 区间 → 录用/不录用结论
- 监督：`result` 列（`通过` → 录用 hire；`不通過` → 不录用 reject）
- 参考：`data/interviewer_data/SKILL_template.md` 的「评分标准 + 总分与评估建议」结构

### 0.3 与 SKILL_template.md 的关系

- 评分表（8 个判断项 + 列结构 + 三个总分区间）作为**初始技能** `initial.md`，保持原文不变
- 数据里**没有简历字段**，所以 `question` 只含动态部分（候选人 + 面试记录 + 输出指令）
- 岗位要求（`jd`）是固定文案，放技能固定区，不进 `question`

---

## 1. 任务与评分设计

### 1.1 输出格式

让模型只输出总分，录用结论**不直接由模型输出**，而是按固定的「总分与评估建议」映射表确定——结论完全由评分表驱动，呼应"评分表是评估结果最重要依据"。

```text
<score>6.5</score>
<reason>（可选，供反思阶段参考）</reason>
```

解析失败 / 无 `<score>` 视为 0 分。

### 1.2 区间映射

与 `SKILL_template.md` 三个区间一致（0–4 / 4.5–6 / 6.5–8，0.5 粒度下自然无重叠，边界按规则连续化）：

| 总分 | 区间 | 结论 |
|---|---|---|
| `总分 ≤ 4` | 低分区 | 不录用（reject） |
| `4 < 总分 < 6.5` | 中间区 | 可考虑进入复试（不直接录用/不录用） |
| `总分 ≥ 6.5` | 高分区 | 录用（hire） |

### 1.3 奖励规则（统一 hard，单一奖励信号）

设计决策：**取消独立 soft 奖励，统一使用 hard 作为唯一奖励**。中间区（`4 < score < 6.5`）给部分奖励 `hard = 0.3`；`soft` 字段返回与 `hard` 同值，仅满足 trainer 契约（`id`/`hard`/`soft`），不承载独立语义。

gold 映射：`通过` → hire，`不通過` → reject。

| 预测区间 | hard | soft |
|---|---|---|
| 命中（高分区且 gold=通过；低分区且 gold=不通過） | 1 | 1.0（=hard） |
| 中间区（无论 gold 是什么） | 0.3 | 0.3（=hard） |
| 反区（低分区但 gold=通过；高分区但 gold=不通過） | 0 | 0 |
| 解析失败 / 无 `<score>` | 0 | 0 |

要点：

- **连续 hard 被官方支持**：`compute_score` docstring 注明 "hard may be continuous (0.0-1.0) when using smoothed reward"，gate 与日志均按 float 处理
- **不做分数平滑**：区间内一律固定值（命中 1 / 中间区 0.3 / 反区 0），不做"距边界距离"式的连续化；上述"连续 hard"仅指跨样本聚合时取浮点均值
- `evaluation.gate_metric` 保持默认 `hard` 即可（soft==hard 时 `mixed`/`soft` 均等价）
- **指标口径**：hard 均值是"平滑奖励均值"而非字面准确率；基线「全猜高分」= `24×1/34 + 10×0/34 ≈ 0.71`，「全猜中间区」= `0.3`
- **反思分组副作用**：`run_minibatch_reflect` 按 `hard` 是否为 0 分 success/failure，`hard=0.3` 的中间区轨迹会被归入 **success 组**；需在 `analyst_success.md` 中说明中间区是"部分正确"，引导分析师把规则推向决定性正确区间，而非强化停留在中间区

### 1.4 转换位置与数据流

分数 → 结论的转换**统一发生在打分阶段（evaluator）**，物化阶段不做任何转换：

```text
物化侧（保留原始标签，不转换）                        打分侧（evaluator，统一转换）
result 列 ─────────────► ground_truth/answers = "不通過"  ── gold_to_zone() ──► reject
模型输出 <score>6.5</score> ─────────────────────────── parse + score_to_zone() ──► hire
                                                                              │
                           hard = 1（命中）/ 0.3（中间区）/ 0（反区），soft = hard  ──► 写入 rollout 结果
```

两个方向的转换规则：

- **gold 侧**：物化时**原样**写入 `result` 列的 `"通过"/"不通過"`（不转成 hire/reject）；转换在 evaluator 的 `gold_to_zone(label)`：`通过 → hire`、`不通過 → reject`
- **预测侧**：模型只输出 `<score>`，不输出结论；evaluator 解析 `<score>` 后由 `score_to_zone(score)` 按固定边界转区间（`≤4 → reject`、`4 < score < 6.5 → middle`、`≥6.5 → hire`），再与 gold 区间比较产出统一 hard（soft 同值）

为什么集中在 evaluator 而不是物化时转换：

- **单一事实来源**：`parse_score()` / `score_to_zone()` / `gold_to_zone()` 集中在 `evaluator.py` 一处；以后改边界或扩展结论类别只改这一处
- **可核对性**：物化产物保留 `"不通過"` 原始标签，人工抽查能直接对上源 CSV
- **边界一致性安全**：区间表在 skill 固定区（不 train），evaluator 写死的边界与模板永远一致，不会出现"技能改了边界、代码没跟上"的漂移

落地位置：`evaluator.py` 负责解析与转换并产出 hard/soft；`rollout.py` 调用 evaluator 并把 `id` / `hard` / `soft` 写入结果 dict；trainer 只消费这两个字段。

### 1.5 hard / soft 的影响边界（训练流程中的消费位置）

`hard` 与 `soft` 在训练流程中的消费位置不同，边界如下（基于现有核心代码行为，本次不改核心）：

| 环节 | 消费信号 | soft 的作用 |
|---|---|---|
| ② REFLECT 反思分组 → 修改意见生成 | **仅 hard** | 无（`soft` 不被读取） |
| ⑥ EVALUATE / gate（accept-reject、best_skill 选优） | 默认 **hard**（由 `evaluation.gate_metric` 决定） | 可选：`mixed` / `soft` 时参与 |
| 日志 / `history.json` / 指标观测 | hard + soft 都聚合（`compute_score`） | 观测值 |

具体机制：

- **修改意见生成（② REFLECT）只认 hard**：`run_minibatch_reflect`（`skillopt/gradient/reflect.py`）按 `hard` 是否为 0 把轨迹分成 failure / success 两组，决定 analyst 用 `analyst_error` 还是 `analyst_success` prompt 分析、产出哪类 patch。`soft` 完全不参与分组——中间区轨迹（hard=0.3）会进入 **success 组**
- **训练评判（⑥ EVALUATE / gate）默认只看 hard**：`select_gate_score(hard, soft, gate_metric, mixed_weight)`（`skillopt/evaluation/gate.py`）被 baseline、每步 selection、gate、best_skill 选择共用；`gate_metric` 为 `hard`（默认）时 `soft` 不参与，为 `soft`/`mixed` 时才进入比较
- **日志观测**：`compute_score`（`skillopt/utils/scoring.py`）把 hard/soft 都聚合成均值，写入日志与 `history.json`

对本 benchmark 的含义：由于 `soft == hard`，无论 `gate_metric` 配成 `hard` / `soft` / `mixed` 结果都等价；**唯一实质边界是反思分组**——中间区的 0.3 会被当"成功"分析，靠 `analyst_success.md` 的说明引导其不强化中间区（见 2.2 与文件清单）。

---

## 2. 「只 train 评分细则」的软约束方案

训练约束：**希望绝大部分修改落在「评分细则表」规则行上**（增/删/改）；表头列结构、三个总分区间、岗位要求及其它表述应尽量保持不动。

约束强度说明：这是**软约束**，不做机制级强制。自定义 tag 不会被核心识别为保护区（`_PROTECTED_REGIONS` 只认识 `SLOW_UPDATE` / `APPENDIX`），它们的作用是给 analyst 明确的区域标识和指令锚点。若训练中偶尔修改到固定区，属于可接受范围；如需更强约束，可后续在 env 内做编辑过滤（见 2.4）。

### 2.1 `initial.md` 结构（复制 SKILL_template.md 原文，只加标识 tag）

```markdown
<!-- SKILL_FIXED_START -->
本次分析岗位为：香港物业管理保安员。
岗位核心职责：巡逻、访客登记、处理客户投诉及查询、车辆记录。
岗位入职条件：学历小六或以上、具备一年及以上相关工作经验。
（使用说明：按下方评分细则逐项打分，合计总分后按「总分与评估建议」区间给出录用结论）
<!-- SKILL_FIXED_END -->

<!-- TABLE_HEADER_START -->
### 香港物业管理保安员岗位面试评分表
| 判断项 | 合格标准 | 0分 | 0.5分 | 1分 |
|--------|----------|-----|-------|-----|
<!-- TABLE_HEADER_END -->
| **1. 自我介绍** | … | … | … | … |   ← 8 行规则 = 主要可编辑区
| **2. 工作单位** | … | … | … | … |
…（可增减规则行）

<!-- SKILL_FIXED_TAIL_START -->
### 总分与评估建议
| 总分范围 | 评估建议 |
|----------|----------|
| **0 – 4 分** | 不建议进入复试 → 不录用 |
| **4.5 – 6 分** | 可考虑进入复试，需在复试中重点核实薄弱项（中间区） |
| **6.5 – 8 分** | 建议进入复试 → 录用 |
<!-- SKILL_FIXED_TAIL_END -->
```

- 表头行、岗位要求、总分区间表等**固定内容用 tag 标识**，作为 analyst 的识别锚点（无强制保护）
- 三组 tag 仅存在于本环境的 skill 中，对其它环境无任何影响
- 规则行是**主要可编辑区**，允许增/删/改

### 2.2 约束机制：环境专属 analyst prompt（软约束）

写环境专属 `prompts/analyst_error.md`、`prompts/analyst_success.md`（复制通用版改写），明确约束：

- 编辑应尽量且优先落在「评分细则表」规则行内（`TABLE_HEADER` 与 `TABLE_HEADER_END` 之间的内容）；`<!-- SKILL_FIXED_* -->` / `<!-- TABLE_HEADER_* -->` 之间的文本属于固定区域，非必要不要修改
- 若确需调整，禁止改动表头行（五列结构：判断项/合格标准/0分/0.5分/1分）、禁止改变列数、禁止修改总分区间映射（0–4 / 4.5–6 / 6.5–8）
- 新增规则行必须用 `insert_after` 锚定某条已有规则行，禁止用 `append`
- target 尽量用整行唯一文本（避免短词命中固定区域造成误编辑）
- 不讨论输出格式、语言风格等与评分细则无关的表述；所有反思结论应落到"评分细则如何改"

### 2.3 配置层

- `optimizer.skill_update_mode: patch`（**禁用** `full_rewrite` / `rewrite_from_suggestions`，防止整篇重写破坏格式）
- 关闭 `use_slow_update`、`use_meta_skill`（避免往保护附录注入额外表述，污染 target 看到的 skill）

### 2.4 后续增强（如需要）

若软约束下固定区被修改频率过高，可二选一（均不改核心代码）：

1. 在 `InterviewerAdapter` 中覆盖 `reflect()`：过滤 `run_minibatch_reflect` 产出的 edits，丢弃 target 落在固定 tag 区域内的编辑（代码全在 `skillopt/envs/interviewer/`）
2. 进一步强化 analyst prompt 措辞（如给出"禁止"清单与示例）

---

## 3. 数据物化与切分

写 `scripts/materialize_interviewer.py`（仿 `materialize_searchqa.py`）：

1. 读取 CSV（路径通过外部参数 `--csv` 传入，默认用法：`data/interviewer_data/data.csv`），跳过全空行
2. 规范化 item：

```json
{
  "id": "interviewer_001",
  "candidate": "甘俊明",
  "context": "面试对话全文",
  "question": "候选人：甘俊明（男，35岁）\n\n面试记录：\n…\n\n请根据技能中的评分标准逐项打分，输出总分，格式：<score>X.X</score>",
  "answers": ["不通過"],
  "ground_truth": "不通過",
  "task_type": "interviewer"
}
```

- `id` 唯一（`interviewer_001`… 或候选名）
- `question` 离线生成（固定、可复现），只含动态部分；jd 和评分表留在技能固定区
3. 切分策略（本次采用**全量共用**）：train/val/test 三份都写入**全部 34 条**（内容相同）
4. 输出 `data/interviewer_split/{train,val,test}/items.json` + `split_manifest.json`

**为什么全量共用**：样本只有 34 条，若按比例抽分，train/val/test 每份只剩个位数，反而让各环节覆盖变片面（比如 test 可能抽不到足够的「不通過」样本）；本次暂时不考虑过拟合问题，保证训练、selection、最终评估都覆盖完整样本分布。

**可行性确认**：`SplitDataLoader` 的 `split_dir` 模式只是分别从三个目录加载 items，**没有跨 split 去重或一致性校验**；同一批 `id` 出现在不同 split 目录不会冲突（各环节的 rollout 输出目录相互独立）。训练 epoch 会按 shuffle 后的 34 条完整走一遍。

**保留切换能力**：物化脚本提供 `--split-method stratified` 参数（按 `result` 分层切分 train/val/test），当前默认 `full`（34/34/34）；后续有更多数据或需要真实泛化评估时，直接切换即可。

注意：全量共用下，val/test 与 train 同分布，训练中的 gate/selection 与最终评估指标都会偏乐观（更接近"记忆化"而非"泛化"），详见第 7 节风险。

---

## 4. 文件清单

| 文件 | 内容 |
|---|---|
| `scripts/materialize_interviewer.py` | CSV → split 数据（`--csv` 必填传入源 CSV；默认全量 34/34/34，可选 `--split-method stratified`） |
| `skillopt/envs/interviewer/__init__.py` | 空包 |
| `skillopt/envs/interviewer/dataloader.py` | `InterviewerDataLoader(SplitDataLoader)`，只实现 `load_split_items()` |
| `skillopt/envs/interviewer/evaluator.py` | `<score>` 解析、区间判定、hard/soft 打分 |
| `skillopt/envs/interviewer/rollout.py` | `system=skill`，`user=question` → `chat_target` → 打分 → 写 `predictions/<id>/conversation.json` |
| `skillopt/envs/interviewer/adapter.py` | 继承 `EnvAdapter`，实现 4 个抽象方法，`reflect()` 继承默认 |
| `skillopt/envs/interviewer/skills/initial.md` | SKILL_template.md 原文 + 自定义标识 tag |
| `skillopt/envs/interviewer/prompts/analyst_error.md`、`analyst_success.md` | 带软约束的分析师 prompt（编辑落在评分细则行；success 侧含"中间区=部分正确、不强化中间区"的说明） |
| `configs/interviewer/default.yaml` | `env.name: interviewer`，Patch 模式，小 batch |
| 修改 `scripts/train.py`、`scripts/eval_only.py` | 各注册一行 `_ENV_REGISTRY["interviewer"] = ...`（包 try/except） |

**不需要动的文件（核心代码零改动）**：`skillopt/engine/trainer.py`、`skillopt/envs/base.py`、`skillopt/datasets/base.py`、`skillopt/gradient/*`、`skillopt/optimizer/*`、`skillopt/evaluation/*`、`skillopt/model/*`、`skillopt/config.py`。

**接口契约**（易踩坑）：

- item 必须有 `"id"`（str）
- `rollout()` 返回每条必须有 `"id"`、`"hard"`（0~1 连续，平滑奖励）、`"soft"`（与 hard 同值）
- adapter `__init__` 形参名与配置扁平化后的键对齐（`get_adapter` 按签名注入）
- `env.name` 必须等于注册名 `interviewer`
- **必须写 `predictions/<id>/conversation.json`**，否则反思阶段 `skip_no_patches`，技能学不到东西
- 用 `skillopt.model.chat_target` 路由模型，不裸调 OpenAI/Claude

---

## 5. 配置要点

`configs/interviewer/default.yaml` 关键项：

```yaml
_base_: ../_base_/default.yaml

train:
  train_size: 34
  batch_size: 8
  accumulation: 1
  num_epochs: 2

gradient:
  minibatch_size: 4
  merge_batch_size: 4
  analyst_workers: 8

optimizer:
  learning_rate: 2
  min_learning_rate: 2
  lr_scheduler: constant
  skill_update_mode: patch
  use_slow_update: false
  use_meta_skill: false

evaluation:
  gate_metric: hard

env:
  name: interviewer
  skill_init: skillopt/envs/interviewer/skills/initial.md
  split_mode: split_dir
  split_dir: data/interviewer_split
  max_completion_tokens: 4096
  workers: 4
  limit: 0
```

说明：

- `train_size: 34`：全量共用，一个 epoch 正好完整过一遍全部样本
- `batch_size: 8`：每个 epoch 约 4~5 个 step（34 / 8）
- `learning_rate: 2`：样本少，编辑预算调小减少无效编辑
- `minibatch_size: 4`：样本少，反思 minibatch 相应调小
- `evaluation.gate_metric: hard`：统一 hard 奖励，soft==hard，无需 mixed
- `max_completion_tokens: 4096`：粤语 context 最长 2848 字符 + 技能文本，够用
- 调试阶段用 `limit: 10`、`batch_size: 4`

---

## 6. 运行与验证

### 6.1 数据物化

```bash
python scripts/materialize_interviewer.py --csv data/interviewer_data/data.csv
```

核对：train/val/test 数量均为 34（全量共用），标签分布均为 通过 24 / 不通過 10。

### 6.2 训练

先加载环境变量（`openai_compatible` 后端在模块导入时读取 `OPENAI_COMPATIBLE_*`，必须在运行前 source，否则 API key 会兜底成占位符 `dummy` 导致 401）：

```bash
set -a; source .env; set +a
```

```bash
python scripts/train.py \
  --config configs/interviewer/default.yaml \
  --cfg-options \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4-flash \
    model.target=deepseek-v4-flash \
  --out_root outputs/interviewer_quickstart
```

### 6.3 评估

同样先加载环境变量：

```bash
set -a; source .env; set +a
```

```bash
python scripts/eval_only.py \
  --config configs/interviewer/default.yaml \
  --cfg-options \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4-flash \
    model.target=deepseek-v4-flash \
  --skill outputs/interviewer_quickstart/best_skill.md \
  --split valid_unseen
```

### 6.4 约束生效验证（软约束）

1. 检查 `history.json` / 训练日志中 edits 的 `target` 分布：绝大多数应指向「评分细则表」规则行
2. 对比最终 skill 与 `initial.md`：表头行、列数、总分区间、岗位要求应基本保持不变
3. 若固定区被频繁修改，说明 analyst prompt 约束不足，需强化措辞或启用 2.4 的 env 内编辑过滤

---

## 7. 风险提示

- **全量共用 → 评估乐观**：val/test 与 train 同分布，训练中 gate/selection 与最终评估都是"见过的样本"，指标偏记忆化、无泛化信号；作为试点可接受，后续用新增数据（或切回 `--split-method stratified`）做真实评估
- **标签偏斜**：通过 24 / 不通過 10；全量共用下训练与评估分布一致，偏斜影响主要在基线统计口径（全猜高分的基线 hard≈0.71，全猜中间区为 0.3）
- **中间区被归入 success 组**：`hard=0.3` 使中间区轨迹进入反思的成功组，`analyst_success.md` 已加入"部分正确不强化"说明；若训练后技能出现"倾向给中间分"的倾向，需检查并强化该说明
- **jd 单一**：技能会是「香港保安员招聘评估」的专项技能，泛化有限
- **软约束强度有限**：固定区（表头/区间表/岗位要求）无法机制级禁止修改，个别编辑可能落在固定区；可接受，如频率过高再启用 2.4 增强
- **Patch 模式残余风险**：多步编辑可能把规则行改成非法 markdown 行；分析师 prompt 约束可覆盖大部分场景，若发现可再补"应用后格式校验"回调
- **`append` 限制**：新增行必须用 `insert_after`（append 会落到文首），已在分析师 prompt 中约束

---

## 附录：关键参考

- `docs/QuickStart学习笔记.md`：SearchQA quickstart 实战笔记（调用链、接口契约、DeepSeek 接线）
- `docs/新增train说明.md`：新增数据集说明（必须新写的文件、注册、配置）
- `docs/guide/new-benchmark.md`：官方英文手把手教程（docfaithful 完整最小示例）
- `skillopt/envs/officeqa/`：最接近本方案的参考环境（dataloader / adapter 结构）
- `skillopt/optimizer/skill.py`：保护区机制（`_PROTECTED_REGIONS`，本次不修改，仅了解其行为）
