# 新增 Interviewer Benchmark 方案

> 目标：基于 `data/interviewer_data/data.csv`（香港物业管理保安员面试数据）新增一个 SkillOpt benchmark。
> 任务定义：给定招聘岗位要求（`jd`）+ 面试对话记录（`context`），由模型按评分表给出总分，据此判断是否录用；监督标签为 `result` 列（`通过` / `不通過`）。
> 本文档为完整实现方案，含任务设计、评分规则、保护机制、文件清单、配置与验证步骤。

---

## 目录

0. [背景与数据](#0-背景与数据)
1. [任务与评分设计](#1-任务与评分设计)
2. [「只 train 评分细则」的保护方案](#2-只-train-评分细则的保护方案)
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

### 1.3 hard / soft 规则

gold 映射：`通过` → hire，`不通過` → reject。

| 预测区间 | hard | soft |
|---|---|---|
| 命中（高分区且 gold=通过；低分区且 gold=不通過） | 1 | 1.0 |
| 中间区（无论 gold 是什么） | 0 | 0.5 |
| 反区（低分区但 gold=通过；高分区但 gold=不通過） | 0 | 0 |
| 解析失败 / 无 `<score>` | 0 | 0 |

要点：

- **hard**：只在"命中"时给 1，即预测区间与 gold 的决定性区间一致
- **soft**：中间区给固定 0.5 奖励（用户明确要求）；命中区给 1.0
- 可选进阶：soft 可在命中区内用总分距边界距离连续化（如 `1 - |score - 边界| / 2.5`）；起步建议先用固定值，信号更稳
- **不要用 LLM-judge 做 soft**（不稳定会毁 optimizer）

---

## 2. 「只 train 评分细则」的保护方案

训练约束：**只允许增删改「评分细则表」规则行的内容**；表头列结构、三个总分区间、岗位要求及其它所有表述一律不动。

分四层保障：

### 2.1 第 1 层：`initial.md` 结构（复制 SKILL_template.md 原文，只加标记）

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
| **1. 自我介绍** | … | … | … | … |   ← 8 行规则 = 唯一可编辑区
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

- 表头行（列结构）用 `TABLE_HEADER` 标记保护，规则行裸露可编辑
- 岗位要求、总分区间表等其它表述全部用 `SKILL_FIXED` / `SKILL_FIXED_TAIL` 保护
- 三组标记在**其它环境不存在**时完全无影响（见 2.2）

### 2.2 第 2 层：机制层（保护区）

`skillopt/optimizer/skill.py` 已有通用保护区机制（`_PROTECTED_REGIONS` 标记对，`APPENDIX` 就是这么加的）。做两处小改动：

1. 在 `_PROTECTED_REGIONS` 元组追加三组标记对：
   - `(SKILL_FIXED_START, SKILL_FIXED_END)`
   - `(TABLE_HEADER_START, TABLE_HEADER_END)`
   - `(SKILL_FIXED_TAIL_START, SKILL_FIXED_TAIL_END)`
2. 把三组标记加进 `_strip_slow_update_markers` 与 `evaluation/gate.py` 的 strip 列表

效果（既有逻辑自动生效）：

- `replace` / `delete` / `insert_after` 锚点落在固定区 → 自动 `skipped_protected_region` 跳过
- 规则行的 `replace`（改写细则）、`insert_after`（增行）、`delete`（删行）→ 正常生效
- `append` 会落到文档最早保护区之前 → 在分析师 prompt 中禁止使用 `append`

### 2.3 第 3 层：Prompt 层（分析师约束）

写环境专属 `prompts/analyst_error.md`、`prompts/analyst_success.md`（复制通用版改写），明确约束：

- 只能增删改「评分细则表」规则行的内容
- 禁止修改表头行、禁止改变列数、禁止修改总分区间映射、禁止修改岗位要求及其它任何表述
- 新增规则行必须用 `insert_after` 锚定某条已有规则行，禁止用 `append`
- 不讨论输出格式、语言风格等与评分细则无关的表述

### 2.4 第 4 层：配置层

- `optimizer.skill_update_mode: patch`（**禁用** `full_rewrite` / `rewrite_from_suggestions`，防止整篇重写破坏格式）
- 关闭 `use_slow_update`、`use_meta_skill`（避免往保护附录注入额外表述，污染 target 看到的 skill）

---

## 3. 数据物化与切分

写 `scripts/materialize_interviewer.py`（仿 `materialize_searchqa.py`）：

1. 读取 `data/interviewer_data/data.csv`，跳过全空行
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
3. 按固定 seed 做**按 `result` 分层**随机切分（保 通过24/不通過10 比例）
4. 输出 `data/interviewer_split/{train,val,test}/items.json` + `split_manifest.json`

建议比例：train 20 / val 7 / test 7。

---

## 4. 文件清单

| 文件 | 内容 |
|---|---|
| `scripts/materialize_interviewer.py` | CSV → split 数据（分层切分） |
| `skillopt/envs/interviewer/__init__.py` | 空包 |
| `skillopt/envs/interviewer/dataloader.py` | `InterviewerDataLoader(SplitDataLoader)`，只实现 `load_split_items()` |
| `skillopt/envs/interviewer/evaluator.py` | `<score>` 解析、区间判定、hard/soft 打分 |
| `skillopt/envs/interviewer/rollout.py` | `system=skill`，`user=question` → `chat_target` → 打分 → 写 `predictions/<id>/conversation.json` |
| `skillopt/envs/interviewer/adapter.py` | 继承 `EnvAdapter`，实现 4 个抽象方法，`reflect()` 继承默认 |
| `skillopt/envs/interviewer/skills/initial.md` | SKILL_template.md 原文 + 保护标记 |
| `skillopt/envs/interviewer/prompts/analyst_error.md`、`analyst_success.md` | 带编辑约束的分析师 prompt |
| `configs/interviewer/default.yaml` | `env.name: interviewer`，Patch 模式，小 batch |
| 修改 `scripts/train.py`、`scripts/eval_only.py` | 各注册一行 `_ENV_REGISTRY["interviewer"] = ...`（包 try/except） |
| 修改 `skillopt/optimizer/skill.py`、`skillopt/evaluation/gate.py` | 保护区元组追加自定义标记 + strip 列表 |

**不需要动的通用文件**：`skillopt/engine/trainer.py`、`skillopt/envs/base.py`、`skillopt/datasets/base.py`、`skillopt/gradient/*`、`skillopt/model/*`、`skillopt/config.py`。

**接口契约**（易踩坑）：

- item 必须有 `"id"`（str）
- `rollout()` 返回每条必须有 `"id"`、`"hard"`（0/1）、`"soft"`（0~1 float）
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
  train_size: 20
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

- `learning_rate: 2`：34 条小样本，编辑预算调小减少无效编辑
- `minibatch_size: 4`：样本少，反思 minibatch 相应调小
- `max_completion_tokens: 4096`：粤语 context 最长 2848 字符 + 技能文本，够用
- 调试阶段用 `limit: 10`、`batch_size: 4`

---

## 6. 运行与验证

### 6.1 数据物化

```bash
python scripts/materialize_interviewer.py
```

核对：train/val/test 数量（20/7/7）与标签分布（train 约 14/6）。

### 6.2 训练

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

### 6.4 保护生效验证

1. 检查 `history.json` / 训练日志中 edits 的 `status`，应出现 `skipped_protected_region`
2. 对比最终 skill 与 `initial.md`：表头行、列数、三个总分区间、岗位要求必须逐字一致
3. 只允许差异出现在「评分细则表」的规则行（增/删/改）

---

## 7. 风险提示

- **样本量小（34 条）**：train 仅 20 条，指标波动大，结果作为试点跑通管线；后续扩充数据更有意义
- **标签偏斜**：通过 24 / 不通過 10，切分已按 result 分层缓解
- **jd 单一**：技能会是「香港保安员招聘评估」的专项技能，泛化有限
- **Patch 模式残余风险**：多步编辑可能把规则行改成非法 markdown 行；分析师 prompt 约束可覆盖大部分场景，若发现可再补"应用后格式校验"回调
- **`append` 限制**：新增行必须用 `insert_after`（append 会落到文首），已在分析师 prompt 中约束

---

## 附录：关键参考

- `docs/QuickStart学习笔记.md`：SearchQA quickstart 实战笔记（调用链、接口契约、DeepSeek 接线）
- `docs/新增train说明.md`：新增数据集说明（必须新写的文件、注册、配置）
- `docs/guide/new-benchmark.md`：官方英文手把手教程（docfaithful 完整最小示例）
- `skillopt/envs/officeqa/`：最接近本方案的参考环境（dataloader / adapter 结构）
- `skillopt/optimizer/skill.py`：保护区机制（`_PROTECTED_REGIONS`）
