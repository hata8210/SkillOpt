# SkillOpt QuickStart 学习笔记

> 基于一次完整跑通 SearchQA quickstart 的实战会话整理，按对话问答顺序归纳。
> 仓库：`microsoft/SkillOpt`；本次实操模型：DeepSeek（`deepseek-v4-flash`，走 OpenAI 兼容后端）。

---

## 目录

0. [实战背景与准备（跑通 quickstart 全程）](#0-实战背景与准备)
1. [SearchQA 数据集结构](#1-searchqa-数据集结构)
2. [split 是什么](#2-split-是什么)
3. [train.py / rollout.py / evaluator.py 的关系](#3-trainpy--rolloutpy--evaluatorpy-的关系)
4. [ReflACTTrainer 通用性 & 新增数据集](#4-reflacttrainer-通用性--新增数据集)
5. [rollout 的两种模式：直接推理 vs Agent 执行](#5-rollout-的两种模式直接推理-vs-agent-执行)
6. [materialize_searchqa.py 的作用](#6-materialize_searchqapy-的作用)
7. [官方 2000 条 ID 的决定](#7-官方-2000-条-id-的决定)
8. [reflection 体现在哪个环节、作用是什么](#8-reflection-体现在哪个环节作用是什么)
9. [searchqa/reflect.py：通用还是自实现](#9-searchqareflectpy通用还是自实现)
10. [新增train说明.md vs new-benchmark.md 对比](#10-新增train说明md-vs-new-benchmarkmd-对比)
11. [hard 与 soft 指标如何界定](#11-hard-与-soft-指标如何界定)
12. [optimizer 与 reflection 的区别](#12-optimizer-与-reflection-的区别)
13. [minibatch reflect vs epoch 级机制](#13-minibatch-reflect-vs-epoch-级机制)

---

## 0. 实战背景与准备

### 目标

跑通官方 SearchQA quickstart：

```bash
git clone https://github.com/microsoft/SkillOpt.git
cd SkillOpt
python -m pip install -e ".[searchqa]"

cp .env.example .env
set -a; source .env; set +a

python scripts/materialize_searchqa.py

python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --out_root outputs/searchqa_quickstart

python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --skill outputs/searchqa_quickstart/best_skill.md \
  --split valid_unseen
```

模型调用使用环境变量里的 DEEPSEEK 信息：

```bash
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_API_KEY=sk-...        # 35 位
DEEPSEEK_MODEL=deepseek-v4-flash
```

### 模型后端接线（重要）

SkillOpt 里 DeepSeek 走**通用 OpenAI 兼容后端**（`openai_compatible`），需要把 DEEPSEEK 变量映射为它认识的环境变量（写入 `.env`）：

```bash
export OPENAI_COMPATIBLE_BASE_URL="${DEEPSEEK_API_BASE:-https://api.deepseek.com}"
export OPENAI_COMPATIBLE_API_KEY="${DEEPSEEK_API_KEY:-}"
export OPENAI_COMPATIBLE_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
```

训练/评估时还要显式把后端和模型写进配置（`--cfg-options`），否则会落到默认的 `azure_openai` / `gpt-5.5`：

```bash
--cfg-options \
  model.optimizer_backend=openai_compatible \
  model.target_backend=openai_compatible \
  model.optimizer=deepseek-v4-flash \
  model.target=deepseek-v4-flash
```

说明：`model.optimizer`（负责改技能）和 `model.target`（负责做题）可以指向同一个模型，这里都用 DeepSeek。

### 踩坑记录（按出现顺序）

1. **依赖安装需要联网**：`pip install -e ".[searchqa]"` 会装 `datasets`、`azure-identity` 等，沙箱里需授权联网。
2. **HF 数据集下载卡住**：`materialize_searchqa.py` 要下载 `lucadiliello/searchqa`（train parquet ≈ 284MB、validation ≈ 41MB），下载到一半停滞。解决：用 `curl -C -` 断点续传 + `sha256sum` 校验，再手动放到 HF 缓存目录（blobs + snapshot 软链）。
3. **pyarrow SIGILL（关键坑）**：本机 VM 的 CPU 缺 AVX/BMI2 指令，pyarrow 读 parquet 的嵌套列（`answers`/`labels`）时在 `DefLevelsToBitmapBmi2WithRepeatedParent` 内核崩溃（`vmovq` 指令非法），pyarrow 12~25 全崩。解决：改用 **fastparquet** 读 parquet，写了一个等价物化脚本（逻辑同 `materialize_searchqa.py`，输出格式一致），生成 `data/searchqa_split/{train,val,test}/items.json`。
4. **config 的 env 段被删（仓库 bug）**：`skillopt/config.py` 里扁平键 `env`（来自 `env.name`）与结构化 section 名 `env` 同名冲突，`_resolve_layer_format_duplicates` / `_drop_base_keys_overridden_by_layer` 把整个 `env:` 配置段删掉（17b4823 引入的回归），导致所有带 `env:` 的结构化配置加载后 env 丢失。已修复（跳过对 dict 型 section 的删除）并加回归测试，41 个测试通过。
5. **DEEPSEEK 环境变量会话中途消失 / 忘记 source**：父 shell 的快照轮换导致变量丢失，训练报 401 `Authentication Fails`；`openai_compatible` 后端在模块导入时读 env，key 缺失会兜底成占位符 `dummy` 并打到默认 OpenAI 端点，epoch 末 slow update 表现为 `Incorrect API key provided: dummy` + `[slow update] no guidance produced`。解决：把凭据**直接写进 `.env`**（自包含，不依赖父 shell；`.env` 已在 `.gitignore` 中），每次运行前先 `set -a; source .env; set +a`。
6. **训练中断**：4-epoch 训练在 step 15/40 时会话断开（当时最佳 selection 0.54）。按需求改为 **epochs=1** 重跑（`train.num_epochs=1`，共 10 步），约 1 小时完成。

### 最终结果

```text
训练（1 epoch，DeepSeek deepseek-v4-flash）：
  最终 test hard = 0.5371

评估（eval_only --split valid_unseen，1400 条）：
  hard = 0.5293  soft = 0.5763
  输出：outputs/eval_searchqa_deepseek-v4-flash_20260825_214114/eval_summary.json

训练产物：outputs/searchqa_quickstart/
  best_skill.md（学习到的问答技能，2500+ 字符）
  history.json（每步历史）、config.json、steps/、selection_eval_baseline/ ...
```

---

## 1. SearchQA 数据集结构

### 来源与规模

- 来源：Jeopardy!（危险边缘）题库 + 基于问题的 Google 搜索文档，经 MRQA 2019 格式化
- HF 仓库：`lucadiliello/searchqa`
  - train 117,384 条、validation 16,980 条
  - 2 个 parquet：train ≈ 284MB、validation ≈ 41MB
- 本地物化：`data/searchqa_split/` → train 400 / val 200 / test 1400（由官方 ID manifest 抽取）

### 原始 parquet 列结构（5 列）

| 列名 | 类型 | 说明 |
|---|---|---|
| `key` | `string` | 样本唯一 ID，32 位十六进制（类似 MD5），manifest 靠它定位样本 |
| `question` | `string` | Jeopardy 风格的问题/线索 |
| `context` | `string` | 多个搜索结果文档拼接的长文本，带结构化标记 |
| `answers` | `sequence<string>` | 标准答案列表（本数据每行一般只有 1 个答案） |
| `labels` | `list<{end: list<int64>, start: list<int64>}>` | 答案在 context 中的 span；本版本 117,384 行全部为 null，未使用 |

### context 的结构化标记

`context` 是多个文档拼接的文本，用三种标签分隔：

- `[DOC]` — 一个新文档开始
- `[TLE]` — 文档标题（title）
- `[PAR]` — 段落（paragraph）

```text
[DOC] [TLE] KING OF THE 'TECHNO-THRILLER' - The New York Times [PAR] May 1, 1988 ...
[DOC] [TLE] Jeopardy! #1 Flashcards | Quizlet [ ...
```

### 物化后的 items.json

`data/searchqa_split/{train,val,test}/items.json` 是 JSON 数组，每条 4 个字段：

```json
{
  "id": "221c83e6630f4e7983da48fa28da1882",
  "question": "The New York Times Magazine has called him \"King of the Techno-Thriller\"",
  "context": "[DOC] [TLE] KING OF THE 'TECHNO-THRILLER' - The New York Times [PAR] ...",
  "answers": ["Tom Clancy"]
}
```

注意：`answers` 虽然后设显示为嵌套 `LIST<LIST<STRING>>`，实际每行解析出来是单元素字符串列表（如 `["Tom Clancy"]`）。

### demo 中数据如何被使用

- 输入构造（`skillopt/envs/searchqa/rollout.py`）：skill 文档 + `question` + 截断到 6000 字符的 `context` 拼进 prompt，要求模型输出 `<answer>...</answer>`；`_truncate_context` 按 `[DOC]` 边界截断
- 评估（`skillopt/envs/searchqa/evaluator.py`）：提取 `<answer>`（无标签取最后非空行）→ SQuAD 规范化 → `em` / `f1` / `sub_em`

---

## 2. split 是什么

`split` = 数据切分：把完整数据集按样本 ID 划分成互不相交的子集，各司其职。

### 两层形态

1. **ID manifest**（`data/searchqa_id_split/`）：只存 ID 的清单（train 400 / val 200 / test 1400），不存数据正文；`split_manifest.json` 说明 `source_id_field: key`
2. **物化后的可运行数据**（`data/searchqa_split/`）：按 manifest ID 从 11 万条里捞出样本，写成 `{train,val,test}/items.json`

### 各 split 用途

| split | 数量 | 别名 | 用途 |
|---|---|---|---|
| `train` | 400 | `train` | 训练循环每步取 40 条：rollout → 反思 → 编辑 → 更新技能 |
| `val` | 200 | `valid_seen` / `selection` | 训练中评估：baseline、每步 200 条 selection、gate 是否接受新技能 |
| `test` | 1400 | `valid_unseen` | 最终评估，训练中完全不用，衡量泛化 |

别名映射见 `skillopt/datasets/base.py`：`valid_seen → val`、`valid_unseen → test`。

### 配置控制

```yaml
env:
  split_mode: split_dir   # split_dir=用预切分目录；ratio=按 split_ratio 随机切
  split_dir: data/searchqa_split
```

### 单条数据能看出属于哪个 split 吗？

不能。item 本身没有 split 字段，归属完全由"这个 ID 出现在哪份 manifest"决定；物化后靠**目录**区分（`train/` 下就是 train）。

---

## 3. train.py / rollout.py / evaluator.py 的关系

### 调用链

```text
scripts/train.py（训练入口）
 └─ 构建 SearchQAAdapter + ReflACTTrainer（skillopt/engine/trainer.py）
     └─ trainer.train() 里所有"跑一批样本"的地方都调 adapter.rollout(...)
         └─ skillopt/envs/searchqa/adapter.py 的 rollout()
             └─ skillopt/envs/searchqa/rollout.py 的 run_batch() → process_one()
                 ├─ 拼 prompt、调 LLM（chat_target 或 run_target_exec）
                 └─ 调 skillopt/envs/searchqa/evaluator.py 的 evaluate() 打分
```

### 各文件职责

- **`scripts/train.py` — 编排层**：解析 CLI、加载配置、通过环境注册表找 adapter、启动 trainer；不关心"怎么答一道题"
- **`skillopt/envs/searchqa/adapter.py` — 环境适配器**：继承 `EnvAdapter`，`rollout()` 只做转发给 `run_batch`
- **`skillopt/envs/searchqa/rollout.py` — 执行层**：`run_batch()` 多线程并发 + 断点续跑；`process_one()` 处理单条（拼 prompt → 调模型 → 打分）
- **`skillopt/envs/searchqa/evaluator.py` — 评估层**：纯字符串处理（不调 LLM），提取答案 → 规范化 → em/f1/sub_em

### 数据流

```text
trainer 取 train 的 40 条（BatchSpec）
  → adapter.build_train_env() / build_eval_env()（从 split 取样本）
  → adapter.rollout(env, skill, out_dir)
  → rollout.run_batch() 并行跑 process_one()
  → 每条的 evaluate() 打分
  → 汇总 hard/soft 回报 trainer
  → trainer 用 hard 做 gate（accept/reject）、记录 history.json
```

`eval_only.py` 走的是同一条路：`adapter.rollout` → `run_batch` → `process_one` → `evaluate`。

---

## 4. ReflACTTrainer 通用性 & 新增数据集

### ReflACTTrainer 是通用的

`skillopt/engine/trainer.py` 的 `ReflACTTrainer` 不 import 任何具体环境，只依赖 `skillopt/envs/base.py` 的抽象接口 `EnvAdapter`：

- 必须实现（`@abstractmethod`）：`build_train_env` / `build_eval_env` / `rollout` / `get_task_types`
- 有默认实现可覆盖：`setup` / `get_dataloader` / `reflect`（默认走 `run_minibatch_reflect`）等

训练循环、gate、selection、慢更新、meta-skill 全是通用逻辑。

### train.py 怎么找到 SearchQAAdapter

靠环境注册表（`scripts/train.py` 和 `scripts/eval_only.py` 各一份）：

```python
_ENV_REGISTRY: dict[str, type] = {}

def _register_builtins():
    from skillopt.envs.searchqa.adapter import SearchQAAdapter
    _ENV_REGISTRY["searchqa"] = SearchQAAdapter

def get_adapter(cfg):
    _register_builtins()
    env_name = cfg.get("env", "alfworld")     # 来自配置 env.name
    adapter_cls = _ENV_REGISTRY[env_name]
    # 用 inspect 按 __init__ 签名注入扁平化配置里的参数
    return adapter_cls(**adapter_kwargs)
```

### adapter.rollout 实现在哪

- `skillopt/envs/searchqa/adapter.py:76` — `SearchQAAdapter.rollout()`：转发参数给 `run_batch`
- `skillopt/envs/searchqa/rollout.py:360` — `run_batch()`：并发 + 断点续跑
- `skillopt/envs/searchqa/rollout.py:153` — `process_one()`：单条执行 + 调 `evaluate()`

### 新数据集：写哪些、复用哪些

**必须新写：**

| 文件 | 内容 |
|---|---|
| `skillopt/envs/<name>/adapter.py` | 继承 `EnvAdapter`，实现 `__init__`、`build_train_env`、`build_eval_env`、`rollout`、`get_task_types` |
| `skillopt/envs/<name>/dataloader.py` | 继承 `SplitDataLoader`，实现读数据（`load_split_items()` 是唯一必须） |
| `skillopt/envs/<name>/rollout.py` | 任务执行：调模型/agent、产出含 `id`/`hard`/`soft` 的结果 |
| `skillopt/envs/<name>/evaluator.py` | 打分逻辑（可选拆分；评分也可直接写在 rollout 里） |
| `skillopt/envs/<name>/prompts/` | 可选：环境专属 analyst/system prompt |
| `skillopt/envs/<name>/skills/initial.md` | 初始技能文档 |
| `configs/<name>/default.yaml` | 配置（`env.name` 填环境名） |
| `scripts/train.py` + `scripts/eval_only.py` | 各注册一行 `_ENV_REGISTRY["<name>"] = YourAdapter` |

**完全通用不用动：** `trainer.py`、`envs/base.py`、`datasets/base.py`、`gradient/*`、`model/*`、`config.py`、`prompts/*`。

**接口契约（易踩坑）：**

- item 必须有 `"id"`（str）
- `rollout()` 返回每条必须有 `"id"`、`"hard"`（0/1 或连续）、`"soft"`（0~1）
- adapter `__init__` 形参名要与配置扁平化后的键对齐（`get_adapter` 按签名注入）
- `env.name` 必须等于注册名

---

## 5. rollout 的两种模式：直接推理 vs Agent 执行

`process_one()` 里由 `is_target_exec_backend()`（`skillopt/model/backend_config.py:167`）判定：

```python
if is_target_exec_backend():
    # 模式二：agent 执行（Codex / Claude Code / Cursor / Copilot）
    response, raw, system, user = _run_codex_once(...)
else:
    # 模式一：直接 prompt 推理（chat_target 一次调用）
    resp_text, _ = chat_target(system=system, user=user, ...)
```

### 模式一：prompt 直接推理（本 demo 用的）

- `rollout.py:273` 附近：`_build_system(skill)` + `_build_user(question, context)` → `chat_target()` 一次调用 → 拿 `<answer>` 打分
- 特点：单次对话、无工具、无代码执行

### 模式二：agent 执行

- `_run_codex_once()`（`rollout.py:101`）：`prepare_workspace()` 建工作区（技能写成 `skillopt-target` 技能、题目写成 `task.md`）→ `run_target_exec()` 启动 agent 读任务、按技能执行，返回 `<answer>`
- 特点：模型是"干活"的 agent，可查文件、写代码、跑命令；适合需要检索/计算/跑代码的任务

### 切换方式

```bash
# agent 模式
--cfg-options model.target_backend=codex_exec          # 或 claude_code_exec / cursor_exec / copilot_exec

# 直接推理模式
--cfg-options model.target_backend=openai_compatible  # 或 openai_chat / qwen_chat / minimax_chat / claude_chat
```

判定逻辑：`TARGET_BACKEND in {"codex_exec", "claude_code_exec", "cursor_exec", "copilot_exec"}`。

---

## 6. materialize_searchqa.py 的作用

quickstart 的第 4 步：

```bash
python scripts/materialize_searchqa.py
```

### 干什么

把"只含 ID 的清单"变成"真正可跑的样本数据"：

- **输入**：`data/searchqa_id_split/` 的 manifest（train/val/test 的 `items.json`，只存 ID）
- **过程**：联网加载完整 HF 数据集 `lucadiliello/searchqa` → 按 manifest 的 `key` 匹配 → 规范化成 `{id, question, context, answers}`
- **输出**：`data/searchqa_split/{train,val,test}/items.json` + `split_manifest.json`

### 不调用会怎样

直接报错，训练跑不起来：

```text
ValueError: Missing 'train/' subdirectory in split_dir: .../data/searchqa_split
```

因为 config 配了 `split_mode: split_dir`，`SplitDataLoader.setup()` → `_load_all_splits()` 找不到 `train/` 就抛错。

### 等价替代

只要最终 `data/searchqa_split/{train,val,test}` 就绪即可：

- 自己写脚本生成同格式 `items.json`（本次就是用 fastparquet 脚本等价实现）
- 或 config 改 `split_mode: ratio` + 给 `data_path`，让 dataloader 自己切

---

## 7. 官方 2000 条 ID 的决定

### 在哪里决定

在仓库里**随代码提交的 manifest**：`data/searchqa_id_split/`（train 400 / val 200 / test 1400，共 2000 个唯一 ID，跨 split 零重复）。`materialize_searchqa.py` 只是读取它。

### 单条能区分 train 还是 eval 吗

不能从 item 本身区分（没有 split 字段）；用 ID 查 manifest，或看物化目录（`train/`、`val/`、`test/`）。

### 输出格式由谁决定

`materialize_searchqa_splits` 决定输出 `{split}/items.json`（JSON 数组、`{id, question, context, answers}`）+ `split_manifest.json`。真正的约束在下游：`SplitDataLoader` 读目录里第一个 `.json`，adapter/rollout 要求 item 含 `id/question/context/answers`。

### 新数据集要重写吗

**要**（或等价替换）：manifest 路径、HF 数据集名、`key` 字段、必填字段、规范化逻辑全是 SearchQA 特化。三种做法：照抄一个物化脚本 / 绕过它自备 split 目录 / 用 `split_mode: ratio`。

---

## 8. reflection 体现在哪个环节、作用是什么

### 环节

每个 step 的 6 个阶段中的第 ② 阶段（日志 `[2/6 REFLECT minibatch]`）：

```text
① ROLLOUT    target 模型做题（40 条）
② REFLECT    ← 反思在这里（trainer.py:1237 → adapter.reflect → run_minibatch_reflect）
③ AGGREGATE  合并 patch
④ SELECT     按预算挑编辑
⑤ UPDATE     应用进技能（skill 104 → 3151 字符）
⑥ EVALUATE   val 200 条重跑，决定 accept/reject
```

### 机制

1. 输入：刚 rollout 的轨迹（模型输出、预测 vs gold、`predictions/<id>/conversation.json`）
2. 按结果拆 failure / success 组，各自按 M=8 切 minibatch（日志 `failure=10→2 groups success=30→4 groups`）
3. 并行调"分析师"LLM（**optimizer 模型**）：每组一个 minibatch，读多条轨迹 + 当前技能 + 专属 prompt（`analyst_error.md` / `analyst_success.md`）
4. 输出 patch：`failure_summary` + `edits`（`append` / `insert_after` / `replace` / `delete`），存 `patches/minibatch_{fail,succ}_XXX.json`

### 作用

把 rollout 的 0/1 分数变成"技能应该怎么改"的结构化编辑指令——类比神经网络的 loss→梯度：

| 神经网络 | SkillOpt |
|---|---|
| forward（推理） | ROLLOUT |
| loss + 梯度 | REFLECT（找失败共性 + 生成编辑） |
| 优化器更新参数 | SELECT + UPDATE 技能文档 |

关键点：找**共性模式**（不修个例）、失败和成功都反思、minibatch 一起看（省调用更稳）。这也是"反思式训练"（ReflACT = Reflective Agentic Training）名字的由来。

---

## 9. searchqa/reflect.py：通用还是自实现

**不是实现，是空壳**：`skillopt/envs/searchqa/reflect.py` 整个文件 91 字节，只有 docstring（"Prompts are now loaded from .md files by the base adapter"），是 v0.1.0 留下的占位模块。

`SearchQAAdapter` **没有覆盖 `reflect()`**，所以走基类默认实现：

```text
adapter.reflect(...)          # skillopt/envs/base.py:234（默认实现）
  → run_minibatch_reflect()   # skillopt/gradient/reflect.py:485（通用引擎）
```

SearchQA 特有的只是两份 prompt 文件（`skillopt/envs/searchqa/prompts/analyst_error.md`、`analyst_success.md`），两级优先级：环境专属 > 通用默认（`skillopt/prompts/`）。

### 通用逻辑（run_minibatch_reflect）流程

1. 按 `hard` 拆 failure / success（`failure_only=true` 时跳过 success）
2. 各自 shuffle（failure 用 `batch_seed`，success 用 `batch_seed+1`）
3. 按 `minibatch_size`（M=8）切组
4. 断点续跑：`patches/` 下已有文件就跳过
5. `ThreadPoolExecutor(workers)` 并行：失败组 → `run_error_analyst_minibatch`；成功组 → `run_success_analyst_minibatch`（都走 `chat_optimizer`）
6. 每组产出 patch dict 写盘，返回给 trainer 进入 ③④⑤

**新环境默认不用写 reflect**；只有需要自定义反思逻辑（看隐藏参考、特殊失败分类等）才覆盖。

---

## 10. 新增train说明.md vs new-benchmark.md 对比

- `docs/新增train说明.md`：基于 SearchQA quickstart 的**中文实战笔记**（数据结构 + 脚本关联 + 新增数据集清单 + 踩坑）
- `docs/guide/new-benchmark.md`：官方**英文手把手教程**（带完整最小示例 `docfaithful`）

### 重叠（一致）

- 新数据集要写：dataloader / rollout / adapter / config + 注册
- trainer 与反思管线通用、`reflect()` 默认继承
- 契约：item 有 `id`；结果有 `id`/`hard`/`soft`
- 参考：officeqa 最简单、`skillopt/envs/_template/` 可作起点

### 官方有、新增train说明.md 漏掉的（重点）

1. **conversation.json 是反思的前提**：`rollout/predictions/<id>/conversation.json` 缺失会被反思跳过，表现为 `skip_no_patches`
2. **评分在 rollout 里做**：ABC 没有 `evaluate()` 方法；`evaluator.py` 只是可选模块拆分
3. **必须用 `skillopt.model.chat_target`** 路由到配置的 chat 后端，不裸调 OpenAI/Claude
4. 调试建议：`batch_size: 4` + `limit: 10` 起步；noisy 评分毁 optimizer
5. 注册要包 `try/except ImportError`
6. 报错排查：`Unknown environment`=忘了注册；`TypeError: Can't instantiate abstract class`=没实现 4 个抽象方法
7. dataloader 只需 `load_split_items()`；`load_raw_items` 是 ratio 模式可选
8. `_base_` 必须是字符串不是 list
9. `skills/initial.md` 要先创建
10. 非 `id/hard/soft` 字段进 `RolloutResult.extras`（`skillopt/types.py`）
11. conversation 两种格式：`{role: system/user/assistant}` 或 `{type, content}` 都能被 `fmt_trajectory` 读
12. `get_task_types` 可从 dataloader 各 split items 收集 task_type

### 新增train说明.md 有、官方漏掉的

- SearchQA 数据集结构细节（列、context 标记、items.json、split 别名）
- 脚本调用链总览与文件职责
- rollout 两种模式与切换
- 接口契约细节（`__init__` 形参名对齐配置键、`env.name` 匹配）
- DeepSeek 实操命令与踩坑记录

### 出入点

- `evaluator.py` 是否必须：新增train说明.md 说必须，官方说评分在 rollout 里做——**以官方为准**更通用

---

## 11. hard 与 soft 指标如何界定

### 单条 item：hard = EM，soft = F1

`skillopt/envs/searchqa/rollout.py:242`：

```python
result["hard"] = int(eval_result["em"])   # Exact Match：0 或 1
result["soft"] = eval_result["f1"]        # token 级 F1：0.0 ~ 1.0
```

- **hard**：EM。预测与 gold 规范化后完全相等得 1，否则 0（硬指标）
- **soft**：F1。按词重叠算精确率/召回率调和平均，对 gold 列表取最大（软指标，反映部分答对）

规范化（`normalize_answer`，SQuAD 约定）：小写 → 去标点 → 去 a/an/the → 压缩空白。

举例：

| 预测 | gold | em (hard) | f1 (soft) |
|---|---|---|---|
| `Tom Clancy` | `["Tom Clancy"]` | 1 | 1.0 |
| `Tom Clancey` | `["Tom Clancy"]` | 0 | ~0.5 |
| `techno-thriller writer Tom Clancy` | `["Tom Clancy"]` | 0 | 1.0 |

### batch 聚合：取均值

`compute_score`（`skillopt/utils/scoring.py`）：`hard = 均值`（即准确率）、`soft = 均值`。日志 `[1/6 done] hard=0.7500 soft=0.7833` = 40 条中 30 条 EM 正确，F1 均值 0.7833。

### gate 怎么消费

`select_gate_score`（`skillopt/evaluation/gate.py:85`）把 (hard, soft) 压成比较分数：

- `gate_metric: hard`（默认）→ score = hard
- `soft` → score = soft
- `mixed` → (1−w)·hard + w·soft（`gate_mixed_weight` 默认 0.5）

gate 用这个分数 accept/reject 候选技能（日志 `ACCEPT (new best) hard=0.5150 > prev best 0.5050`）。

### 新数据集通用约定

`hard` 0/1（或连续奖励）、`soft` 0~1；trainer 只按这两个字段聚合、gate、选优，指标本身可换（ROUGE、LLM-judge 等）。

---

## 12. optimizer 与 reflection 的区别

"optimizer" 有两层意思，容易混：**模型角色**（optimizer 模型）和**流程阶段**（③④⑤）。

### 流程层面：reflection 是②，optimizer 对应③④⑤

| | ② REFLECTION | ③④⑤ OPTIMIZER |
|---|---|---|
| 本质 | 诊断/发散：发现问题、提候选修改 | 决策/收敛：合并、按预算挑选、落地 |
| 输入 | 失败/成功轨迹 + 当前技能 | patches + 编辑预算 |
| 输出 | 多个 patch（可重叠、带 failure_summary） | 排名后 top-L edits + 更新后的技能 |
| 约束 | 只提共性、可泛化的修改 | `optimizer.learning_rate`（edit_budget）、`lr_scheduler`、`skill_update_mode` |

日志对照：

```text
[2/6 done] failure_patches=2 success_patches=4        ← reflection 产出 6 个 patch
[3/6 done] merged 7 edits                              ← aggregate 合并
[4/6 SELECT] 7 -> 4 edits (budget=4, lr_control=fixed) ← select 按预算砍
[5/6 UPDATE] skill_len 4884 -> 6298                    ← update 应用
```

### 模型层面：同一个 optimizer 模型，不同角色

reflection 和 optimizer 阶段都调 `chat_optimizer`（`model.optimizer`），不是两个模型，只是 prompt 不同：

- reflection → 分析师 prompt（`analyst_*.md`）
- aggregate → 合并 prompt（`merge_*.md`）
- select → 排名 prompt（`ranking.md`，超预算时调用，`rank_and_select` 在 `skillopt/optimizer/clip.py:25`）
- update → 重写 prompt（rewrite 模式）

真正的模型分工：**target 负责"答题"（chat_target），optimizer 负责"复盘 + 改技能"**。

类比：reflection = 医生会诊（开处方建议）；optimizer = 主治医师拍板（合并建议、按预算选药、写进治疗方案）。

---

## 13. minibatch reflect vs epoch 级机制

前面详细讲的（分析师看轨迹、产出 patch edits）是 **minibatch 级（每步）reflect**（日志每步的 `[2/6 REFLECT minibatch]`）。epoch 级还有两个机制，但不叫 reflect：

| | minibatch REFLECT（每步） | SLOW UPDATE（epoch 末） | META SKILL（epoch 末） |
|---|---|---|---|
| 触发 | 每个 step 的 ② | `trainer.py:1730` | `trainer.py:2050` |
| 输入 | 当前 batch 轨迹 | 新抽样集（`slow_update_samples=20`）上新旧技能对比 rollout | 纵向对比对（上 epoch 技能 vs 当前技能） |
| 产出 | patch edits（候选编辑） | 一段"慢更新指导"文本 | optimizer 侧"元技能"记忆 |
| 落到哪 | 给 ③④⑤ 合并/挑选/应用 | 注入技能**保护段**（`<!-- SLOW_UPDATE_START -->`…`<!-- SLOW_UPDATE_END -->`，minibatch 反思不能改） | 作为下一 epoch 的 `meta_skill_context` |
| 作用 | 逐步快速迭代 | 跨 epoch 稳定、慢速指导（防抖） | 跨 epoch 优化器长期记忆 |

日志对照：

```text
[2/6 REFLECT minibatch] failure=10→2 groups success=30→4 groups  ← 每步
[SLOW UPDATE epoch 1] ...                                          ← epoch 末
[META SKILL epoch 1] skipped — first epoch                         ← epoch 末（1 epoch 时跳过）
```

配置：`gradient.minibatch_size` 控制 minibatch reflect；`optimizer.use_slow_update` / `optimizer.slow_update_samples` 控制 slow update；`optimizer.use_meta_skill` 控制 meta skill（默认都开）。

---

## 附录 A：关键文件索引

| 文件 | 作用 |
|---|---|
| `scripts/train.py` | 训练入口，环境注册表 |
| `scripts/eval_only.py` | 评估入口，环境注册表 |
| `scripts/materialize_searchqa.py` | SearchQA 数据物化（本机用 fastparquet 等价实现） |
| `skillopt/engine/trainer.py` | 通用 ReflACTTrainer 训练循环 |
| `skillopt/envs/base.py` | `EnvAdapter` 抽象接口（含默认 reflect） |
| `skillopt/envs/searchqa/adapter.py` | SearchQA 适配器 |
| `skillopt/envs/searchqa/rollout.py` | 执行层（run_batch / process_one） |
| `skillopt/envs/searchqa/evaluator.py` | 评估层（EM / F1 / sub_em） |
| `skillopt/envs/searchqa/reflect.py` | 空壳（仅 docstring） |
| `skillopt/gradient/reflect.py` | 通用 minibatch 反思引擎 |
| `skillopt/datasets/base.py` | `SplitDataLoader` / split 别名映射 |
| `skillopt/model/openai_compatible_backend.py` | DeepSeek 等 OpenAI 兼容后端 |
| `skillopt/model/backend_config.py` | `is_target_exec_backend()` 判定 |
| `skillopt/evaluation/gate.py` | gate / `select_gate_score` |
| `skillopt/utils/scoring.py` | `compute_score`（hard/soft 聚合） |
| `skillopt/optimizer/clip.py` | `rank_and_select`（SELECT 阶段） |
| `skillopt/optimizer/slow_update.py` | epoch 末慢更新 |
| `skillopt/optimizer/meta_skill.py` | epoch 末元技能 |
| `configs/searchqa/default.yaml` | SearchQA 配置 |
| `configs/_base_/default.yaml` | 通用默认配置 |
| `data/searchqa_id_split/` | 官方 ID manifest（400/200/1400） |
| `data/searchqa_split/` | 物化后的可运行数据 |

## 附录 B：常用命令速查

```bash
# 训练（1 epoch，DeepSeek）
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4-flash \
    model.target=deepseek-v4-flash \
    train.num_epochs=1 \
  --out_root outputs/searchqa_quickstart

# 评估最佳技能（valid_unseen = test）
python scripts/eval_only.py \
  --config configs/searchqa/default.yaml \
  --cfg-options \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4-flash \
    model.target=deepseek-v4-flash \
  --skill outputs/searchqa_quickstart/best_skill.md \
  --split valid_unseen

# 跑测试（禁用第三方插件避免干扰）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_codex_config_aliases.py -q
```

## 附录 C：本次修改的仓库文件

- `skillopt/config.py`：修复 env 段被删的回归（扁平键 `env` 与 section 同名冲突），跳过对 dict 型 section 的删除
- `tests/test_codex_config_aliases.py`：新增 `test_structured_env_section_survives_base_inheritance` 回归测试
- `.env`（gitignored）：写入 OPENAI_COMPATIBLE_*（自包含凭据）
- `data/searchqa_split/`：物化数据（fastparquet 生成）
- `outputs/searchqa_quickstart/`：训练产物
