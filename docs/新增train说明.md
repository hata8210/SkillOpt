# SkillOpt 新增训练数据集说明

> 本文档基于本仓库（microsoft/SkillOpt）SearchQA quickstart 实战整理，包含三部分：
> ① 数据集结构说明；② 脚本程序关联与分工；③ 如何新增一个训练数据集。

---

## 一、SearchQA 数据集结构

### 1.1 来源与规模

- 来源：Jeopardy!（危险边缘）题库问题 + 基于问题的 Google 搜索检索文档，经 MRQA 2019 格式化整理
- 原始 HF 数据集（`lucadiliello/searchqa`）：
  - train 117,384 条、validation 16,980 条
  - 数据文件为 2 个 parquet（train 约 284MB、validation 约 41MB）
- 本地物化后（`data/searchqa_split/`）：train 400 / val 200 / test 1400，由官方 ID manifest（`data/searchqa_id_split/`）抽取生成

### 1.2 原始 parquet 列结构（5 列）

| 列名 | 类型 | 说明 |
|---|---|---|
| `key` | `string` | 样本唯一 ID，32 位十六进制（类似 MD5），manifest 用它定位样本 |
| `question` | `string` | Jeopardy 风格的问题/线索 |
| `context` | `string` | 多个搜索结果文档拼接的长文本，带结构化标记 |
| `answers` | `sequence<string>` | 标准答案列表（本数据每行一般只有 1 个答案） |
| `labels` | `list<{end: list<int64>, start: list<int64>}>` | 答案在 context 中的字符 span；本版本 117,384 行全部为 null，未使用 |

### 1.3 context 的结构化标记

`context` 把多个搜索文档拼在一起，用三种标签分隔：

- `[DOC]` — 一个新文档的开始
- `[TLE]` — 文档标题（title）
- `[PAR]` — 段落（paragraph）

示例（截断）：

```
[DOC] [TLE] KING OF THE 'TECHNO-THRILLER' - The New York Times [PAR] May 1, 1988 ...
[DOC] [TLE] Jeopardy! #1 Flashcards | Quizlet [ ...
```

### 1.4 物化后的本地数据（items.json）

`data/searchqa_split/{train,val,test}/items.json` 是 JSON 数组，每条 4 个字段：`id`、`question`、`context`、`answers`。

```json
{
  "id": "221c83e6630f4e7983da48fa28da1882",
  "question": "The New York Times Magazine has called him \"King of the Techno-Thriller\"",
  "context": "[DOC] [TLE] KING OF THE 'TECHNO-THRILLER' - The New York Times [PAR] ...",
  "answers": ["Tom Clancy"]
}
```

注意：`answers` 虽然 parquet schema 显示为嵌套 `LIST<LIST<STRING>>`，实际每行解析出来是单元素字符串列表（如 `["Tom Clancy"]`）。

### 1.5 demo 中数据如何被使用

- 输入构造（`skillopt/envs/searchqa/rollout.py`）：skill 文档 + `question` + 截断到 6000 字符的 `context` 拼进 prompt，模型输出 `<answer>...</answer>` 标签；`_truncate_context` 按 `[DOC]` 边界截断
- 评估（`skillopt/envs/searchqa/evaluator.py`）：提取 `<answer>`（无标签取最后非空行）→ SQuAD 规范化（小写、去标点、去 a/an/the、压缩空白）→ 指标：
  - `em`（Exact Match）→ 训练日志里的 `hard`，用于 gate/选优
  - `f1`（token 级 F1，取所有 gold 答案最大值）→ `soft`
  - `sub_em`（子串匹配）

### 1.6 split（数据切分）

`split` 是把数据集按样本 ID 划分成互不相交的子集，各司其职：

| split | 数量 | 别名 | 用途 |
|---|---|---|---|
| `train` | 400 | `train` | 训练循环里每步取 40 条做 rollout → 反思 → 生成编辑 → 更新技能 |
| `val` | 200 | `valid_seen` / `selection` | 训练过程中评估：baseline、每步 200 条 selection、gate 是否接受新技能 |
| `test` | 1400 | `valid_unseen` | 最终评估，训练中完全不参与，衡量泛化 |

别名映射见 `skillopt/datasets/base.py`：`valid_seen → val`、`valid_unseen → test`。

配置（`configs/searchqa/default.yaml`）：

```yaml
env:
  split_mode: split_dir   # split_dir=用预切分目录；ratio=按 split_ratio 随机切
  split_dir: data/searchqa_split
```

---

## 二、脚本程序关联与分工

### 2.1 总体调用链

```
scripts/train.py（训练入口）
 └─ 构建 SearchQAAdapter + ReflACTTrainer（skillopt/engine/trainer.py）
     └─ trainer.train() 里所有"跑一批样本"的地方都调 adapter.rollout(...)
         └─ skillopt/envs/searchqa/adapter.py 的 rollout()
             └─ skillopt/envs/searchqa/rollout.py 的 run_batch() → process_one()
                 ├─ 拼 prompt、调 LLM（chat_target 或 run_target_exec）
                 └─ 调 skillopt/envs/searchqa/evaluator.py 的 evaluate() 打分
```

### 2.2 各文件职责

**`scripts/train.py` — 编排层**
- 解析 CLI、加载配置、通过环境注册表找到 adapter、启动 `ReflACTTrainer`
- 只关心训练流程（baseline → step → 每步 6 阶段 → 最终 test 评估），不关心"怎么答一道题"

**`skillopt/engine/trainer.py` — 通用训练器（ReflACTTrainer）**
- 与环境无关，只依赖 `EnvAdapter` 抽象接口
- 负责训练循环、gate、selection、慢更新、meta-skill 等通用逻辑

**`skillopt/envs/searchqa/adapter.py` — 环境适配器**
- 继承 `EnvAdapter`，实现 `__init__` / `build_train_env` / `build_eval_env` / `rollout` / `get_task_types`
- `rollout()`（约 76 行）只做转发，把参数传给 `run_batch`

**`skillopt/envs/searchqa/rollout.py` — 执行层**
- `run_batch()`：多线程并发跑一批样本（`workers=24`），支持断点续跑（从 `results.jsonl` 恢复）
- `process_one()`：处理单条样本，两种模式见 2.3

**`skillopt/envs/searchqa/evaluator.py` — 评估层**
- 纯字符串处理，不调用 LLM：提取答案 → 规范化 → em/f1/sub_em
- `rollout.py` 里 `result["hard"] = int(eval_result["em"])`、`result["soft"] = eval_result["f1"]`

**`skillopt/datasets/base.py` — 通用数据加载**
- `SplitDataLoader` / `BatchSpec`，`split_mode`（split_dir / ratio）逻辑在这里

### 2.3 rollout 的两种模式

`process_one()` 里由 `is_target_exec_backend()`（`skillopt/model/backend_config.py`）判定：

```python
if is_target_exec_backend():
    # 模式二：agent 执行（Codex / Claude Code / Cursor / Copilot）
    response, raw, system, user = _run_codex_once(...)
else:
    # 模式一：直接 prompt 推理（chat_target 一次调用）
    resp_text, _ = chat_target(system=system, user=user, ...)
```

- **模式一：prompt 直接推理**：`_build_system(skill)` + `_build_user(question, context)` → `chat_target()` 一次调用 → 拿 `<answer>` 打分。单次对话、无工具、无代码执行。本 demo 使用
- **模式二：agent 执行**：`_run_codex_once()`（`rollout.py:101`）→ `prepare_workspace()` 建工作区（技能写成 `skillopt-target`，题目写成 `task.md`）→ `run_target_exec()` 启动 agent 读任务、按技能执行，返回 `<answer>`。适合需要检索、计算、跑代码的任务

切换方式：`model.target_backend` 为 `codex_exec / claude_code_exec / cursor_exec / copilot_exec` 时是 agent 模式；为 `openai_compatible / openai_chat / qwen_chat / minimax_chat` 等时是直接推理。命令行用 `--cfg-options model.target_backend=...`。

本 demo 实际配置：

> **运行前必须先加载环境变量**：`openai_compatible` 后端在模块导入时读取 `OPENAI_COMPATIBLE_*`，务必先 `set -a; source .env; set +a` 再启动训练/评估。漏掉的话 API key 会兜底成占位符 `dummy`、base_url 落到 `https://api.openai.com/v1`，optimizer 调用（含每个 epoch 末的 slow update）会全部 401：`Incorrect API key provided: dummy`，slow update 阶段显示 `[slow update] no guidance produced`。

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4-flash \
    model.target=deepseek-v4-flash \
    train.num_epochs=1 \
  --out_root outputs/searchqa_quickstart
```

---

## 三、新增一个训练数据集

### 3.1 核心结论

`ReflACTTrainer` 和整个优化管线（反思/聚合/选择/更新、模型后端、配置加载）都是**通用**的；
需要新写的只有"数据加载 → 任务执行 → 打分"这层薄适配器，以及注册和配置。

### 3.2 必须新写的文件

| 文件 | 内容 |
|---|---|
| `skillopt/envs/<name>/adapter.py` | 继承 `EnvAdapter`，实现 `__init__`、`build_train_env`、`build_eval_env`、`rollout`、`get_task_types`（`reflect` 用基类默认即可） |
| `skillopt/envs/<name>/dataloader.py` | 继承 `SplitDataLoader`，实现读数据、产出带 `id` 的 items |
| `skillopt/envs/<name>/rollout.py` | 任务执行：怎么调模型/agent、产出结果（必须带 `id`、`hard` 0/1、`soft` 0~1） |
| `skillopt/envs/<name>/evaluator.py` | 打分逻辑（可仿 SearchQA 的 EM/F1，也可自定义） |
| `skillopt/envs/<name>/prompts/` | 可选：环境专属 system/analyst prompt；不写则用 `skillopt/prompts/` 通用版 |
| `skillopt/envs/<name>/skills/initial.md` | 初始技能文档 |
| `configs/<name>/default.yaml` | 配置文件（`env.name` 填环境名） |
| 修改 `scripts/train.py`、`scripts/eval_only.py` | 各加一行 `_ENV_REGISTRY["<name>"] = YourAdapter` |
| 可选 | 数据物化脚本 + ID manifest（参考 `scripts/materialize_searchqa.py`） |

### 3.3 完全不用动的通用文件

- `skillopt/engine/trainer.py`（ReflACTTrainer）
- `skillopt/envs/base.py`（EnvAdapter 基类，含默认 `reflect`）
- `skillopt/datasets/base.py`（SplitDataLoader / BatchSpec）
- `skillopt/gradient/*`（反思 / 聚合 / 选择 / 更新）
- `skillopt/model/*`（openai_compatible 等模型后端）
- `skillopt/config.py`（配置加载与扁平化）
- `skillopt/prompts/*`（通用 prompt 兜底）

### 3.4 接口契约（最容易踩坑）

- item 必须有 `"id"`（str）
- `rollout()` 返回的每条结果必须有 `"id"`、`"hard"`（0/1）、`"soft"`（0~1 float）——trainer 的 gate/selection 全看这两个字段
- adapter `__init__` 的形参名要和配置扁平化后的键对上（如 `split_dir`、`workers`），因为 `get_adapter` 是按签名自动注入的
- 环境注册：`_register_builtins()` 里 `_ENV_REGISTRY["<name>"] = YourAdapter`（train 和 eval 两份）
- 配置选择：`env.name` 必须等于注册的 `<name>`

### 3.5 新增步骤速览

1. 准备数据：整理成 train/val/test 三份，每份是含 `id` 及任务字段的 JSON/JSONL（参考 `data/searchqa_split/`）
2. 写 `skillopt/envs/<name>/` 下 adapter、dataloader、rollout、evaluator、skills/initial.md
3. 写 `configs/<name>/default.yaml`（继承 `configs/_base_/default.yaml`，改 `env.name`、`split_dir` 等）
4. 在 `scripts/train.py` 和 `scripts/eval_only.py` 的 `_register_builtins()` 各注册一行
5. 运行：

```bash
python scripts/train.py \
  --config configs/<name>/default.yaml \
  --cfg-options \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4-flash \
    model.target=deepseek-v4-flash \
  --out_root outputs/<name>_quickstart

python scripts/eval_only.py \
  --config configs/<name>/default.yaml \
  --cfg-options \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4-flash \
    model.target=deepseek-v4-flash \
  --skill outputs/<name>_quickstart/best_skill.md \
  --split valid_unseen
```

---

## 附录：本次 quickstart 踩过的坑（环境相关）

- **pyarrow SIGILL**：VM CPU 缺 AVX/BMI2 指令，pyarrow 读 parquet 嵌套列（`answers`）崩溃；用 `fastparquet` 读取生成同格式 split 可绕过
- **DEEPSEEK 环境变量丢失**：父 shell 变量会话中途消失导致 401；把凭据写进 `.env`（自包含，已 gitignore）解决
- **忘记 source `.env` 导致 401**：`openai_compatible` 后端 import 时读取 env，key 缺失兜底为 `dummy` 并向默认 OpenAI 端点发请求（slow update/反思报 `Incorrect API key provided: dummy`）；每次运行前 `set -a; source .env; set +a`
- **`env` 配置段被删**：`skillopt/config.py` 的扁平键 `env` 与 section `env` 同名冲突（17b4823 引入的回归），已修复并加回归测试
