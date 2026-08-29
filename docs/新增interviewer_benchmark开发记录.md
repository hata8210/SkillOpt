# Interviewer Benchmark 开发记录

> 按 `docs/新增interviewer_benchmark.md` 实现，本文档记录实现过程中**实际输出或改动的程序脚本文件与内容**。
> 环境：本仓库（microsoft/SkillOpt），Python 3.12；本地验证通过（未调用真实 LLM）。

---

## 一、新增文件

| 文件 | 内容说明 |
|---|---|
| `scripts/materialize_interviewer.py` | 物化脚本：源 CSV 路径由必填参数 `--csv` 从外部传入（如 `data/interviewer_data/data.csv`，34 条）→ 生成 `data/interviewer_split/{train,val,test}/items.json` + `split_manifest.json`（manifest 的 `source_csv` 记录实际传入的 CSV 相对路径）。默认 `--split-method full`（三个 split 全量 34 条），支持 `stratified`（按 result 分层 20/7/7，seed=42）。核心函数：`load_items` / `_normalize_row` / `_build_question` / `_stratified_split` / `write_split` |
| `skillopt/envs/interviewer/__init__.py` | 空包声明 |
| `skillopt/envs/interviewer/dataloader.py` | `InterviewerDataLoader(SplitDataLoader)`，只实现 `load_split_items()` 读 split 目录下第一个 `.json` |
| `skillopt/envs/interviewer/evaluator.py` | 打分核心：`parse_score`（解析 `<score>`）、`score_to_zone`（≤4 reject / (4,6.5) middle / ≥6.5 hire）、`gold_to_zone`（通过→hire、不通過→reject）、`score_episode`（返回 `(hard, soft)`，soft==hard：命中 1.0、中间区 0.3、反区/解析失败 0.0，**无分数平滑**） |
| `skillopt/envs/interviewer/rollout.py` | `_rollout_one`：`system=skill_content` + `user=question` → `chat_target` → `score_episode` → 写 `predictions/<id>/conversation.json`。`run_batch`：ThreadPoolExecutor 并发、断点续跑（`results.jsonl`）、异常兜底为 hard=0 并记 `fail_reason` |
| `skillopt/envs/interviewer/adapter.py` | `InterviewerAdapter(EnvAdapter)`：实现 `setup` / `get_dataloader` / `build_train_env` / `build_eval_env` / `rollout` / `get_task_types`；`reflect` 继承基类默认实现。`__init__` 形参与配置扁平键对齐（split_dir/workers/minibatch_size/edit_budget 等） |
| `skillopt/envs/interviewer/skills/initial.md` | 初始技能：从 `SKILL_template.md` 精确提取评分表生成。结构 = `SKILL_FIXED`（岗位要求+使用说明）→ `TABLE_HEADER`（表头 5 列）→ 8 条规则行（可编辑区）→ `SKILL_FIXED_TAIL`（总分三区间） |
| `skillopt/envs/interviewer/prompts/analyst_error.md` | 失败分析师 prompt：诊断评分细则缺陷，编辑只允许落在规则行；固定区（三组 tag）不改、列结构/三区间不改、新增行用 `insert_after`、target 用整行唯一文本；JSON 输出结构与通用版一致 |
| `skillopt/envs/interviewer/prompts/analyst_success.md` | 成功分析师 prompt：与 error 版同约束；额外说明 `hard=0.3` 中间区是**部分正确而非成功**，禁止强化中间分，推动规则让总分落到决定性区间 |
| `configs/interviewer/default.yaml` | 配置：`env.name: interviewer`、`split_mode: split_dir`、`split_dir: data/interviewer_split`、`train_size: 34`、`batch_size: 8`、`minibatch_size: 4`、`learning_rate: 2`、`skill_update_mode: patch`、`use_slow_update: false`、`use_meta_skill: false`、`evaluation.gate_metric: hard` |

## 二、修改文件

| 文件 | 改动内容 |
|---|---|
| `scripts/train.py` | `_register_builtins()` 末尾追加 `try/except ImportError` 注册：`_ENV_REGISTRY["interviewer"] = InterviewerAdapter` |
| `scripts/eval_only.py` | 同上，追加 interviewer 注册 |

## 三、生成数据（gitignored，不入库）

| 路径 | 内容 |
|---|---|
| `data/interviewer_split/{train,val,test}/items.json` | 每个 split 34 条全量样本（id、candidate、age、sex、jd、context、question、answers、ground_truth、task_type、split） |
| `data/interviewer_split/split_manifest.json` | 源 CSV、split 方法、counts（34/34/34）、item 字段清单 |

## 四、实现过程中的修复

- `skillopt/envs/interviewer/rollout.py` `run_batch`：初版并发循环用 `while done:` 在全部 future 完成后仍会进入循环导致 `KeyError`（冒烟测试暴露）；改为标准 `pending_futs -= done` 写法后修复
- `outputs/interviewer_blank2` 训练中 slow update 报 401 `Incorrect API key provided: dummy`：`openai_compatible` 后端在模块导入时读 `OPENAI_COMPATIBLE_*`，进程 env 未加载时 base_url 兜底到 `https://api.openai.com/v1`、api_key 兜底成占位符 `dummy`，optimizer 调用全部失败（slow update 被吞掉并提示 `no guidance produced`）。修复：运行前 `set -a; source .env; set +a`（`.env` 里已写 DeepSeek 的 `OPENAI_COMPATIBLE_*`）

## 五、本地验证结果（均通过）

- 物化：full → 34/34/34（通过24/不通過10）；stratified → 20/7/7（14/6、5/2、5/2）
- 配置加载 + `get_adapter`（train/eval_only 两个注册表均能实例化 `InterviewerAdapter`），dataloader 加载 34/34/34
- evaluator 断言矩阵：命中 1.0 / 中间区 0.3 / 反区 0 / 解析失败 0，区间边界（4、6.5）与 gold 别名映射全部通过
- rollout 冒烟（mock `chat_target`）：4 条并发、`conversation.json` 结构正确、`results.jsonl` 断点续跑不重复调用、模型异常兜底 hard=0 + `fail_reason`
- 新增 Python 文件 `py_compile` 全部通过

## 六、待真实运行验证（需 LLM/网络）

- `python scripts/train.py --config configs/interviewer/default.yaml`（+ DeepSeek `--cfg-options`）
- `python scripts/eval_only.py --skill outputs/interviewer_quickstart/best_skill.md --split valid_unseen`
- 检查反思分组、`history.json` 中 edits 的 target 分布（应集中在规则行）
