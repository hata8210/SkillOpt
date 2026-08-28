"""Rollout for the interviewer hire-decision benchmark.

For each item: ``system = skill_content``, ``user = item["question"]`` (which
contains the candidate profile + interview transcript + output instruction).
The target model replies with a total score ``<score>X.X</score>`` which the
evaluator converts into the unified ``hard`` reward. Per-item trajectories are
persisted under ``out_root/predictions/<id>/conversation.json`` so the shared
reflection stage can consume them.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from skillopt.model import chat_target
from skillopt.envs.interviewer.evaluator import score_episode


def _rollout_one(
    item: dict,
    skill_content: str,
    *,
    prediction_dir: Path,
    max_completion_tokens: int,
) -> dict:
    system = skill_content
    user = str(item.get("question") or "")
    prediction, _usage = chat_target(
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
    )
    hard, soft = score_episode(prediction, item.get("ground_truth", ""))

    task_dir = prediction_dir / str(item["id"])
    task_dir.mkdir(parents=True, exist_ok=True)
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": prediction},
    ]
    (task_dir / "conversation.json").write_text(
        json.dumps(conversation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "id": str(item["id"]),
        "hard": hard,
        "soft": soft,
        "predicted_answer": prediction,
        "task_description": str(item.get("question", "")),
        "question": str(item.get("question", "")),
        "candidate": str(item.get("candidate", "")),
        "context": str(item.get("context", "")),
        "ground_truth": str(item.get("ground_truth", "")),
        "task_type": str(item.get("task_type", "interviewer")),
        "target_system_prompt": system,
        "target_user_prompt": user,
        "n_turns": 1,
    }


def run_batch(
    *,
    items: list[dict],
    skill_content: str,
    out_root: str,
    workers: int = 4,
    max_completion_tokens: int = 4096,
) -> list[dict]:
    """Run a batch of items, resume-aware via ``out_root/results.jsonl``."""
    os.makedirs(out_root, exist_ok=True)
    prediction_dir = Path(out_root, "predictions")
    results_path = Path(out_root, "results.jsonl")

    done_ids: set[str] = set()
    results: list[dict] = []
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(str(r["id"]))
                    results.append(r)
                except json.JSONDecodeError:
                    pass
        if results:
            print(f"    [rollout] resuming: {len(results)}/{len(items)} already done", flush=True)

    pending = [it for it in items if str(it["id"]) not in done_ids]
    if pending:
        with results_path.open("a", encoding="utf-8") as outf:
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
                futures = {
                    ex.submit(
                        _rollout_one,
                        item,
                        skill_content,
                        prediction_dir=prediction_dir,
                        max_completion_tokens=max_completion_tokens,
                    ): item
                    for item in pending
                }
                pending_futs = set(futures)
                while pending_futs:
                    done, _ = wait(pending_futs, timeout=5, return_when=FIRST_COMPLETED)
                    for fut in done:
                        item = futures[fut]
                        try:
                            result = fut.result()
                        except Exception as exc:  # noqa: BLE001
                            result = {
                                "id": str(item["id"]),
                                "hard": 0,
                                "soft": 0.0,
                                "predicted_answer": "",
                                "question": str(item.get("question", "")),
                                "candidate": str(item.get("candidate", "")),
                                "context": str(item.get("context", "")),
                                "ground_truth": str(item.get("ground_truth", "")),
                                "task_type": str(item.get("task_type", "interviewer")),
                                "target_system_prompt": skill_content,
                                "target_user_prompt": str(item.get("question", "")),
                                "n_turns": 0,
                                "fail_reason": f"error: {type(exc).__name__}: {exc}",
                            }
                        results.append(result)
                        outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                        outf.flush()
                    pending_futs -= done

    return results
