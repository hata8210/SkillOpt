"""Materialize the interviewer hire-decision dataset into SkillOpt split dirs.

Reads a CSV supplied via ``--csv`` (e.g. ``data/interviewer_data/data.csv``,
34 valid rows) and writes ``data/interviewer_split/{train,val,test}/items.json``
plus ``split_manifest.json``.

Split strategies
----------------
* ``full`` (default): write the SAME full 34 items into every split dir so
  train / val / test all cover the complete sample distribution.
* ``stratified``: split by ``result`` (通过/不通過) proportionally into
  train / val / test (default 20 / 7 / 7) with a fixed seed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd

DEFAULT_SPLIT_DIR = "data/interviewer_split"
SPLIT_NAMES = ("train", "val", "test")

PASS_LABELS = ("通过", "录用", "hire")
REJECT_LABELS = ("不通過", "不录用", "reject")


def _build_question(candidate: str, age: object, sex: str, context: str) -> str:
    age_text = "" if age is None or (isinstance(age, float) and pd.isna(age)) else f"（{int(age)}岁）"
    return (
        f"候选人：{candidate}（{sex}，{age_text[1:-1] if age_text else ''}）\n\n"
        f"面试记录：\n{context}\n\n"
        "请根据技能中的评分标准逐项打分，合计总分后输出总分，格式：<score>X.X</score>"
    )


def _normalize_row(index: int, row: dict) -> dict:
    result = str(row.get("result") or "").strip()
    candidate = str(row.get("candidate") or "").strip()
    context = str(row.get("context") or "").strip()
    if not candidate or not context or not result:
        raise ValueError(f"row {index}: missing candidate/context/result")

    age = row.get("age")
    if age is None or (isinstance(age, float) and pd.isna(age)):
        age = None
    else:
        age = int(age)

    return {
        "id": f"interviewer_{index:03d}",
        "candidate": candidate,
        "age": age,
        "sex": str(row.get("sex") or "").strip(),
        "jd": str(row.get("jd") or "").strip(),
        "context": context,
        "question": _build_question(candidate, age, str(row.get("sex") or "").strip(), context),
        "answers": [result],
        "ground_truth": result,
        "task_type": "interviewer",
    }


def load_items(csv_path: str) -> list[dict]:
    df = pd.read_csv(csv_path)
    df = df.dropna(how="all")  # drop fully-empty trailing row
    items: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        items.append(_normalize_row(i, row))
    return items


def _stratified_split(items: list[dict], sizes: tuple[int, int, int], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    passes = [it for it in items if it["ground_truth"] in PASS_LABELS]
    rejects = [it for it in items if it["ground_truth"] in REJECT_LABELS]
    rng.shuffle(passes)
    rng.shuffle(rejects)

    train_n, val_n, test_n = sizes
    def _quota(items: list, target: tuple[int, int, int]) -> tuple[int, int, int]:
        total = len(items)
        # proportional allocation across the three splits
        counts = [round(total * n / sum(target)) for n in target]
        # fix rounding drift so counts sum back to total
        diff = total - sum(counts)
        counts[0] += diff
        return tuple(counts)

    p_train, p_val, p_test = _quota(passes, (train_n, val_n, test_n))
    r_train, r_val, r_test = _quota(rejects, (train_n, val_n, test_n))

    return {
        "train": passes[:p_train] + rejects[:r_train],
        "val": passes[p_train:p_train + p_val] + rejects[r_train:r_train + r_val],
        "test": passes[p_train + p_val:] + rejects[r_train + r_val:],
    }


def write_split(csv_path: str, out_dir: str, items: list[dict], method: str, seed: int, sizes: tuple[int, int, int]) -> dict[str, int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if method == "full":
        per_split = {name: list(items) for name in SPLIT_NAMES}
    else:
        per_split = _stratified_split(items, sizes, seed)

    counts: dict[str, int] = {}
    for name in SPLIT_NAMES:
        split_items = [
            {**item, "split": name}
            for item in per_split[name]
        ]
        split_dir = out / name
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "items.json").open("w", encoding="utf-8") as f:
            json.dump(split_items, f, ensure_ascii=False, indent=2)
        counts[name] = len(split_items)

    manifest = {
        "source_csv": os.path.relpath(csv_path, out_dir),
        "split_method": method,
        "split_seed": seed,
        "counts": counts,
        "item_fields": [
            "id", "candidate", "age", "sex", "jd", "context",
            "question", "answers", "ground_truth", "task_type", "split",
        ],
    }
    with (out / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to the interviewer dataset CSV.")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--split-method", choices=("full", "stratified"), default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=20, help="stratified: train size")
    parser.add_argument("--val", type=int, default=7, help="stratified: val size")
    parser.add_argument("--test", type=int, default=7, help="stratified: test size")
    args = parser.parse_args()

    items = load_items(args.csv)
    counts = write_split(args.csv, args.split_dir, items, args.split_method, args.seed, (args.train, args.val, args.test))

    label_counts = {k: sum(1 for it in items if it["ground_truth"] in PASS_LABELS) for k in ("通过",)}
    print(f"loaded {len(items)} items from {args.csv}  (通过={label_counts['通过']} 不通過={len(items) - label_counts['通过']})")
    print(f"split method={args.split_method} → {counts}")
    print(f"output: {args.split_dir}/{{train,val,test}}/items.json + split_manifest.json")


if __name__ == "__main__":
    main()
