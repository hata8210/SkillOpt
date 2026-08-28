from __future__ import annotations

import glob
import json
import os

from skillopt.datasets.base import SplitDataLoader


class InterviewerDataLoader(SplitDataLoader):
    """Load interviewer items from ``items.json`` inside each split dir."""

    def load_split_items(self, split_path: str) -> list[dict]:
        json_files = sorted(glob.glob(os.path.join(split_path, "*.json")))
        if not json_files:
            raise FileNotFoundError(f"No .json file found in {split_path}")
        with open(json_files[0], encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError(
                f"Expected JSON array in {json_files[0]}, got {type(items).__name__}"
            )
        return items
