#!/usr/bin/env python3
"""Prepare a 16-task SWE-Gym train JSONL dataset filtered to specific instance IDs.

Writes ``swegym_train_16.jsonl`` — a drop-in replacement for the full 293-task
``swegym_train_293.jsonl`` that only contains the 16 tasks whose Apptainer images
we have pre-loaded as local Docker images.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_tasks import fetch_dataset_instances

SPLIT = "train"
OUTPUT = Path(__file__).resolve().parent / "swegym_train_16.jsonl"

# ---- The 16 instance IDs matching the locally-loaded Docker images ----
INSTANCE_IDS_16 = frozenset([
    # pandas-dev (8 tasks)
    "pandas-dev__pandas-49118",
    "pandas-dev__pandas-49766",
    "pandas-dev__pandas-50713",
    "pandas-dev__pandas-51605",
    "pandas-dev__pandas-51936",
    "pandas-dev__pandas-52076",
    "pandas-dev__pandas-52077",
    "pandas-dev__pandas-52516",
    # getmoto/moto (8 tasks)
    "getmoto__moto-4950",
    "getmoto__moto-4986",
    "getmoto__moto-5020",
    "getmoto__moto-5134",
    "getmoto__moto-5582",
    "getmoto__moto-5587",
    "getmoto__moto-5865",
    "getmoto__moto-5959",
])


def row_for_instance(instance: dict, split: str) -> dict:
    instance_id = str(instance["instance_id"])
    return {
        "prompt": [
            {"role": "user", "content": str(instance["problem_statement"]).strip()}
        ],
        "label": "",
        "metadata": {
            "instance_id": instance_id,
            "instance": instance,
            "split": split,
        },
    }


def main() -> None:
    all_instances = fetch_dataset_instances(SPLIT, refresh=False)
    selected = [
        inst for inst in all_instances
        if str(inst["instance_id"]) in INSTANCE_IDS_16
    ]

    if len(selected) != len(INSTANCE_IDS_16):
        found = {str(inst["instance_id"]) for inst in selected}
        missing = INSTANCE_IDS_16 - found
        raise SystemExit(
            f"Found {len(selected)}/{len(INSTANCE_IDS_16)} instances in dataset. "
            f"Missing: {', '.join(sorted(missing))}"
        )

    rows = [row_for_instance(inst, SPLIT) for inst in selected]
    OUTPUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n"
    )
    print(f"Wrote {len(rows)} tasks to {OUTPUT}")


if __name__ == "__main__":
    main()
