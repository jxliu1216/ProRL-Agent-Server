#!/usr/bin/env python3
"""Prepare a 64-task SWE-Gym train JSONL dataset filtered to specific instance IDs.

Writes ``swegym_train_64.jsonl`` — a drop-in replacement for the full 293-task
``swegym_train_293.jsonl`` that only contains the 64 tasks whose Apptainer images
we have pre-loaded as local Docker images.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_tasks import fetch_dataset_instances

SPLIT = "train"
OUTPUT = Path(__file__).resolve().parent / "swegym_train_64.jsonl"

# ---- The 64 instance IDs matching the locally-loaded Docker images ----
INSTANCE_IDS_64 = frozenset([
    # conan-io (3 tasks)
    "conan-io__conan-13403",
    "conan-io__conan-15422",
    "conan-io__conan-15699",
    # dask (6 tasks)
    "dask__dask-10972",
    "dask__dask-7191",
    "dask__dask-8686",
    "dask__dask-9213",
    "dask__dask-9378",
    "dask__dask-9627",
    # getmoto/moto (15 tasks)
    "getmoto__moto-4950",
    "getmoto__moto-4986",
    "getmoto__moto-5020",
    "getmoto__moto-5134",
    "getmoto__moto-5582",
    "getmoto__moto-5587",
    "getmoto__moto-5865",
    "getmoto__moto-5959",
    "getmoto__moto-6114",
    "getmoto__moto-6178",
    "getmoto__moto-6299",
    "getmoto__moto-6469",
    "getmoto__moto-7111",
    "getmoto__moto-7167",
    "getmoto__moto-7212",
    # iterative/dvc (5 tasks)
    "iterative__dvc-1651",
    "iterative__dvc-2231",
    "iterative__dvc-4778",
    "iterative__dvc-4785",
    "iterative__dvc-5839",
    # pandas-dev (13 tasks)
    "pandas-dev__pandas-49118",
    "pandas-dev__pandas-49766",
    "pandas-dev__pandas-50713",
    "pandas-dev__pandas-51605",
    "pandas-dev__pandas-51936",
    "pandas-dev__pandas-52076",
    "pandas-dev__pandas-52077",
    "pandas-dev__pandas-52516",
    "pandas-dev__pandas-53856",
    "pandas-dev__pandas-57058",
    "pandas-dev__pandas-57089",
    "pandas-dev__pandas-57173",
    "pandas-dev__pandas-57957",
    # Project-MONAI (12 tasks)
    "Project-MONAI__MONAI-1571",
    "Project-MONAI__MONAI-2238",
    "Project-MONAI__MONAI-2696",
    "Project-MONAI__MONAI-3403",
    "Project-MONAI__MONAI-4109",
    "Project-MONAI__MONAI-4583",
    "Project-MONAI__MONAI-5183",
    "Project-MONAI__MONAI-5423",
    "Project-MONAI__MONAI-5543",
    "Project-MONAI__MONAI-5640",
    "Project-MONAI__MONAI-6560",
    "Project-MONAI__MONAI-6756",
    # pydantic (4 tasks)
    "pydantic__pydantic-8004",
    "pydantic__pydantic-8072",
    "pydantic__pydantic-8316",
    "pydantic__pydantic-8511",
    # python/mypy (6 tasks)
    "python__mypy-11135",
    "python__mypy-11567",
    "python__mypy-12943",
    "python__mypy-15184",
    "python__mypy-16869",
    "python__mypy-5617",
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
        if str(inst["instance_id"]) in INSTANCE_IDS_64
    ]

    if len(selected) != len(INSTANCE_IDS_64):
        found = {str(inst["instance_id"]) for inst in selected}
        missing = INSTANCE_IDS_64 - found
        raise SystemExit(
            f"Found {len(selected)}/{len(INSTANCE_IDS_64)} instances in dataset. "
            f"Missing: {', '.join(sorted(missing))}"
        )

    rows = [row_for_instance(inst, SPLIT) for inst in selected]
    OUTPUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n"
    )
    print(f"Wrote {len(rows)} tasks to {OUTPUT}")


if __name__ == "__main__":
    main()
