from __future__ import annotations

import json
from pathlib import Path

from benchmarks.common import asset_data_path, benchmark_data_path, load_tab_samples


def load_reject_test_rows(max_samples: int | None = None) -> list[tuple[str, int]]:
    path = asset_data_path("train", "data", "reject", "test.txt")
    rows: list[tuple[str, int]] = []
    for text, label in load_tab_samples(path):
        rows.append((text, int(label)))
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    return rows


def default_multiturn_path() -> Path:
    return benchmark_data_path("bert_value_multiturn.jsonl")


def load_multiturn_sessions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    sessions: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sessions.append(json.loads(line))
    return sessions
