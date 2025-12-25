from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List

import pandas as pd
import yaml


class MetricsWriter:
    """CSV writer for step-level metrics."""
    def __init__(self, file_path: str, fieldnames: Iterable[str]) -> None:
        self.file_path = file_path
        self.fieldnames = list(fieldnames)
        self._file = open(file_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def write(self, row: Dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def save_config(config: Dict, path: str) -> None:
    """Save run configuration as YAML."""
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=True)


def merge_rank_metrics(run_dir: str, world_size: int) -> str:
    """Merge per-rank CSVs into a single merged_metrics.csv."""
    metrics_dir = os.path.join(run_dir, "metrics")
    files = [os.path.join(metrics_dir, f"rank{r}.csv") for r in range(world_size)]
    frames: List[pd.DataFrame] = []
    for path in files:
        frames.append(pd.read_csv(path))
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["step", "rank"]).reset_index(drop=True)
    out_path = os.path.join(run_dir, "merged_metrics.csv")
    merged.to_csv(out_path, index=False)
    return out_path
