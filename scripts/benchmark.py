from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from background_app.config import AppSettings
from background_app.data import DatasetData
from background_app.processing import process_dataset


def make_dataset(curves: int, points: int) -> DatasetData:
    x = np.linspace(2.0, 25.0, points)
    background = 50.0 + 0.2 * (x - 12.0) ** 2
    rows = []
    for index in range(curves):
        shift = 0.04 * np.sin(index / 7.0)
        peak = (80.0 + index % 13) * np.exp(-((x - (9.0 + shift)) / 0.3) ** 2)
        rows.append(background + peak)
    return DatasetData(x, np.vstack(rows), [f"curve-{index + 1}" for index in range(curves)], [Path("synthetic.xy")])


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the background-correction processing core")
    parser.add_argument("--curves", type=int, default=100)
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument("--workers", type=int, default=0, help="0 selects adaptive automatic mode")
    args = parser.parse_args()
    settings = AppSettings(); settings.runtime.workers = args.workers; settings.data.range_enabled = False
    dataset = make_dataset(args.curves, args.points)
    started = time.perf_counter(); result = process_dataset(dataset, settings); elapsed = time.perf_counter() - started
    print(f"curves={args.curves} points={args.points} workers={result.workers_used} elapsed={elapsed:.3f}s curves/s={args.curves / elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
