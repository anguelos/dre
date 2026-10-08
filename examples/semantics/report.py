#!/usr/bin/env python3
# Report: all evaluations → one accuracy table (percent), methods as rows, dataset × version as columns, with
# RE2RNN's published zero-shot numbers (Table 4) for reference.
import json
from pathlib import Path

import fargv

PUBLISHED = {"RE system (published)": {"ATIS": 87.01, "TREC": 64.40, "SMS": 93.20},
             "FA-RNN (published)": {"ATIS": 86.53, "TREC": 61.95, "SMS": 93.00},
             "BiFA-GRU (published)": {"ATIS": 88.63, "TREC": 62.90, "SMS": 93.20}}


def main() -> None:
    p, _ = fargv.parse({"data_root": "data", "datasets": "ATIS TREC SMS", "versions": "clean noise5 noise20"})
    datasets, versions = p.datasets.split(), p.versions.split()
    columns = [(d, v) for d in datasets for v in versions]
    print("| method | " + " | ".join(f"{d} {v}" for d, v in columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    for name, numbers in PUBLISHED.items():
        print(f"| {name} | " + " | ".join(f"{numbers[d]:.2f}" if v == "clean" else "" for d, v in columns) + " |")
    for method_dir in sorted((Path(p.data_root) / "eval").glob("*")):
        cells = []
        for d, v in columns:
            path = method_dir / f"{d}.{v}.json"
            cells.append(f"{100 * json.loads(path.read_text())['accuracy']:.2f}" if path.exists() else "")
        print(f"| {method_dir.name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
