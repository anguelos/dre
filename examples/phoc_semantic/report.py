#!/usr/bin/env python3
# Report: every trained variant (at its best epoch by unseen-word loss, and at its last epoch) against the lookup
# baselines, as a markdown table of top-1 / top-5 nearest-neighbour accuracy on each evaluation set.
import glob
import json
from pathlib import Path

import fargv
import torch

SETS = ("unseen_clean", "unseen_noisy", "known_noisy")


def describe(config: dict) -> str:
    return f"hidden {config['hidden']}, dropout {config['dropout']}, {config.get('loss', 'cosine')} loss"


def main() -> None:
    p, _ = fargv.parse({"states": "data/models/*.pt", "baselines": "data/results"})
    rows = []
    for path in sorted(Path(p.baselines).glob("baseline_*.json")):
        result = json.loads(path.read_text())
        rows.append((f"lookup by {result['distance']}", [(result[s]["top1"], result[s]["top5"]) for s in SETS]))
    for path in sorted(glob.glob(p.states)):
        state = torch.load(path, map_location="cpu", weights_only=False)  # our own file
        for label, epoch in [("best unseen loss", state["best_epoch"]), ("last", state["epoch"])]:
            record = next(h for h in state["history"] if h["epoch"] == epoch)
            rows.append((f"{Path(path).stem.split('.')[-1]} ({describe(state['config'])}), epoch {epoch} ({label})",
                         [(record[f"{s}_top1"], record[f"{s}_top5"]) for s in SETS]))
    print("| method | " + " | ".join(f"{s} top1 / top5" for s in SETS) + " |")
    print("|---" * (len(SETS) + 1) + "|")
    for name, cells in rows:
        print(f"| {name} | " + " | ".join(f"{100 * a:.1f} / {100 * b:.1f}" for a, b in cells) + " |")


if __name__ == "__main__":
    main()
