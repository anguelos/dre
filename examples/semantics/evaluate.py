#!/usr/bin/env python3
# Evaluate: predictions (gold<TAB>predicted per line) → accuracy, plus spam precision/recall for SMS, as JSON.
import json

import fargv


def main() -> None:
    p, _ = fargv.parse({"predictions": "", "output": "", "positive": "spam"})
    pairs = [line.rstrip("\n").split("\t") for line in open(p.predictions) if line.strip()]
    result = {"n": len(pairs), "accuracy": sum(g == q for g, q in pairs) / len(pairs)}
    if any(g == p.positive for g, _ in pairs):
        true_positive = sum(g == q == p.positive for g, q in pairs)
        predicted, actual = sum(q == p.positive for _, q in pairs), sum(g == p.positive for g, _ in pairs)
        result.update(precision=true_positive / predicted if predicted else 0.0, recall=true_positive / actual)
    with open(p.output, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
