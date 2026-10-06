"""Plot a benchmark summary: latency and throughput against concurrency, one line per target."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def plot_summary(summary_json: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads(Path(summary_json).read_text())["rows"]
    out = Path(out_dir or Path(summary_json).parent)
    by_target: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_target[r["target"]].append(r)

    charts = [
        ("ttft_p95_s", "TTFT p95 (s)", "ttft_p95.png"),
        ("e2e_p95_s", "End-to-end latency p95 (s)", "e2e_p95.png"),
        ("output_tok_per_s", "Output tokens per second", "throughput.png"),
    ]
    paths = []
    for key, label, fname in charts:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, items in by_target.items():
            items = sorted(items, key=lambda r: r["concurrency"])
            ax.plot([r["concurrency"] for r in items], [r[key] for r in items], marker="o", label=name)
        ax.set_xlabel("Concurrent requests")
        ax.set_ylabel(label)
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
