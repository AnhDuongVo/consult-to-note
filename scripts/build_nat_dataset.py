"""Convert an ACI-Bench split into the JSON dataset format used by `nat eval`.

Each item: {"id": encounter_id, "question": dialogue, "answer": reference note}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from consult_to_note.evaluation import load_aci

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--data-dir", default="data/aci-bench")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = load_aci(args.split, args.data_dir)[: args.n]
    items = [{"id": r["encounter_id"], "question": r["dialogue"], "answer": r["note"]} for r in rows]
    out = Path(args.out or f"data/nat_aci_{args.split}.json")
    out.write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"wrote {len(items)} items to {out}")
