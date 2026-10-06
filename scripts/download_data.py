"""Download public evaluation data.

* ACI-Bench (Yim et al., Scientific Data 2023, CC BY 4.0): dialogues + reference notes, ~1 MB of CSV.
* PriMock57 (Papadopoulos Korfiatis et al., 2022, CC BY 4.0): 57 mock consultations with separate doctor
  and patient audio. Large (Git LFS), so it is only cloned with --primock.

Usage:
    python scripts/download_data.py            # ACI-Bench only
    python scripts/download_data.py --primock  # also clone PriMock57 (needs git and git-lfs)
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import httpx

ACI_BASE = "https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data/"
ACI_FILES = [
    "train.csv",
    "valid.csv",
    "clinicalnlp_taskB_test1.csv",
    "clinicalnlp_taskC_test2.csv",
    "clef_taskC_test3.csv",
]


def download_aci(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for name in ACI_FILES:
            dest = target / name
            if dest.exists():
                print(f"exists  {dest}")
                continue
            resp = client.get(ACI_BASE + name)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"saved   {dest} ({len(resp.content) // 1024} KB)")


def clone_primock(target: Path) -> None:
    if target.exists():
        print(f"exists  {target}")
        return
    subprocess.run(["git", "lfs", "install"], check=True)
    subprocess.run(["git", "clone", "https://github.com/babylonhealth/primock57.git", str(target)], check=True)
    print("PriMock57 audio: audio/dayN_consultationNN_{doctor,patient}.wav")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--primock", action="store_true")
    args = ap.parse_args()
    download_aci(args.data_dir / "aci-bench")
    if args.primock:
        clone_primock(args.data_dir / "primock57")
