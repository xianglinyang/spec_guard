from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from search_based_attack.schemas import CandidateRecord, IterationState


class RunLogger:
    def __init__(self, base_dir: str, exp_name: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / f"{exp_name}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = self.run_dir / "candidates.jsonl"
        self.iterations_path = self.run_dir / "iterations.jsonl"
        self.summary_path = self.run_dir / "summary.json"

    def log_candidate(self, rec: CandidateRecord) -> None:
        with self.candidates_path.open("a") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def log_iteration(self, state: IterationState) -> None:
        with self.iterations_path.open("a") as f:
            f.write(json.dumps(asdict(state), ensure_ascii=False) + "\n")

    def write_summary(self, payload: dict) -> None:
        with self.summary_path.open("w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
