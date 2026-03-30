#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize preliminary matrix experiment results.")
    p.add_argument("--run-root", required=True, help="Path like runs_prelim/<run_id>")
    return p.parse_args()


def parse_api_counts(run_log: Path) -> tuple[int, int, int]:
    total = 0
    openrouter = 0
    local = 0
    if not run_log.exists():
        return total, openrouter, local
    for line in run_log.read_text(errors="ignore").splitlines():
        if "HTTP Request: POST " in line and "/chat/completions" in line:
            total += 1
            if "openrouter.ai" in line:
                openrouter += 1
            if "127.0.0.1" in line or "localhost" in line:
                local += 1
    return total, openrouter, local


def parse_gpu_metrics(gpu_log: Path) -> tuple[float | None, float | None, float | None]:
    if not gpu_log.exists():
        return None, None, None
    util_vals: list[float] = []
    mem_vals: list[float] = []
    for line in gpu_log.read_text(errors="ignore").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            util = float(parts[-2])
            mem = float(parts[-1])
        except Exception:
            continue
        util_vals.append(util)
        mem_vals.append(mem)
    if not util_vals:
        return None, None, None
    return mean(util_vals), max(util_vals), max(mem_vals) if mem_vals else None


def parse_task_jsons(logdir: Path) -> list[dict]:
    records: list[dict] = []
    if not logdir.exists():
        return records
    for fp in logdir.rglob("*.json"):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        records.append(data)
    return records


def safe_mean(vals: list[float]) -> float | None:
    return mean(vals) if vals else None


def fmt(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.6f}"


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    manifest = run_root / "manifest.tsv"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    rows: list[dict[str, str]] = []
    with manifest.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(row)

    per_combo: list[dict[str, str]] = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}

    for row in rows:
        seed = row["seed"]
        defense = row["defense"]
        attack_mode = row["attack_mode"]
        status = int(row["status"])
        logdir = Path(row["logdir"])
        run_log = Path(row["run_log"])
        gpu_log = Path(row["gpu_log"])

        task_records = parse_task_jsons(logdir) if status == 0 else []
        utility_vals: list[float] = []
        security_vals: list[float] = []
        duration_vals: list[float] = []

        for rec in task_records:
            utility = rec.get("utility")
            security = rec.get("security")
            duration = rec.get("duration")
            attack_type = rec.get("attack_type")
            inj_id = rec.get("injection_task_id")

            if isinstance(duration, (int, float)):
                duration_vals.append(float(duration))

            if attack_mode == "none":
                if attack_type is None:
                    if isinstance(utility, bool):
                        utility_vals.append(1.0 if utility else 0.0)
            else:
                if attack_type is not None and inj_id is not None:
                    if isinstance(security, bool):
                        security_vals.append(1.0 if security else 0.0)

        utility_mean = safe_mean(utility_vals)
        security_mean = safe_mean(security_vals)
        asr = (1.0 - security_mean) if security_mean is not None else None
        duration_mean = safe_mean(duration_vals)
        api_total, api_openrouter, api_local = parse_api_counts(run_log)
        gpu_avg, gpu_max, gpu_mem_max = parse_gpu_metrics(gpu_log)

        out = {
            "seed": seed,
            "defense": defense,
            "attack_mode": attack_mode,
            "status": str(status),
            "utility_mean": fmt(utility_mean),
            "security_mean": fmt(security_mean),
            "asr": fmt(asr),
            "duration_mean_sec": fmt(duration_mean),
            "api_calls_total": str(api_total),
            "api_calls_openrouter": str(api_openrouter),
            "api_calls_local": str(api_local),
            "gpu_util_avg": fmt(gpu_avg),
            "gpu_util_max": fmt(gpu_max),
            "gpu_mem_max_mib": fmt(gpu_mem_max),
            "logdir": str(logdir),
            "run_log": str(run_log),
            "gpu_log": str(gpu_log),
        }
        per_combo.append(out)
        grouped.setdefault((defense, attack_mode), []).append(out)

    per_combo_csv = run_root / "prelim_per_combo.csv"
    with per_combo_csv.open("w", newline="") as f:
        if per_combo:
            w = csv.DictWriter(f, fieldnames=list(per_combo[0].keys()))
            w.writeheader()
            w.writerows(per_combo)

    summary_rows: list[dict[str, str]] = []
    for (defense, attack_mode), items in sorted(grouped.items()):
        def _m(field: str) -> float | None:
            vals = []
            for it in items:
                v = it.get(field, "")
                if v == "":
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    continue
            return safe_mean(vals)

        summary_rows.append(
            {
                "defense": defense,
                "attack_mode": attack_mode,
                "n_runs": str(len(items)),
                "utility_mean": fmt(_m("utility_mean")),
                "security_mean": fmt(_m("security_mean")),
                "asr_mean": fmt(_m("asr")),
                "duration_mean_sec": fmt(_m("duration_mean_sec")),
                "api_calls_total_mean": fmt(_m("api_calls_total")),
                "api_calls_openrouter_mean": fmt(_m("api_calls_openrouter")),
                "api_calls_local_mean": fmt(_m("api_calls_local")),
                "gpu_util_avg_mean": fmt(_m("gpu_util_avg")),
                "gpu_util_max_mean": fmt(_m("gpu_util_max")),
                "gpu_mem_max_mib_mean": fmt(_m("gpu_mem_max_mib")),
            }
        )

    summary_csv = run_root / "prelim_summary.csv"
    with summary_csv.open("w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    md_path = run_root / "prelim_tables.md"
    with md_path.open("w") as f:
        f.write("# Preliminary Tables\n\n")
        f.write("## Utility (attack=none)\n\n")
        f.write("| defense | utility_mean |\n|---|---:|\n")
        for r in summary_rows:
            if r["attack_mode"] == "none":
                f.write(f"| {r['defense']} | {r['utility_mean']} |\n")
        f.write("\n## Security / ASR (attack=sboa)\n\n")
        f.write("| defense | security_mean | asr_mean |\n|---|---:|---:|\n")
        for r in summary_rows:
            if r["attack_mode"] == "sboa":
                f.write(f"| {r['defense']} | {r['security_mean']} | {r['asr_mean']} |\n")
        f.write("\n## Efficiency (mean)\n\n")
        f.write("| defense | attack_mode | duration_mean_sec | api_calls_total_mean | gpu_util_avg_mean | gpu_mem_max_mib_mean |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(
                f"| {r['defense']} | {r['attack_mode']} | {r['duration_mean_sec']} | {r['api_calls_total_mean']} | {r['gpu_util_avg_mean']} | {r['gpu_mem_max_mib_mean']} |\n"
            )

    print(f"[done] wrote {per_combo_csv}")
    print(f"[done] wrote {summary_csv}")
    print(f"[done] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
