#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PBE-oriented evaluation script for the multi-agent framework.

This script computes three families of metrics from saved JSONL results:
A) Sequential rationality (approximate): regret gap based on per-agent payoff tables
B) On-path Bayesian consistency: check belief updates vs. precision-weighted formula (requires 'evaluation' field)
C) Early-stop verification: L2 belief change threshold across rounds

Usage:
    python -m eval_pbe --file /path/to/results.jsonl [--epsilon 0.05] [--tau 1e-6]
    python -m eval_pbe --dir  /path/to/results_dir   [--epsilon 0.05] [--tau 1e-6]

Outputs a summary JSON next to the input file(s) and prints a compact report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from statistics import mean
import math
import glob


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    return items


def l2_distance(vec1: List[float], vec2: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def find_latest_jsonl_in_dir(dir_path: Path) -> Path | None:
    matches = sorted(dir_path.glob("**/*.jsonl"))
    return matches[-1] if matches else None


def sequential_rationality_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Regret gap per round and aggregate, using each participant's payoff table.
    Assumes result["payoff"][t] is a dict where keys include participant roles and
    each maps to a dict of {action: payoff} for that participant at round t.

    Chosen action is read from interaction_history[t*2 + k]["strategy"],
    iterating through role order as it appears.
    """
    payoffs: List[Dict[str, Any]] = result.get("payoff", [])
    history: List[Dict[str, Any]] = result.get("interaction_history", [])

    if not payoffs or not history:
        return {"status": "no-data"}

    # Build per-round chosen actions for each role, preserving order
    # Each round has len(unique_roles) messages. We infer roles from the first round.
    roles: List[str] = []
    for msg in history[:2]:
        r = msg.get("role")
        if r and r not in roles:
            roles.append(r)
    roles = roles or list(payoffs[0].keys())

    round_gaps: List[float] = []
    per_role_gaps: Dict[str, List[float]] = {r: [] for r in roles}

    # Determine round size as number of roles
    R = len(roles) if len(roles) > 0 else 2

    for t, pf in enumerate(payoffs):
        # messages for this round
        start = t * R
        msgs = history[start : start + R]
        for msg in msgs:
            role = msg.get("role")
            chosen = (msg.get("strategy") or "").lower().strip()
            payoff_table = pf.get(role, {}) if isinstance(pf, dict) else {}
            # Try common keys normalization
            norm = {
                str(k).lower().strip(): to_float(v, float("nan"))
                for k, v in payoff_table.items()
                if isinstance(k, str)
            }
            # Compute regret gap
            if not norm:
                continue
            max_payoff = max(v for v in norm.values() if isinstance(v, (int, float)))
            chosen_payoff = norm.get(chosen, float("nan"))
            if isinstance(chosen_payoff, float) and not math.isnan(chosen_payoff):
                gap = max(0.0, max_payoff - chosen_payoff)
                round_gaps.append(gap)
                per_role_gaps.setdefault(role, []).append(gap)

    summary = {
        "status": "ok",
        "avg_gap": mean(round_gaps) if round_gaps else None,
        "median_gap": sorted(round_gaps)[len(round_gaps) // 2] if round_gaps else None,
        "per_role_avg_gap": {
            r: (mean(v) if v else None) for r, v in per_role_gaps.items()
        },
        "num_round_observations": len(round_gaps),
    }
    return summary


def bayesian_consistency_metrics(
    result: Dict[str, Any], delta: float, tau: float
) -> Dict[str, Any]:
    """
    Verify on-path Bayesian update per dimension using evaluation (e, w) vs belief_state snapshots.
    We approximate per-round posterior by folding evaluations of that round in chronological order,
    starting from previous round's belief for each role.

    Requirements:
    - result["belief_state"]: list per round, each element like {role: {target_role: {dim: val,...}}}
    - result["evaluation"]: list in the speaking order; each eval dict must contain:
      {"belief": {dim: e_val,...}, "confidence": {dim: w_val,...}}
      where e_val is the assessed attribute (we treat as e), w_val as weight.

    Note: If structure deviates, we best-effort and return partial metrics.
    """
    belief_state: List[Dict[str, Any]] = result.get("belief_state", [])
    evals: List[Dict[str, Any]] = result.get("evaluation", [])
    history: List[Dict[str, Any]] = result.get("interaction_history", [])

    if not belief_state:
        return {"status": "no-belief"}
    if not evals or not history:
        return {"status": "no-evaluation"}

    # Infer roles order per round from history
    roles: List[str] = []
    for msg in history[:2]:
        r = msg.get("role")
        if r and r not in roles:
            roles.append(r)
    R = len(roles) if roles else 2

    # For each round, two evals (if R=2). Map evals to speaker in order.
    # belief_state[t] is a snapshot after applying both evals of round t.
    # We'll check that applying the two evals sequentially starting from previous round's belief
    # yields approximately the snapshot.

    # Build role->target mapping from first belief snapshot
    # For role i, there should be exactly one target j
    role_targets: Dict[str, str] = {}
    if belief_state:
        first = belief_state[0]
        for i_role, targets in first.items():
            if isinstance(targets, dict) and targets:
                role_targets[i_role] = list(targets.keys())[0]

    # Helper: get belief vector per role from a snapshot dict
    def snapshot_to_vec(
        snapshot: Dict[str, Any], i_role: str
    ) -> Tuple[List[str], List[float]]:
        tgt = role_targets.get(i_role)
        if tgt is None:
            return ([], [])
        dims = []
        vals = []
        try:
            dim_map = snapshot[i_role][tgt]
            for k, v in dim_map.items():
                dims.append(k)
                vals.append(to_float(v, 0.0))
        except Exception:
            pass
        return dims, vals

    # Initialize counters
    abs_errs: List[float] = []
    dir_violations = 0
    updates = 0

    # We don't have prior snapshot (round -1), so assume the first snapshot is after round 1.
    # We cannot reconstruct exactly the intermediate posterior after the first speech; we only check final snapshot per round.

    eval_idx = 0
    for t in range(len(belief_state)):
        snap_t = belief_state[t]
        # previous belief for each role: use (t-1) snapshot if available, else reuse current as baseline
        snap_prev = belief_state[t - 1] if t - 1 >= 0 else snap_t

        # Apply R evals of this round to previous beliefs for each listener
        # Each evaluation corresponds to a speaker s; listeners are the others (here R=2)
        folded_posteriors: Dict[str, Dict[str, float]] = {}
        folded_dims: Dict[str, List[str]] = {}

        for k in range(R):
            if eval_idx >= len(evals) or (t * R + k) >= len(history):
                break
            speaker = history[t * R + k].get("role")
            # listeners are roles except speaker
            listeners = [r for r in roles if r != speaker]
            ev = evals[eval_idx]
            eval_idx += 1
            e_map = ev.get("belief", {})
            w_map = ev.get("confidence", {})

            for listener in listeners:
                dims_prev, vals_prev = snapshot_to_vec(snap_prev, listener)
                if not dims_prev:
                    continue
                # Initialize folded posterior if first time
                if listener not in folded_posteriors:
                    folded_posteriors[listener] = {
                        d: v for d, v in zip(dims_prev, vals_prev)
                    }
                    folded_dims[listener] = list(dims_prev)
                # Confidence track: we don't store per-dim c in snapshot; approximate using w_map keys with c in [0,1]
                # We reconstruct c_t by treating it as equal to previous w running value if available is missing in data.
                # However, since true c is not stored in results, we cannot verify c exactly. We only check μ direction and magnitude proportional to w.
                for d in folded_dims[listener]:
                    mu_prev = folded_posteriors[listener][d]
                    e_val = to_float(e_map.get(d), mu_prev)
                    w_val = to_float(w_map.get(d), 0.0)
                    # Use a proxy c_prev = 1 - (1 - small) to avoid div-by-zero; since absolute check is impossible, we only verify directional update
                    c_prev = 1e-6
                    mu_post = (c_prev * mu_prev + w_val * e_val) / max(
                        1e-6, (c_prev + w_val)
                    )
                    folded_posteriors[listener][d] = mu_post

        # Compare folded_posteriors with snap_t for listeners
        for listener, dim_list in folded_dims.items():
            _, vec_true = snapshot_to_vec(snap_t, listener)
            vec_hat = [folded_posteriors[listener][d] for d in dim_list]
            if vec_true and vec_hat and len(vec_true) == len(vec_hat):
                # absolute error per dim
                for a, b in zip(vec_true, vec_hat):
                    abs_errs.append(abs(a - b))
                updates += len(vec_true)
                # directional check against prev
                _, vec_prev = snapshot_to_vec(snap_prev, listener)
                for mu_p, mu_t, mu_hat in zip(vec_prev, vec_true, vec_hat):
                    # If true moved towards hat direction; we check sign consistency only
                    if (mu_hat - mu_p) * (mu_t - mu_p) < 0:
                        dir_violations += 1

    if updates == 0:
        return {"status": "insufficient", "note": "missing dims or evaluations"}

    return {
        "status": "ok-partial",  # partial because exact c is not reconstructable
        "abs_error_mean": mean(abs_errs) if abs_errs else None,
        "abs_error_p95": (
            sorted(abs_errs)[int(0.95 * len(abs_errs))] if abs_errs else None
        ),
        "direction_violations": dir_violations,
        "num_updates": updates,
        "note": "Confidence c not stored; used proxy for directional/magnitude sanity. For exact check, store c per dim.",
    }


def early_stop_metrics(result: Dict[str, Any], epsilon: float) -> Dict[str, Any]:
    belief_state: List[Dict[str, Any]] = result.get("belief_state", [])
    if not belief_state:
        return {"status": "no-belief"}

    # infer roles from all snapshots
    roles = set()
    for snap in belief_state:
        if isinstance(snap, dict):
            roles.update(snap.keys())
    roles = list(roles)

    # helper to get vector for a role
    def vec(snapshot: Dict[str, Any], role: str) -> List[float]:
        try:
            tgt = list(snapshot[role].keys())[0]
            return [to_float(v) for v in snapshot[role][tgt].values()]
        except Exception:
            return []

    # Build per-role belief vectors in order
    per_role_vectors: Dict[str, List[List[float]]] = {r: [] for r in roles}
    for snap in belief_state:
        for r in roles:
            vec_vals = vec(snap, r)
            if vec_vals:  # only append if not empty
                per_role_vectors[r].append(vec_vals)

    l2_changes: Dict[str, List[float]] = {r: [] for r in roles}
    for r, vectors in per_role_vectors.items():
        for i in range(1, len(vectors)):
            v_prev = vectors[i - 1]
            v_t = vectors[i]
            if len(v_t) == len(v_prev):
                raw_distance = l2_distance(v_t, v_prev)
                max_distance = math.sqrt(len(v_t))
                normalized_distance = (
                    raw_distance / max_distance if max_distance > 0 else 0.0
                )
                l2_changes[r].append(normalized_distance)

    # check consecutive three < epsilon at the tail
    early_stop_roles: Dict[str, bool] = {}
    for r, arr in l2_changes.items():
        ok = False
        if len(arr) >= 3:
            if arr[-1] < epsilon and arr[-2] < epsilon and arr[-3] < epsilon:
                ok = True
        early_stop_roles[r] = ok

    return {
        "status": "ok",
        "l2_changes": l2_changes,
        "early_stop_roles": early_stop_roles,
    }


def evaluate_file(
    path: Path, epsilon: float, tau: float, delta: float
) -> Dict[str, Any]:
    results = load_jsonl(path)
    summaries: List[Dict[str, Any]] = []

    for res in results:
        seq = sequential_rationality_metrics(res)
        bayes = bayesian_consistency_metrics(res, delta=delta, tau=tau)
        stop = early_stop_metrics(res, epsilon=epsilon)
        summaries.append(
            {
                "id": res.get("id"),
                "sequential": seq,
                "bayesian": bayes,
                "early_stop": stop,
            }
        )

    out = {
        "source": str(path),
        "num_cases": len(results),
        "cases": summaries,
    }

    out_path = path.with_name("pbe_eval.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="PBE-oriented evaluation for multi-agent runs"
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=str, help="Path to a JSONL result file")
    # g.add_argument("--dir", type=str, help="Directory to search for latest JSONL")
    parser.add_argument(
        "--epsilon", type=float, default=0.05, help="Early-stop L2 threshold"
    )
    parser.add_argument(
        "--tau", type=float, default=1e-6, help="Numerical tolerance (unused currently)"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="Confidence growth factor δ used in belief updates",
    )
    args = parser.parse_args()

    if not args.file:
        raise SystemExit("Error: --file argument is required and cannot be empty.")

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"Error: File not found: {file_path}")

    summary = evaluate_file(
        file_path, epsilon=args.epsilon, tau=args.tau, delta=args.delta
    )

    # Compact console report
    print("\n=== PBE Evaluation Summary ===")
    print(f"Source: {summary['source']} | Cases: {summary['num_cases']}")
    # Aggregate quick stats across cases where available
    all_gaps = []
    early_ok = 0
    abs_errors = []
    dir_violations_total = 0
    for c in summary["cases"]:
        seq = c.get("sequential", {})
        if isinstance(seq.get("avg_gap"), (int, float)):
            all_gaps.append(seq["avg_gap"])
        stop = c.get("early_stop", {}).get("early_stop_roles", {})
        if any(v for v in stop.values() if v):  # At least one role satisfied early stop
            early_ok += 1
        bayes = c.get("bayesian", {})
        if isinstance(bayes.get("abs_error_mean"), (int, float)):
            abs_errors.append(bayes["abs_error_mean"])
        if isinstance(bayes.get("direction_violations"), (int, float)):
            dir_violations_total += bayes["direction_violations"]
    if all_gaps:
        print(f"Avg regret gap across cases: {mean(all_gaps):.3f}")
    if abs_errors:
        print(f"Avg Bayesian abs error across cases: {mean(abs_errors):.3f}")
        print(f"Total direction violations: {dir_violations_total}")
    print(f"Early-stop satisfied cases: {early_ok}/{summary['num_cases']}")
    print("==============================\n")


if __name__ == "__main__":
    main()
