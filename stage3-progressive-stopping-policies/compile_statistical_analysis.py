import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from stage3_common import OUTPUT_DIR, compute_metrics, normalize_decision_rows_for_metrics, write_json


POLICY_ORDER = ["vrb", "vrrs", "vrrs_br"]
POLICY_LABELS = {
    "vrb": "VRB",
    "vrrs": "VRRS",
    "vrrs_br": "VRRS-BR",
}
DATASET_ORDER = ["nq", "popqa", "triviaqa", "hotpotqa"]
REPLAY_FILES = {
    "vrb": "replay_vrb_decisions.jsonl",
    "vrrs": "replay_vrrs_decisions.jsonl",
    "vrrs_br": "replay_vrrs_br_decisions.jsonl",
}


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def exact_binomial_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = 0.0
    cutoff = min(k, n - k)
    for i in range(cutoff + 1):
        tail += math.comb(n, i)
    return min(1.0, 2.0 * tail / (2**n))


def cluster_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row.get("dataset") or ""), str(row.get("question") or row.get("example_id") or "")


def build_clusters(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[cluster_key(row)].append(row)
    return dict(grouped)


def grouped_bootstrap(
    policy_rows: Dict[str, List[Dict[str, Any]]],
    reps: int,
    seed: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    rng = random.Random(seed)
    clusters_by_policy = {policy: build_clusters(rows) for policy, rows in policy_rows.items()}
    cluster_keys = sorted(set().union(*(clusters.keys() for clusters in clusters_by_policy.values())))
    by_dataset: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for key in cluster_keys:
        by_dataset[key[0]].append(key)

    metrics_store: Dict[str, Dict[str, List[float]]] = {
        policy: defaultdict(list) for policy in POLICY_ORDER
    }

    for _ in range(reps):
        sampled_keys: List[Tuple[str, str]] = []
        for dataset, keys in by_dataset.items():
            del dataset
            for _ in range(len(keys)):
                sampled_keys.append(rng.choice(keys))

        for policy in POLICY_ORDER:
            sample_rows: List[Dict[str, Any]] = []
            policy_clusters = clusters_by_policy[policy]
            for key in sampled_keys:
                sample_rows.extend(policy_clusters.get(key, []))
            metrics = compute_metrics(sample_rows)
            for metric_name in [
                "gate_success_pct",
                "balanced_accuracy_pct",
                "continue_recall_pct",
                "continue_precision_pct",
                "mcc",
            ]:
                value = metrics.get(metric_name)
                if value is not None:
                    metrics_store[policy][metric_name].append(float(value))

    summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for policy in POLICY_ORDER:
        for metric_name, values in metrics_store[policy].items():
            values = sorted(values)
            summary[policy][metric_name] = {
                "mean": sum(values) / len(values),
                "ci95_low": percentile(values, 0.025),
                "ci95_high": percentile(values, 0.975),
            }
    return summary


def cluster_mean_success(clusters: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> Dict[Tuple[str, str], float]:
    result = {}
    for key, rows in clusters.items():
        success = 0
        for row in rows:
            correct = bool(row["stage3_round1_correct"])
            continued = bool(row["stage3_final_continue_decision"])
            success += 1 if ((correct and not continued) or ((not correct) and continued)) else 0
        result[key] = success / len(rows)
    return result


def paired_cluster_significance(
    policy_a: str,
    policy_b: str,
    policy_rows: Dict[str, List[Dict[str, Any]]],
    reps: int,
    seed: int,
) -> Dict[str, Any]:
    clusters_a = cluster_mean_success(build_clusters(policy_rows[policy_a]))
    clusters_b = cluster_mean_success(build_clusters(policy_rows[policy_b]))
    keys = sorted(set(clusters_a) & set(clusters_b))
    diffs = [clusters_b[key] - clusters_a[key] for key in keys]
    observed = sum(diffs) / len(diffs)
    nonzero = [diff for diff in diffs if abs(diff) > 1e-12]
    wins = sum(1 for diff in nonzero if diff > 0)
    losses = sum(1 for diff in nonzero if diff < 0)
    ties = len(diffs) - len(nonzero)
    sign_p = exact_binomial_two_sided(min(wins, losses), len(nonzero))

    rng = random.Random(seed)
    extreme = 0
    abs_observed = abs(observed)
    for _ in range(reps):
        permuted = [diff if rng.random() < 0.5 else -diff for diff in nonzero]
        stat = abs(sum(permuted) / len(diffs))
        if stat >= abs_observed - 1e-12:
            extreme += 1
    permutation_p = (extreme + 1) / (reps + 1)

    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "cluster_count": len(diffs),
        "mean_difference_success_rate": observed,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_p": sign_p,
        "permutation_p": permutation_p,
    }


def majority_cluster_success(rows: List[Dict[str, Any]]) -> Any:
    success = 0
    for row in rows:
        correct = bool(row["stage3_round1_correct"])
        continued = bool(row["stage3_final_continue_decision"])
        success += 1 if ((correct and not continued) or ((not correct) and continued)) else 0
    rate = success / len(rows)
    if rate > 0.5:
        return True
    if rate < 0.5:
        return False
    return None


def cluster_collapsed_mcnemar(
    policy_a: str,
    policy_b: str,
    policy_rows: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    clusters_a = build_clusters(policy_rows[policy_a])
    clusters_b = build_clusters(policy_rows[policy_b])
    keys = sorted(set(clusters_a) & set(clusters_b))

    a_fail_b_success = 0
    a_success_b_fail = 0
    excluded_tied_clusters = 0
    for key in keys:
        a_state = majority_cluster_success(clusters_a[key])
        b_state = majority_cluster_success(clusters_b[key])
        if a_state is None or b_state is None:
            excluded_tied_clusters += 1
            continue
        if a_state is False and b_state is True:
            a_fail_b_success += 1
        elif a_state is True and b_state is False:
            a_success_b_fail += 1

    discordant = a_fail_b_success + a_success_b_fail
    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "discordant_a_fail_b_success": a_fail_b_success,
        "discordant_a_success_b_fail": a_success_b_fail,
        "discordant_total": discordant,
        "excluded_tied_clusters": excluded_tied_clusters,
        "exact_mcnemar_p": exact_binomial_two_sided(min(a_fail_b_success, a_success_b_fail), discordant),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_policy_files(output_dir: Path, policy: str) -> List[Path]:
    replay_candidate = output_dir / REPLAY_FILES[policy]
    if replay_candidate.exists():
        return [replay_candidate]
    return sorted(output_dir.glob(f"*__{policy}_decisions.jsonl"))


def load_policy_rows(decision_files: List[Path], labels_file: Path | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for decision_file in decision_files:
        loaded = normalize_decision_rows_for_metrics(decision_file, labels_file=labels_file)
        rows.extend(loaded["rows"])
    return rows


def dataset_sort_key(dataset: str) -> tuple[int, str]:
    try:
        return (DATASET_ORDER.index(dataset), dataset)
    except ValueError:
        return (len(DATASET_ORDER), dataset)


def build_metrics_tables(policy_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    combined_rows: List[Dict[str, Any]] = []
    dataset_rows: List[Dict[str, Any]] = []
    metrics_payload: Dict[str, Dict[str, Any]] = defaultdict(dict)

    all_datasets = sorted(
        {str(row.get("dataset")) for rows in policy_rows.values() for row in rows if row.get("dataset")},
        key=dataset_sort_key,
    )

    for policy in POLICY_ORDER:
        combined = compute_metrics(policy_rows[policy])
        metrics_payload[policy]["combined"] = combined
        combined_rows.append({"policy": policy, "policy_label": POLICY_LABELS[policy], **combined})

        for dataset in all_datasets:
            rows = [row for row in policy_rows[policy] if row.get("dataset") == dataset]
            dataset_metrics = compute_metrics(rows)
            metrics_payload[policy][dataset] = dataset_metrics
            dataset_rows.append(
                {
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "dataset": dataset,
                    **dataset_metrics,
                }
            )

    return {
        "metrics_payload": metrics_payload,
        "combined_rows": combined_rows,
        "dataset_rows": dataset_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile stage 3 policy metrics and statistical tests.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Stage 3 output directory.")
    parser.add_argument("--labels-file", default=None, help="Optional shared round-1 labels file for legacy replay outputs.")
    parser.add_argument("--bootstrap-reps", type=int, default=4000, help="Grouped bootstrap repetitions.")
    parser.add_argument("--permutation-reps", type=int, default=50000, help="Paired sign-flip permutation repetitions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    labels_file = Path(args.labels_file).resolve() if args.labels_file else None
    stats_dir = output_dir / "statistics"

    policy_files = {policy: discover_policy_files(output_dir, policy) for policy in POLICY_ORDER}
    missing = [policy for policy, files in policy_files.items() if not files]
    if missing:
        raise FileNotFoundError(f"Missing policy decision files in {output_dir}: {', '.join(missing)}")

    policy_rows = {
        policy: load_policy_rows(files, labels_file=labels_file)
        for policy, files in policy_files.items()
    }

    metrics_tables = build_metrics_tables(policy_rows)
    bootstrap = grouped_bootstrap(policy_rows, reps=args.bootstrap_reps, seed=args.seed)
    paired = [
        paired_cluster_significance("vrb", "vrrs", policy_rows, reps=args.permutation_reps, seed=args.seed),
        paired_cluster_significance("vrrs", "vrrs_br", policy_rows, reps=args.permutation_reps, seed=args.seed),
        paired_cluster_significance("vrb", "vrrs_br", policy_rows, reps=args.permutation_reps, seed=args.seed),
    ]
    mcnemar = [
        cluster_collapsed_mcnemar("vrb", "vrrs", policy_rows),
        cluster_collapsed_mcnemar("vrrs", "vrrs_br", policy_rows),
        cluster_collapsed_mcnemar("vrb", "vrrs_br", policy_rows),
    ]

    stats_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "bootstrap_repetitions": args.bootstrap_reps,
        "permutation_repetitions": args.permutation_reps,
        "policy_files": {policy: [str(path) for path in files] for policy, files in policy_files.items()},
        "labels_file": None if labels_file is None else str(labels_file),
        "metrics": metrics_tables["metrics_payload"],
        "bootstrap": bootstrap,
        "paired_tests": paired,
        "mcnemar_robustness": mcnemar,
    }
    write_json(stats_dir / "statistical_analysis.json", payload)
    write_csv(stats_dir / "policy_metrics.csv", metrics_tables["combined_rows"])
    write_csv(stats_dir / "dataset_metrics.csv", metrics_tables["dataset_rows"])

    bootstrap_rows = []
    for policy in POLICY_ORDER:
        for metric_name, values in bootstrap.get(policy, {}).items():
            bootstrap_rows.append(
                {
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "metric": metric_name,
                    **values,
                }
            )
    write_csv(stats_dir / "bootstrap_intervals.csv", bootstrap_rows)
    write_csv(stats_dir / "paired_tests.csv", paired)
    write_csv(stats_dir / "mcnemar_robustness.csv", mcnemar)

    print(json.dumps(
        {
            "statistics_dir": str(stats_dir),
            "policy_files": {policy: [str(path) for path in files] for policy, files in policy_files.items()},
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
