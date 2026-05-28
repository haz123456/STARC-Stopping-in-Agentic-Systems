import argparse
from pathlib import Path
from typing import Dict

from stage3_common import load_summary_or_decision_metrics


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def extract_row(summary: Dict) -> Dict[str, object]:
    metrics = summary.get("metrics", {})
    return {
        "policy": summary.get("policy"),
        "examples": metrics.get("examples"),
        "gate_success_pct": metrics.get("gate_success_pct"),
        "stop_recall_pct": metrics.get("stop_recall_pct"),
        "continue_recall_pct": metrics.get("continue_recall_pct"),
        "continue_precision_pct": metrics.get("continue_precision_pct"),
        "balanced_accuracy_pct": metrics.get("balanced_accuracy_pct"),
        "mcc": metrics.get("mcc"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VRB, VRRS, and VRRS-BR outputs.")
    parser.add_argument("--files", nargs="+", required=True, help="Summary JSON files or decision JSONL files.")
    parser.add_argument("--labels-file", default=None, help="Optional shared round-1 labels file for legacy decision JSONL inputs.")
    args = parser.parse_args()

    rows = []
    for summary_path in [Path(path).resolve() for path in args.files]:
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        rows.append(extract_row(load_summary_or_decision_metrics(summary_path, labels_file=args.labels_file)))

    order = {"vrb": 0, "vrrs": 1, "vrrs_br": 2}
    rows.sort(key=lambda row: order.get(str(row.get("policy")), 999))

    print("policy | examples | gate success | stop recall | continue recall | continue precision | balanced accuracy | mcc")
    print("--- | --- | --- | --- | --- | --- | --- | ---")
    for row in rows:
        print(
            f"{row['policy']} | {fmt(row['examples'])} | {fmt(row['gate_success_pct'])} | {fmt(row['stop_recall_pct'])} | "
            f"{fmt(row['continue_recall_pct'])} | {fmt(row['continue_precision_pct'])} | "
            f"{fmt(row['balanced_accuracy_pct'])} | {fmt(row['mcc'])}"
        )


if __name__ == "__main__":
    main()
