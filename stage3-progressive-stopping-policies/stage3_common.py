"""Shared file and metric helpers for the stage 3 submission package."""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_DIR / "output"

KNOWN_DECISION_SUFFIXES = (
    "__vrb_decisions",
    "__vrrs_decisions",
    "__vrrs_br_decisions",
)

POLICY_NAME_MAP = {
    "vrb": "vrb",
    "vrrs": "vrrs",
    "vrrs_br": "vrrs_br",
    "threshold_llm_verification_gate": "vrb",
    "saved_verify": "vrb",
    "risk_signal_gate": "vrrs",
    "live_hybrid_no_gpt": "vrrs",
    "live_hybrid_gpt_v2": "vrrs_br",
}


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def canonical_policy_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return POLICY_NAME_MAP.get(normalized, normalized or None)


def write_json(path: str | Path, payload: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clamp01(value: Any) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def to_band(score: Any) -> Optional[str]:
    if score is None:
        return None
    s = float(score)
    if s < 0.4:
        return "low"
    if s < 0.6:
        return "moderate"
    return "high"


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower().replace("%", "")
        if not normalized:
            return None
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


def coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    return None


def normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    normalized = normalized.replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    return normalized.strip(" .,:;!?\"'")


def extract_json_block(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return " ".join(str(text).lower().split())


def tokenize_words(text: Optional[str]) -> List[str]:
    return re.findall(r"[a-z0-9']+", normalize_text(text))


def load_scored_round1_labels(path: str | Path) -> Dict[str, Dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    labels: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        example_id = row.get("example_id")
        if not isinstance(example_id, str):
            continue
        verification = row.get("gpt4_verification_round1") or {}
        labels[example_id] = {
            "round1_correct": verification.get("correct_answer"),
            "question": row.get("question"),
        }
    return labels


def load_round1_label_lookup(path: str | Path) -> Dict[tuple[Optional[str], str], Optional[bool]]:
    label_path = Path(path)
    if label_path.suffix == ".json":
        scored = load_scored_round1_labels(label_path)
        return {(None, example_id): row.get("round1_correct") for example_id, row in scored.items()}

    lookup: Dict[tuple[Optional[str], str], Optional[bool]] = {}
    for row in iter_jsonl(label_path):
        example_id = row.get("example_id")
        if not isinstance(example_id, str):
            continue
        dataset = row.get("dataset")
        dataset_key = str(dataset) if isinstance(dataset, str) else None
        lookup[(dataset_key, example_id)] = coerce_bool(row.get("round1_correct"))
        lookup[(None, example_id)] = coerce_bool(row.get("round1_correct"))
    return lookup


def extract_round1_answer(record: Dict[str, Any]) -> Optional[str]:
    answer = record.get("round1_answer")
    if isinstance(answer, str) and answer.strip():
        return answer

    provisional = record.get("provisional_answers_by_round") or []
    if provisional and isinstance(provisional[0], dict):
        answer = provisional[0].get("final_answer")
        if isinstance(answer, str) and answer.strip():
            return answer

    answer = record.get("best_answer_selected")
    if isinstance(answer, str) and answer.strip():
        return answer
    return None


def extract_round1_state(record: Dict[str, Any]) -> Dict[str, Any]:
    clarification_rounds = record.get("clarification_rounds") or []
    verify_refine_records = record.get("verify_refine_records") or []
    clarification_state = clarification_rounds[0] if clarification_rounds else {}
    verify_refine_round1 = verify_refine_records[0] if verify_refine_records else {}

    return {
        "example_id": record.get("example_id"),
        "dataset": record.get("dataset"),
        "source_idx": record.get("source_idx"),
        "question": record.get("question"),
        "answers": record.get("answers"),
        "round1_answer": extract_round1_answer(record),
        "clarification_state": clarification_state,
        "verify_refine_round1": verify_refine_round1,
        "saved_verifier_record": verify_refine_round1.get("verifier_record"),
        "saved_refine_decision": verify_refine_round1.get("refine_decision"),
        "saved_refine_reason": verify_refine_round1.get("refine_reason"),
        "saved_stopped_after_round": record.get("stopped_after_round"),
        "saved_num_rounds_used": record.get("num_clarification_rounds_used"),
    }


def strip_known_suffixes(name: str) -> str:
    base = name
    for suffix in KNOWN_DECISION_SUFFIXES:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def default_decision_output_path(input_file: str | Path, policy_name: str) -> Path:
    input_path = Path(input_file).resolve()
    name = input_path.name
    if name.endswith(".jsonl"):
        stem = name[:-6]
    elif name.endswith(".json"):
        stem = name[:-5]
    else:
        stem = input_path.stem
    stem = strip_known_suffixes(stem)
    return OUTPUT_DIR / f"{stem}__{policy_name}_decisions.jsonl"


def default_summary_path(output_file: str | Path) -> Path:
    output_path = Path(output_file).resolve()
    name = output_path.name
    if name.endswith(".jsonl"):
        name = name[:-6]
    return output_path.with_name(f"{name}_summary.json")


def infer_labels_file(decision_file: str | Path) -> Optional[Path]:
    decision_path = Path(decision_file).resolve()
    candidates = (
        decision_path.with_name("replay_round1_labels.jsonl"),
        decision_path.with_name("round1_labels.jsonl"),
        decision_path.with_name("threshold_llm_verification_gate_labels.jsonl"),
        decision_path.with_name("risk_signal_gate_labels.jsonl"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def safe_pct(num: int, den: int) -> Optional[float]:
    if den == 0:
        return None
    return 100.0 * num / den


def wilson_interval(num: int, den: int, z: float = 1.96) -> Optional[Dict[str, float]]:
    if den == 0:
        return None
    phat = num / den
    denom = 1.0 + (z * z) / den
    center = (phat + (z * z) / (2.0 * den)) / denom
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) / den) + ((z * z) / (4.0 * den * den)))
        / denom
    )
    return {
        "low_pct": 100.0 * max(0.0, center - margin),
        "high_pct": 100.0 * min(1.0, center + margin),
    }


def matthews_corrcoef(tp: int, tn: int, fp: int, fn: int) -> Optional[float]:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0.0:
        return None
    return (tp * tn - fp * fn) / denom


def compute_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [
        row
        for row in rows
        if isinstance(row.get("stage3_round1_correct"), bool)
        and isinstance(row.get("stage3_final_continue_decision"), bool)
    ]

    stop_on_correct = 0
    continue_on_wrong = 0
    continue_on_correct = 0
    stop_on_wrong = 0
    borderline_reviewed = 0

    for row in usable:
        correct = bool(row["stage3_round1_correct"])
        continued = bool(row["stage3_final_continue_decision"])
        if row.get("stage3_vrrs_br", {}).get("review_applied") is True:
            borderline_reviewed += 1
        if correct and not continued:
            stop_on_correct += 1
        elif correct and continued:
            continue_on_correct += 1
        elif (not correct) and continued:
            continue_on_wrong += 1
        else:
            stop_on_wrong += 1

    examples = len(usable)
    gate_success_count = stop_on_correct + continue_on_wrong
    gate_incorrect_count = continue_on_correct + stop_on_wrong
    actual_stop_total = stop_on_correct + continue_on_correct
    actual_continue_total = continue_on_wrong + stop_on_wrong
    predicted_stop_total = stop_on_correct + stop_on_wrong
    predicted_continue_total = continue_on_wrong + continue_on_correct

    stop_recall = safe_pct(stop_on_correct, actual_stop_total)
    continue_recall = safe_pct(continue_on_wrong, actual_continue_total)
    stop_precision = safe_pct(stop_on_correct, predicted_stop_total)
    continue_precision = safe_pct(continue_on_wrong, predicted_continue_total)
    balanced_accuracy = None
    if stop_recall is not None and continue_recall is not None:
        balanced_accuracy = (stop_recall + continue_recall) / 2.0

    metrics: Dict[str, Any] = {
        "examples": examples,
        "stop_on_correct": stop_on_correct,
        "continue_on_wrong": continue_on_wrong,
        "continue_on_correct": continue_on_correct,
        "stop_on_wrong": stop_on_wrong,
        "gate_success_count": gate_success_count,
        "gate_success_pct": safe_pct(gate_success_count, examples),
        "gate_incorrect_count": gate_incorrect_count,
        "gate_incorrect_pct": safe_pct(gate_incorrect_count, examples),
        "stop_recall_pct": stop_recall,
        "continue_recall_pct": continue_recall,
        "stop_precision_pct": stop_precision,
        "continue_precision_pct": continue_precision,
        "balanced_accuracy_pct": balanced_accuracy,
        "mcc": matthews_corrcoef(
            tp=continue_on_wrong,
            tn=stop_on_correct,
            fp=continue_on_correct,
            fn=stop_on_wrong,
        ),
        "borderline_reviewed_count": borderline_reviewed,
    }

    metrics["gate_success_ci95"] = wilson_interval(gate_success_count, examples)
    metrics["stop_recall_ci95"] = wilson_interval(stop_on_correct, actual_stop_total)
    metrics["continue_recall_ci95"] = wilson_interval(continue_on_wrong, actual_continue_total)
    metrics["stop_precision_ci95"] = wilson_interval(stop_on_correct, predicted_stop_total)
    metrics["continue_precision_ci95"] = wilson_interval(continue_on_wrong, predicted_continue_total)
    return metrics


def row_round1_correct(
    row: Dict[str, Any],
    label_lookup: Dict[tuple[Optional[str], str], Optional[bool]],
) -> Optional[bool]:
    value = row.get("stage3_round1_correct")
    if isinstance(value, bool):
        return value

    example_id = row.get("example_id")
    if not isinstance(example_id, str):
        return None
    dataset = row.get("dataset")
    dataset_key = str(dataset) if isinstance(dataset, str) else None
    return label_lookup.get((dataset_key, example_id), label_lookup.get((None, example_id)))


def normalize_decision_rows_for_metrics(
    path: str | Path,
    labels_file: str | Path | None = None,
) -> Dict[str, Any]:
    decision_path = Path(path).resolve()
    rows = load_jsonl(decision_path)
    if not rows:
        return {"policy": canonical_policy_name(decision_path.stem) or decision_path.stem, "rows": []}

    label_path = Path(labels_file).resolve() if labels_file else infer_labels_file(decision_path)
    label_lookup = load_round1_label_lookup(label_path) if label_path else {}

    normalized_rows: List[Dict[str, Any]] = []
    policy_name = canonical_policy_name(rows[0].get("stage3_active_policy") or rows[0].get("policy") or decision_path.stem)
    for row in rows:
        row_policy = canonical_policy_name(row.get("stage3_active_policy") or row.get("policy") or policy_name)
        round1_correct = row_round1_correct(row, label_lookup)
        final_continue = row.get("stage3_final_continue_decision")
        if not isinstance(final_continue, bool):
            final_continue = coerce_bool(row.get("continue_decision"))

        normalized = dict(row)
        normalized["stage3_active_policy"] = row_policy
        normalized["stage3_round1_correct"] = round1_correct
        normalized["stage3_final_continue_decision"] = final_continue

        if row_policy == "vrrs_br" and "stage3_vrrs_br" not in normalized:
            review_applied = row.get("borderline_continue_to_round2") is not None or row.get("has_borderline_prediction") is True
            normalized["stage3_vrrs_br"] = {"review_applied": review_applied}

        normalized_rows.append(normalized)

    return {
        "policy": policy_name,
        "rows": normalized_rows,
        "labels_file": None if label_path is None else str(label_path),
    }


def load_summary_or_decision_metrics(
    path: str | Path,
    labels_file: str | Path | None = None,
) -> Dict[str, Any]:
    summary_path = Path(path).resolve()
    if summary_path.suffix == ".json":
        return load_json(summary_path)

    loaded = normalize_decision_rows_for_metrics(summary_path, labels_file=labels_file)
    return {
        "policy": loaded["policy"],
        "input_file": str(summary_path),
        "labels_file": loaded["labels_file"],
        "metrics": compute_metrics(loaded["rows"]),
    }


def build_summary(
    policy_name: str,
    input_file: str | Path,
    output_file: str | Path,
    labels_file: str | Path | None,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "policy": policy_name,
        "input_file": str(Path(input_file).resolve()),
        "output_file": str(Path(output_file).resolve()),
        "labels_file": None if labels_file is None else str(Path(labels_file).resolve()),
        "metrics": compute_metrics(rows),
    }