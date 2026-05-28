"""VRB system: replay the saved verify-refine round-1 stop/continue decision."""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from stage3_common import (
    build_summary,
    default_decision_output_path,
    default_summary_path,
    extract_round1_state,
    iter_jsonl,
    load_scored_round1_labels,
    write_json,
    write_jsonl,
)


POLICY_NAME = "vrb"


def build_vrb_record(source_record: Dict[str, Any], round1_correct: Optional[bool]) -> Dict[str, Any]:
    output_record = copy.deepcopy(source_record)
    round1_state = extract_round1_state(source_record)
    verify_refine_round1 = round1_state["verify_refine_round1"] or {}
    continue_decision = bool(round1_state.get("saved_refine_decision", False))
    decision_reason = round1_state.get("saved_refine_reason") or "missing_verify_refine_decision"

    output_record["stage3_active_policy"] = POLICY_NAME
    output_record["stage3_round1_correct"] = round1_correct
    output_record["stage3_final_continue_decision"] = continue_decision
    output_record["stage3_final_decision"] = "continue" if continue_decision else "stop"
    output_record["stage3_final_decision_reason"] = decision_reason
    output_record["stage3_vrb"] = {
        "policy": POLICY_NAME,
        "example_id": round1_state.get("example_id"),
        "question": round1_state.get("question"),
        "round1_answer": round1_state.get("round1_answer"),
        "verify_refine_round1_record": copy.deepcopy(verify_refine_round1),
        "continue_decision": continue_decision,
        "decision_reason": decision_reason,
    }
    return output_record


def run_vrb(
    input_file: str | Path,
    labels_file: str | Path | None = None,
    output_file: str | Path | None = None,
    summary_file: str | Path | None = None,
) -> Dict[str, Any]:
    input_path = Path(input_file).resolve()
    output_path = Path(output_file).resolve() if output_file else default_decision_output_path(input_path, POLICY_NAME)
    summary_path = Path(summary_file).resolve() if summary_file else default_summary_path(output_path)

    labels = load_scored_round1_labels(labels_file) if labels_file else {}
    decision_rows: List[Dict[str, Any]] = []
    for source_record in iter_jsonl(input_path):
        example_id = source_record.get("example_id")
        round1_correct = labels.get(example_id, {}).get("round1_correct")
        decision_rows.append(build_vrb_record(source_record, round1_correct))

    write_jsonl(output_path, decision_rows)
    summary = build_summary(POLICY_NAME, input_path, output_path, labels_file, decision_rows)
    write_json(summary_path, summary)
    return {
        "policy": POLICY_NAME,
        "output_file": str(output_path),
        "summary_file": str(summary_path),
        "examples": len(decision_rows),
    }
