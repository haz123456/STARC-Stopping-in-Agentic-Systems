"""VRRS-BR system: review only the borderline rows from a VRRS decision file."""

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from stage3_common import (
    build_summary,
    clamp01,
    coerce_bool,
    coerce_float,
    default_decision_output_path,
    default_summary_path,
    extract_json_block,
    extract_round1_state,
    iter_jsonl,
    load_scored_round1_labels,
    normalize_label,
    to_band,
    write_json,
    write_jsonl,
)


POLICY_NAME = "vrrs_br"

CONCRETE_ISSUE_TYPES = {
    "contradiction",
    "wrong entity",
    "wrong_entity",
    "competing answer",
    "competing_answer",
    "answer type mismatch",
    "answer_type_mismatch",
    "clue echo",
    "clue_echo",
}


def parse_borderline_review(raw_text: Optional[str]) -> Dict[str, Any]:
    parsed = {
        "continue_to_round2": None,
        "false_answer_risk": None,
        "issue_type": None,
        "concrete_error_detected": None,
        "reason": None,
        "parse_error": None,
        "raw_text": raw_text,
    }
    json_block = extract_json_block(raw_text)
    if json_block is None:
        parsed["parse_error"] = "no_json_found"
        return parsed
    try:
        data = json.loads(json_block)
    except Exception as exc:
        parsed["parse_error"] = f"json_decode_error: {exc}"
        return parsed

    parsed["continue_to_round2"] = coerce_bool(data.get("continue_to_round2"))
    parsed["false_answer_risk"] = clamp01(coerce_float(data.get("false_answer_risk")))
    parsed["issue_type"] = normalize_label(data.get("issue_type"))
    parsed["concrete_error_detected"] = coerce_bool(data.get("concrete_error_detected"))
    parsed["reason"] = data.get("reason")
    return parsed


def should_continue_from_review(parsed: Dict[str, Any]) -> bool:
    if not isinstance(parsed, dict):
        return False
    if parsed.get("continue_to_round2") is not True:
        return False
    if parsed.get("concrete_error_detected") is not True:
        return False
    if parsed.get("issue_type") not in CONCRETE_ISSUE_TYPES:
        return False
    false_risk = parsed.get("false_answer_risk")
    if to_band(false_risk) != "high":
        return False
    return True


def resolve_api_key(explicit_api_key: Optional[str]) -> str:
    api_key = (explicit_api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise ValueError("Set --api-key or OPENAI_API_KEY before running VRRS-BR.")
    return api_key


def run_borderline_review(prompt_text: str, model: str, temperature: float, api_key: str) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict QA control-layer judge. Return JSON only."},
            {"role": "user", "content": prompt_text},
        ],
        temperature=temperature,
        max_tokens=700,
        response_format={"type": "json_object"},
    )
    raw_text = response.choices[0].message.content
    return {
        "raw_output": raw_text,
        "parsed_output": parse_borderline_review(raw_text),
    }


def build_vrrs_br_record(
    vrrs_record: Dict[str, Any],
    round1_correct_override: Optional[bool],
    model: str,
    temperature: float,
    api_key: Optional[str],
    review_limit: Optional[int],
    review_counter: List[int],
) -> Dict[str, Any]:
    output_record = copy.deepcopy(vrrs_record)
    output_record["stage3_active_policy"] = POLICY_NAME

    if round1_correct_override is not None:
        output_record["stage3_round1_correct"] = round1_correct_override

    vrrs_payload = output_record.get("stage3_vrrs") or {}
    decision_band = vrrs_payload.get("decision_band")
    prompt_text = vrrs_payload.get("borderline_prompt_text")

    review_applied = False
    review_result = None
    final_continue_decision = bool(output_record.get("stage3_final_continue_decision", False))
    final_decision_reason = output_record.get("stage3_final_decision_reason") or "missing_vrrs_decision"

    if decision_band == "borderline":
        if review_limit is not None and review_counter[0] >= review_limit:
            final_continue_decision = False
            final_decision_reason = "borderline_review_skipped_due_to_limit"
        else:
            if not isinstance(prompt_text, str) or not prompt_text.strip():
                round1_state = extract_round1_state(output_record)
                prompt_text = round1_state.get("question") or ""
            review_applied = True
            review_counter[0] += 1
            review_result = run_borderline_review(prompt_text, model, temperature, resolve_api_key(api_key))
            parsed = review_result["parsed_output"]
            if should_continue_from_review(parsed):
                final_continue_decision = True
                final_decision_reason = "borderline_review_continue:" + (parsed.get("reason") or "no_reason")
            else:
                final_continue_decision = False
                final_decision_reason = "borderline_review_stop:" + (parsed.get("reason") or "no_reason")

    output_record["stage3_final_continue_decision"] = final_continue_decision
    output_record["stage3_final_decision"] = "continue" if final_continue_decision else "stop"
    output_record["stage3_final_decision_reason"] = final_decision_reason
    output_record["stage3_vrrs_br"] = {
        "policy": POLICY_NAME,
        "review_applied": review_applied,
        "source_decision_band": decision_band,
        "source_vrrs_continue_decision": vrrs_payload.get("final_continue_decision"),
        "borderline_prompt_text": prompt_text,
        "review_result": review_result,
        "final_continue_decision": final_continue_decision,
        "final_decision_reason": final_decision_reason,
    }
    return output_record


def run_vrrs_br(
    input_file: str | Path,
    labels_file: str | Path | None = None,
    output_file: str | Path | None = None,
    summary_file: str | Path | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    review_limit: int | None = None,
) -> Dict[str, Any]:
    input_path = Path(input_file).resolve()
    output_path = Path(output_file).resolve() if output_file else default_decision_output_path(input_path, POLICY_NAME)
    summary_path = Path(summary_file).resolve() if summary_file else default_summary_path(output_path)

    labels = load_scored_round1_labels(labels_file) if labels_file else {}
    review_counter = [0]
    decision_rows: List[Dict[str, Any]] = []
    for vrrs_record in iter_jsonl(input_path):
        example_id = vrrs_record.get("example_id")
        round1_correct = labels.get(example_id, {}).get("round1_correct")
        decision_rows.append(
            build_vrrs_br_record(
                vrrs_record=vrrs_record,
                round1_correct_override=round1_correct,
                model=model,
                temperature=temperature,
                api_key=api_key,
                review_limit=review_limit,
                review_counter=review_counter,
            )
        )

    write_jsonl(output_path, decision_rows)
    summary = build_summary(POLICY_NAME, input_path, output_path, labels_file, decision_rows)
    summary["model"] = model
    summary["reviewed_borderline_rows"] = review_counter[0]
    write_json(summary_path, summary)
    return {
        "policy": POLICY_NAME,
        "output_file": str(output_path),
        "summary_file": str(summary_path),
        "examples": len(decision_rows),
        "reviewed_borderline_rows": review_counter[0],
    }