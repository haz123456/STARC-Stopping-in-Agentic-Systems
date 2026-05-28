"""VRRS system: local round-1 risk gate with borderline treated as stop."""

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from stage3_common import (
    build_summary,
    clamp01,
    coerce_float,
    default_decision_output_path,
    default_summary_path,
    extract_round1_state,
    iter_jsonl,
    load_scored_round1_labels,
    normalize_text,
    to_band,
    tokenize_words,
    write_json,
    write_jsonl,
)


POLICY_NAME = "vrrs"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "with",
}

QUESTION_RISK_PATTERNS = {
    "what_year": re.compile(r"\bwhat year\b|\bin what year\b"),
    "who": re.compile(r"^\s*(who|whom)\b"),
    "when": re.compile(r"^\s*when\b"),
    "which_country": re.compile(r"^\s*which country\b"),
    "which_city": re.compile(r"^\s*which city\b"),
    "how_many": re.compile(r"^\s*how many\b"),
}

META_CLARIFY_PATTERNS = (
    "to better answer the question",
    "what specific information are you looking for",
    "can you please provide more context",
)


def content_tokens(text: Optional[str]) -> List[str]:
    return [token for token in tokenize_words(text) if token not in STOPWORDS]


def answer_type_flags(question: str, answer: str) -> Dict[str, bool]:
    q = normalize_text(question)
    a = normalize_text(answer)
    answer_tokens = tokenize_words(answer)
    return {
        "expects_year": bool(QUESTION_RISK_PATTERNS["what_year"].search(q) or QUESTION_RISK_PATTERNS["when"].search(q)),
        "expects_person": bool(QUESTION_RISK_PATTERNS["who"].search(q)),
        "expects_country": bool(QUESTION_RISK_PATTERNS["which_country"].search(q)),
        "expects_city": bool(QUESTION_RISK_PATTERNS["which_city"].search(q)),
        "expects_count": bool(QUESTION_RISK_PATTERNS["how_many"].search(q)),
        "single_factoid_question": bool(re.match(r"^\s*(who|when|where|which|what|how many)\b", q)),
        "answer_has_year": bool(re.search(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b", a)),
        "answer_has_digit": bool(re.search(r"\d", a)),
        "answer_is_short": len(answer_tokens) <= 2,
        "answer_has_and": bool(re.search(r"\band\b", a)),
        "answer_has_or": bool(re.search(r"\bor\b", a)),
        "answer_has_comma": "," in answer,
        "answer_has_hedge": bool(re.search(r"\b(possible|possibly|may be|might be|could be|likely)\b", a)),
    }


def estimate_competing_candidate_count(answer: str) -> int:
    text = answer or ""
    norm = normalize_text(text)
    quoted = re.findall(r'"([^"]+)"', text)
    if len(quoted) >= 2:
        return len(quoted)

    explicit_enumeration = bool(re.search(r"\b(possible answers?|either|both)\b", norm))
    clause_splits = [
        part.strip(" ,.;:")
        for part in re.split(r"\b(?:or|and)\b", text)
        if part.strip()
    ]
    long_clauses = [part for part in clause_splits if len(tokenize_words(part)) >= 2]
    if explicit_enumeration and len(long_clauses) >= 2:
        return len(long_clauses)
    return 1


def compute_round1_risk_flags(round1_state: Dict[str, Any]) -> Dict[str, Any]:
    question = round1_state.get("question") or ""
    answer = round1_state.get("round1_answer") or ""
    clarification = round1_state.get("clarification_state") or {}
    intermediate_question = clarification.get("intermediate_question") or ""
    intermediate_answer = clarification.get("intermediate_answer") or ""
    pinned_context = clarification.get("pinned_context") or ""
    verifier = (round1_state.get("saved_verifier_record") or {}).get("parsed_output") or {}

    q_norm = normalize_text(question)
    a_norm = normalize_text(answer)
    ia_norm = normalize_text(intermediate_answer)
    pc_norm = normalize_text(pinned_context)
    type_flags = answer_type_flags(question, answer)
    single_factoid = type_flags["single_factoid_question"]
    candidate_count = estimate_competing_candidate_count(answer)
    multi_candidate_answer = (
        single_factoid
        and (
            candidate_count >= 2
            or type_flags["answer_has_and"]
            or type_flags["answer_has_or"]
            or type_flags["answer_has_comma"]
            or type_flags["answer_has_hedge"]
        )
    )
    year_mismatch = (type_flags["expects_year"] or type_flags["expects_count"]) and not (
        type_flags["answer_has_year"] or type_flags["answer_has_digit"]
    )
    clue_echo = type_flags["expects_country"] and type_flags["answer_is_short"] and a_norm and a_norm in q_norm
    answer_only_in_intermediate = bool(a_norm) and a_norm in ia_norm and a_norm not in pc_norm
    answer_missing_from_pinned_context = bool(a_norm) and a_norm not in pc_norm
    clarification_meta = any(pattern in normalize_text(intermediate_question) for pattern in META_CLARIFY_PATTERNS)
    intermediate_meta = any(pattern in ia_norm for pattern in META_CLARIFY_PATTERNS)
    pinned_context_para_count = len(re.findall(r"<para\s+\d+>", pinned_context))
    pinned_context_broad = pinned_context_para_count >= 20

    support = clamp01(coerce_float(verifier.get("direct_support_prob")))
    complete = clamp01(coerce_float(verifier.get("answer_complete_prob")))
    gain = clamp01(coerce_float(verifier.get("refinement_expected_gain")))
    regression = clamp01(coerce_float(verifier.get("regression_risk_prob")))
    recommend_refine = verifier.get("recommend_refine") is True
    missing_present = verifier.get("missing_info_type") not in {None, "none"}

    d_band = to_band(support)
    c_band = to_band(complete)
    q_band = to_band(gain)
    r_band = to_band(regression)

    vrb_high_risk = (
        q_band in {"moderate", "high"}
        and (
            recommend_refine
            or missing_present
            or r_band in {"moderate", "high"}
        )
    )

    grounding_soft_active = (
        answer_only_in_intermediate
        or answer_missing_from_pinned_context
        or pinned_context_broad
    )
    high_confidence_low_grounding = (
        d_band == "high"
        and c_band == "high"
        and q_band == "low"
        and grounding_soft_active
    )

    hard_continue_reasons: List[str] = []
    if multi_candidate_answer:
        hard_continue_reasons.append("multi_candidate_answer")
    if year_mismatch:
        hard_continue_reasons.append("year_or_count_mismatch")
    if clue_echo:
        hard_continue_reasons.append("question_clue_echo")
    if vrb_high_risk:
        hard_continue_reasons.append("vrb_high_risk")

    soft_risk_reasons: List[str] = []
    if answer_only_in_intermediate:
        soft_risk_reasons.append("answer_only_in_intermediate")
    if answer_missing_from_pinned_context:
        soft_risk_reasons.append("answer_missing_from_pinned_context")
    if clarification_meta:
        soft_risk_reasons.append("clarification_meta")
    if intermediate_meta:
        soft_risk_reasons.append("intermediate_meta")
    if pinned_context_broad:
        soft_risk_reasons.append("pinned_context_broad")
    if high_confidence_low_grounding:
        soft_risk_reasons.append("high_confidence_low_grounding")

    return {
        "single_factoid_question": single_factoid,
        "multi_candidate_answer": multi_candidate_answer,
        "year_mismatch": year_mismatch,
        "clue_echo": clue_echo,
        "answer_only_in_intermediate": answer_only_in_intermediate,
        "answer_missing_from_pinned_context": answer_missing_from_pinned_context,
        "clarification_meta": clarification_meta,
        "intermediate_meta": intermediate_meta,
        "pinned_context_broad": pinned_context_broad,
        "candidate_count": candidate_count,
        "vrb_high_risk": vrb_high_risk,
        "high_confidence_low_grounding": high_confidence_low_grounding,
        "support": support,
        "complete": complete,
        "gain": gain,
        "regression": regression,
        "d_band": d_band,
        "c_band": c_band,
        "q_band": q_band,
        "r_band": r_band,
        "recommend_refine": recommend_refine,
        "missing_present": missing_present,
        "hard_continue_reasons": hard_continue_reasons,
        "soft_risk_reasons": soft_risk_reasons,
        "soft_risk_count": len(soft_risk_reasons),
    }


def build_borderline_prompt(round1_state: Dict[str, Any], risk_flags: Dict[str, Any]) -> str:
    verifier = (round1_state.get("saved_verifier_record") or {}).get("parsed_output") or {}
    lines = [
        "You are deciding whether a QA system should stop after round 1 or continue to round 2.",
        "Do not answer the question from scratch.",
        "Only decide whether the current round-1 answer is too risky to stop on.",
        "",
        f"Original question: {round1_state.get('question') or ''}",
        "",
        "Round-1 clarification question:",
        round1_state.get("clarification_state", {}).get("intermediate_question") or "",
        "",
        "Round-1 pinned context:",
        round1_state.get("clarification_state", {}).get("pinned_context") or "",
        "",
        "Round-1 clarification answer:",
        round1_state.get("clarification_state", {}).get("intermediate_answer") or "",
        "",
        "Round-1 provisional final answer:",
        round1_state.get("round1_answer") or "",
        "",
        "Saved verify-refine verifier summary:",
        json.dumps(
            {
                "direct_support_prob": verifier.get("direct_support_prob"),
                "answer_complete_prob": verifier.get("answer_complete_prob"),
                "refinement_expected_gain": verifier.get("refinement_expected_gain"),
                "recommend_refine": verifier.get("recommend_refine"),
                "missing_info_type": verifier.get("missing_info_type"),
                "regression_risk_prob": verifier.get("regression_risk_prob"),
            },
            ensure_ascii=False,
        ),
        "",
        "Local VRRS risk flags:",
        json.dumps(
            {
                "hard_continue_reasons": risk_flags.get("hard_continue_reasons"),
                "soft_risk_reasons": risk_flags.get("soft_risk_reasons"),
                "support": risk_flags.get("support"),
                "complete": risk_flags.get("complete"),
                "gain": risk_flags.get("gain"),
            },
            ensure_ascii=False,
        ),
        "",
        "Decision rule:",
        "- Continue only if there is a concrete reason the round-1 answer is likely wrong.",
        "- Under-grounding by itself is not enough.",
        "- Concrete reasons include contradiction, wrong entity, competing answer, answer-type mismatch, or clue echo.",
        "- If you cannot point to a concrete likely error, choose stop.",
        "",
        "Return exactly one JSON object:",
        "{",
        '  "continue_to_round2": true or false,',
        '  "false_answer_risk": number between 0.0 and 1.0,',
        '  "issue_type": "contradiction" | "wrong_entity" | "competing_answer" | "answer_type_mismatch" | "clue_echo" | "under_grounded_only" | "none",',
        '  "concrete_error_detected": true or false,',
        '  "reason": "short explanation"',
        "}",
    ]
    return "\n".join(lines)


def classify_round1(round1_state: Dict[str, Any]) -> Dict[str, Any]:
    risk_flags = compute_round1_risk_flags(round1_state)
    d_band = risk_flags["d_band"]
    c_band = risk_flags["c_band"]
    q_band = risk_flags["q_band"]
    soft_count = risk_flags["soft_risk_count"]

    if risk_flags["hard_continue_reasons"]:
        return {
            "continue_decision": True,
            "decision_band": "continue",
            "decision_reason": "hard_continue:" + ",".join(risk_flags["hard_continue_reasons"]),
            "risk_flags": risk_flags,
            "borderline_prompt_text": None,
        }

    if d_band == "high" and c_band == "high" and q_band == "low" and soft_count <= 2:
        return {
            "continue_decision": False,
            "decision_band": "stop",
            "decision_reason": "clean_stop",
            "risk_flags": risk_flags,
            "borderline_prompt_text": None,
        }

    if q_band in {"moderate", "high"} and soft_count >= 3:
        return {
            "continue_decision": True,
            "decision_band": "continue",
            "decision_reason": "gain_plus_accumulated_soft_risk",
            "risk_flags": risk_flags,
            "borderline_prompt_text": None,
        }

    if soft_count >= 2 and risk_flags["vrb_high_risk"]:
        return {
            "continue_decision": False,
            "decision_band": "borderline",
            "decision_reason": "borderline_vrb_high_risk_with_soft",
            "risk_flags": risk_flags,
            "borderline_prompt_text": build_borderline_prompt(round1_state, risk_flags),
        }

    return {
        "continue_decision": False,
        "decision_band": "stop",
        "decision_reason": "default_stop",
        "risk_flags": risk_flags,
        "borderline_prompt_text": None,
    }


def build_vrrs_record(source_record: Dict[str, Any], round1_correct: Optional[bool]) -> Dict[str, Any]:
    output_record = copy.deepcopy(source_record)
    round1_state = extract_round1_state(source_record)
    local_result = classify_round1(round1_state)

    final_reason = local_result["decision_reason"]
    if local_result["decision_band"] == "borderline":
        final_reason = "borderline_treated_as_stop"

    output_record["stage3_active_policy"] = POLICY_NAME
    output_record["stage3_round1_correct"] = round1_correct
    output_record["stage3_final_continue_decision"] = bool(
        local_result["continue_decision"] and local_result["decision_band"] != "borderline"
    )
    output_record["stage3_final_decision"] = (
        "continue" if output_record["stage3_final_continue_decision"] else "stop"
    )
    output_record["stage3_final_decision_reason"] = final_reason
    output_record["stage3_vrrs"] = {
        "policy": POLICY_NAME,
        "example_id": round1_state.get("example_id"),
        "question": round1_state.get("question"),
        "round1_answer": round1_state.get("round1_answer"),
        "local_continue_decision": local_result["continue_decision"],
        "decision_band": local_result["decision_band"],
        "decision_reason": local_result["decision_reason"],
        "final_continue_decision": output_record["stage3_final_continue_decision"],
        "final_decision_reason": final_reason,
        "risk_flags": local_result["risk_flags"],
        "borderline_prompt_text": local_result["borderline_prompt_text"],
    }
    return output_record


def run_vrrs(
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
        decision_rows.append(build_vrrs_record(source_record, round1_correct))

    write_jsonl(output_path, decision_rows)
    summary = build_summary(POLICY_NAME, input_path, output_path, labels_file, decision_rows)
    write_json(summary_path, summary)
    return {
        "policy": POLICY_NAME,
        "output_file": str(output_path),
        "summary_file": str(summary_path),
        "examples": len(decision_rows),
    }