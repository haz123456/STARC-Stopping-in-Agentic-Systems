import os
from collections import defaultdict
import random
import json
import time
import copy
import re
import string
import sys

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HELMET_DIR = os.path.join(CURRENT_DIR, "AgenticLU", "HELMET")
if HELMET_DIR not in sys.path:
    sys.path.append(HELMET_DIR)

try:
    from new_arguments import parse_arguments
except ImportError:
    from arguments import parse_arguments

try:
    from new_model_utils import load_LLM
except ImportError:
    from model_utils import load_LLM

from data import (
    load_data,
    ItemDataset,
)
from state_saving import (
    append_failure_trace,
    append_trace,
    atomic_write_json,
    build_compact_result_record,
    build_example_identity,
    build_full_log_record,
    build_output_paths,
    build_stage_record,
    cleanup_restart_files,
    compute_averaged_metrics,
    load_json_payload,
    save_compact_results,
)

import logging


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
PARA_ID_PATTERN = re.compile(r"<para\s+(\d+)>")
NO_GAP_LABELS = {
    "none",
    "no_gap",
    "no gap",
    "resolved",
    "fully resolved",
    "no unresolved gap",
    "no remaining gap",
    "nothing material",
    "n/a",
    "na",
    "null",
    "",
}
LOW_GAP_LABELS = {"low", "minor", "small", "limited"}
MEDIUM_GAP_LABELS = {"medium", "moderate", "partial"}
HIGH_GAP_LABELS = {"high", "large", "major", "severe", "critical"}


def tokenize_with_template(chat, tokenizer):
    chat_formatted = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    )
    tokenized_input = tokenizer(
        chat_formatted,
        return_tensors="pt",
        add_special_tokens=False
    )
    return tokenized_input


def mark_context(context_pieces):
    marked_context = ""
    for idx, context_piece in enumerate(context_pieces):
        marked_context += f"<para {idx}> {context_piece} </para {idx}>"
    return marked_context


def parse_max_len(max_length):
    if isinstance(max_length, str):
        return int(max_length.split(",")[0].strip())
    return int(max_length)


def clamp01(value):
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def coerce_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower().replace("%", "")
        if normalized == "":
            return None
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


def normalize_label(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    normalized = normalized.replace("-", " ").replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" .,:;!?\"'")
    return normalized


def truncate_text_to_tokens(text, tokenizer, max_tokens):
    tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized["input_ids"][0]
    if input_ids.size(0) <= max_tokens:
        return text
    return tokenizer.decode(input_ids[:max_tokens], skip_special_tokens=True)


def truncate_chat_to_budget(chat, tokenizer, max_input_tokens):
    chat_formatted = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    )
    tokenized_input = tokenizer(
        chat_formatted,
        return_tensors="pt",
        add_special_tokens=False
    )

    if tokenized_input.input_ids.size(1) <= max_input_tokens:
        return tokenized_input

    trimmed_chat = list(chat)

    while len(trimmed_chat) > 2:
        if trimmed_chat[0]["role"] == "system":
            drop_idx = 1
        else:
            drop_idx = 0

        trimmed_chat.pop(drop_idx)

        chat_formatted = tokenizer.apply_chat_template(
            trimmed_chat,
            tokenize=False,
            add_generation_prompt=True
        )
        tokenized_input = tokenizer(
            chat_formatted,
            return_tensors="pt",
            add_special_tokens=False
        )

        if tokenized_input.input_ids.size(1) <= max_input_tokens:
            return tokenized_input

    input_ids = tokenized_input.input_ids[:, -max_input_tokens:]
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def safe_generate(model, conversation, prompt_budget, example_idx, stage_name, **generate_kwargs):
    model_inputs = truncate_chat_to_budget(conversation, model.tokenizer, prompt_budget)
    output = model.generate(inputs=model_inputs, **generate_kwargs)
    if output is None:
        logger.info(
            f"skipping example {example_idx + 1} at stage '{stage_name}' because the model returned None"
        )
        return None
    return output


def clone_conversation(conversation):
    return copy.deepcopy(conversation)


def get_chat_text(chat, tokenizer):
    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    )


def parse_index_list(index_string):
    """
    Supports:
      "1,2,3"
      "1-5"
      "1-5,8,10-12"
    Returns sorted unique indices.
    """
    if index_string is None:
        return None

    index_string = str(index_string).strip()
    if index_string == "":
        return None

    indices = set()
    parts = [p.strip() for p in index_string.split(",") if p.strip()]

    for part in parts:
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start_i = int(start_s.strip())
            end_i = int(end_s.strip())
            if end_i < start_i:
                raise ValueError(f"invalid range '{part}' because end < start")
            for i in range(start_i, end_i + 1):
                indices.add(i)
        else:
            indices.add(int(part))

    return sorted(indices)


def load_indices_from_file(path):
    if path is None:
        return None

    indices = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            indices.append(int(line))

    return sorted(set(indices))


def determine_selected_indices(raw_loaded, args):
    explicit_indices = []

    cli_indices = parse_index_list(getattr(args, "example_indices", None))
    if cli_indices is not None:
        explicit_indices.extend(cli_indices)

    file_indices = load_indices_from_file(getattr(args, "example_indices_file", None))
    if file_indices is not None:
        explicit_indices.extend(file_indices)

    if len(explicit_indices) > 0:
        explicit_indices = sorted(set(explicit_indices))
        selected = []
        for idx in explicit_indices:
            if idx < 0 or idx >= raw_loaded:
                raise ValueError(
                    f"requested example index {idx} is out of bounds for dataset of size {raw_loaded}"
                )
            selected.append(idx)
        return selected

    start_idx = getattr(args, "start_idx", None)
    end_idx = getattr(args, "end_idx", None)

    if start_idx is not None or end_idx is not None:
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = raw_loaded - 1

        if start_idx < 0 or end_idx < 0:
            raise ValueError("start_idx and end_idx must be non-negative")
        if end_idx < start_idx:
            raise ValueError("end_idx must be >= start_idx")
        if start_idx >= raw_loaded:
            raise ValueError(
                f"start_idx {start_idx} is out of bounds for dataset of size {raw_loaded}"
            )

        end_idx = min(end_idx, raw_loaded - 1)
        return list(range(start_idx, end_idx + 1))

    return list(range(raw_loaded))


def apply_selected_indices(data, selected_indices):
    if len(selected_indices) == len(data["data"]):
        return data

    logger.info(
        f"selecting {len(selected_indices)} examples from loaded dataset of size {len(data['data'])}"
    )

    data["data"] = data["data"].select(selected_indices)
    return data


def build_subset_tag(args, selected_indices, raw_loaded):
    if len(selected_indices) == raw_loaded:
        return "subsetall"

    if getattr(args, "example_indices", None):
        preview = "-".join(str(i) for i in selected_indices[:10])
        if len(selected_indices) > 10:
            preview += f"-plus{len(selected_indices) - 10}"
        return f"indices{preview}_n{len(selected_indices)}"

    if getattr(args, "example_indices_file", None):
        file_stem = os.path.splitext(os.path.basename(args.example_indices_file))[0]
        return f"indicesfile_{file_stem}_n{len(selected_indices)}"

    start_idx = getattr(args, "start_idx", None)
    end_idx = getattr(args, "end_idx", None)
    if start_idx is not None or end_idx is not None:
        return f"range{selected_indices[0]}-{selected_indices[-1]}_n{len(selected_indices)}"

    return f"subset_n{len(selected_indices)}"


def build_run_signature(args, dataset, test_file):
    return {
        "dataset": dataset,
        "test_file": test_file,
        "model_name_or_path": args.model_name_or_path,
        "seed": args.seed,
        "input_max_length": args.input_max_length,
        "generation_max_length": args.generation_max_length,
        "max_test_samples": args.max_test_samples,
        "num_clarification_rounds": args.num_clarification_rounds,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "shots": args.shots,
        "subset_tag": getattr(args, "_subset_tag", None),
    }


def should_resume_from_payload(payload, expected_signature):
    if payload is None:
        return False

    saved_args = payload.get("args", {})
    saved_signature = {
        "dataset": saved_args.get("datasets"),
        "test_file": saved_args.get("test_files"),
        "model_name_or_path": saved_args.get("model_name_or_path"),
        "seed": saved_args.get("seed"),
        "input_max_length": saved_args.get("input_max_length"),
        "generation_max_length": saved_args.get("generation_max_length"),
        "max_test_samples": saved_args.get("max_test_samples"),
        "num_clarification_rounds": saved_args.get("num_clarification_rounds"),
        "temperature": saved_args.get("temperature"),
        "top_p": saved_args.get("top_p"),
        "do_sample": saved_args.get("do_sample"),
        "shots": saved_args.get("shots"),
        "subset_tag": saved_args.get("_subset_tag"),
    }

    return saved_signature == expected_signature


def truncate_existing_state_to_cap(results, metrics, cap):
    if cap is None:
        return results, metrics

    if len(results) <= cap:
        return results, metrics

    logger.warning(
        f"saved progress has {len(results)} examples but current effective_size={cap}; truncating saved state"
    )

    truncated_results = results[:cap]
    truncated_metrics = defaultdict(list)

    for k, v in metrics.items():
        truncated_metrics[k] = list(v)[:cap]

    return truncated_results, truncated_metrics


def normalize_text_for_similarity(text):
    if text is None:
        return ""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def compute_answer_stability(previous_answer, current_answer):
    prev = normalize_text_for_similarity(previous_answer)
    curr = normalize_text_for_similarity(current_answer)

    if prev == "" and curr == "":
        return 1.0
    if prev == "" or curr == "":
        return 0.0
    if prev == curr:
        return 1.0
    if prev in curr or curr in prev:
        return 0.9

    prev_tokens = set(prev.split())
    curr_tokens = set(curr.split())
    if len(prev_tokens) == 0 and len(curr_tokens) == 0:
        return 1.0
    if len(prev_tokens | curr_tokens) == 0:
        return 0.0
    return len(prev_tokens & curr_tokens) / len(prev_tokens | curr_tokens)


def extract_para_ids(pinned_context):
    if pinned_context is None:
        return set()
    return {int(match) for match in PARA_ID_PATTERN.findall(pinned_context)}


def compute_evidence_stability(previous_pinned_context, current_pinned_context):
    prev_ids = extract_para_ids(previous_pinned_context)
    curr_ids = extract_para_ids(current_pinned_context)

    if len(prev_ids) == 0 and len(curr_ids) == 0:
        return 1.0
    if len(prev_ids | curr_ids) == 0:
        return 0.0
    return len(prev_ids & curr_ids) / len(prev_ids | curr_ids)


def build_final_prompt(test_item):
    if "options" in test_item.keys():
        if isinstance(test_item["options"], list):
            choices = ", ".join(test_item["options"])
        else:
            choices = test_item["options"]
        return (
            "Now, let's answer the final question"
            + test_item["question"]
            + " The choices are: "
            + choices
            + "\n\nChoose the correct option!"
        )

    return (
        "Now, let's answer the final question, be concise in your answer."
        + test_item["question"]
    )


def extract_json_block(text):
    if text is None:
        return None

    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).strip()

    match = JSON_BLOCK_PATTERN.search(text)
    if match is None:
        return None
    return match.group(0)


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "stop"}:
            return True
        if normalized in {"false", "no", "n", "0", "continue"}:
            return False
    return None


def normalize_gap_severity(value):
    normalized = normalize_label(value)
    if normalized is None:
        return None
    if normalized in NO_GAP_LABELS:
        return "none"
    if normalized in LOW_GAP_LABELS:
        return "low"
    if normalized in MEDIUM_GAP_LABELS:
        return "medium"
    if normalized in HIGH_GAP_LABELS:
        return "high"
    return normalized


def severity_rank(value):
    normalized = normalize_gap_severity(value)
    mapping = {
        "none": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
    }
    return mapping.get(normalized)


def normalize_action(value):
    normalized = normalize_label(value)
    if normalized is None:
        return None
    if normalized in {"stop", "halt", "end", "finish"}:
        return "stop"
    if normalized in {"continue", "go on", "another round", "need another round"}:
        return "continue"
    return normalized


def support_score_to_prob(support_score):
    score = coerce_float(support_score)
    if score is None:
        return None
    if score > 1.0:
        return clamp01(score / 5.0)
    return clamp01(score)


def parse_stop_gate_output(raw_text):
    parsed = {
        "answer_supported": None,
        "support_score": None,
        "residual_gap_type": None,
        "residual_gap_text": None,
        "another_round_useful": None,
        "confidence": None,
        "stop_decision": None,
        "current_answer_correct_prob": None,
        "remaining_gap_severity": None,
        "remaining_gap_type": None,
        "next_round_improvement_prob": None,
        "improvement_if_next_round_helps": None,
        "recommended_action": None,
        "brief_rationale": None,
        "parse_error": None,
        "raw_text": raw_text,
    }

    json_block = extract_json_block(raw_text)
    if json_block is None:
        parsed["parse_error"] = "no_json_found"
        return parsed

    try:
        data = json.loads(json_block)
    except Exception as e:
        parsed["parse_error"] = f"json_decode_error: {e}"
        return parsed

    parsed["answer_supported"] = coerce_bool(data.get("answer_supported"))
    parsed["support_score"] = coerce_float(data.get("support_score"))
    parsed["residual_gap_type"] = normalize_label(data.get("residual_gap_type"))
    parsed["residual_gap_text"] = data.get("residual_gap_text")
    parsed["another_round_useful"] = coerce_bool(data.get("another_round_useful"))
    parsed["confidence"] = clamp01(coerce_float(data.get("confidence")))
    parsed["stop_decision"] = normalize_action(data.get("stop_decision"))
    parsed["current_answer_correct_prob"] = clamp01(
        coerce_float(data.get("current_answer_correct_prob"))
    )
    parsed["remaining_gap_severity"] = normalize_gap_severity(
        data.get("remaining_gap_severity")
    )
    parsed["remaining_gap_type"] = normalize_label(
        data.get("remaining_gap_type", data.get("residual_gap_type"))
    )
    parsed["next_round_improvement_prob"] = clamp01(
        coerce_float(data.get("next_round_improvement_prob"))
    )
    parsed["improvement_if_next_round_helps"] = clamp01(
        coerce_float(data.get("improvement_if_next_round_helps"))
    )
    parsed["recommended_action"] = normalize_action(
        data.get("recommended_action", data.get("stop_decision"))
    )
    parsed["brief_rationale"] = data.get("brief_rationale", data.get("residual_gap_text"))

    if parsed["remaining_gap_severity"] is None and parsed["residual_gap_type"] is not None:
        parsed["remaining_gap_severity"] = normalize_gap_severity(parsed["residual_gap_type"])

    return parsed


def residual_gap_is_none(residual_gap_type):
    normalized = normalize_gap_severity(residual_gap_type)
    if normalized is None:
        return False
    return normalized == "none"


def estimate_current_correct_prob(parsed_gate):
    candidates = [
        parsed_gate.get("current_answer_correct_prob"),
        parsed_gate.get("confidence"),
    ]

    support_prob = support_score_to_prob(parsed_gate.get("support_score"))
    if support_prob is not None:
        if parsed_gate.get("answer_supported") is True:
            candidates.append(max(support_prob, 0.5))
        else:
            candidates.append(min(support_prob, 0.49))

    candidates = [clamp01(x) for x in candidates if x is not None]
    if len(candidates) == 0:
        return None
    return max(candidates)


def estimate_next_round_improvement_prob(parsed_gate):
    next_round_prob = clamp01(parsed_gate.get("next_round_improvement_prob"))
    if next_round_prob is not None:
        return next_round_prob

    another_round_useful = parsed_gate.get("another_round_useful")
    severity = normalize_gap_severity(parsed_gate.get("remaining_gap_severity"))
    if severity is None:
        severity = normalize_gap_severity(parsed_gate.get("residual_gap_type"))

    if another_round_useful is True:
        if severity == "high":
            return 0.65
        if severity == "medium":
            return 0.45
        if severity == "low":
            return 0.25
        return 0.35

    if another_round_useful is False:
        if severity == "none":
            return 0.05
        return 0.10

    if severity == "high":
        return 0.55
    if severity == "medium":
        return 0.35
    if severity == "low":
        return 0.15
    if severity == "none":
        return 0.05
    return None


def estimate_improvement_magnitude(parsed_gate, current_correct_prob):
    magnitude = clamp01(parsed_gate.get("improvement_if_next_round_helps"))
    if magnitude is not None:
        return magnitude

    severity = normalize_gap_severity(parsed_gate.get("remaining_gap_severity"))
    if severity is None:
        severity = normalize_gap_severity(parsed_gate.get("residual_gap_type"))

    if severity == "high":
        return 0.50
    if severity == "medium":
        return 0.30
    if severity == "low":
        return 0.15
    if severity == "none":
        return 0.05

    if current_correct_prob is not None:
        return clamp01(1.0 - current_correct_prob)
    return None


def should_stop_after_round(args, round_num, parsed_gate, answer_stability=None, evidence_stability=None):
    if parsed_gate is None:
        return {
            "should_stop": False,
            "stop_reason": "no_gate_output",
            "current_correct_prob": None,
            "next_round_improvement_prob": None,
            "improvement_if_next_round_helps": None,
            "expected_value_next_round": None,
            "remaining_gap_severity": None,
            "recommended_action": None,
            "answer_stability": answer_stability,
            "evidence_stability": evidence_stability,
        }

    if parsed_gate.get("parse_error") is not None:
        return {
            "should_stop": False,
            "stop_reason": parsed_gate["parse_error"],
            "current_correct_prob": None,
            "next_round_improvement_prob": None,
            "improvement_if_next_round_helps": None,
            "expected_value_next_round": None,
            "remaining_gap_severity": None,
            "recommended_action": parsed_gate.get("recommended_action"),
            "answer_stability": answer_stability,
            "evidence_stability": evidence_stability,
        }

    current_correct_prob = estimate_current_correct_prob(parsed_gate)
    next_round_improvement_prob = estimate_next_round_improvement_prob(parsed_gate)
    improvement_if_next_round_helps = estimate_improvement_magnitude(
        parsed_gate,
        current_correct_prob,
    )
    expected_value_next_round = None
    if next_round_improvement_prob is not None and improvement_if_next_round_helps is not None:
        expected_value_next_round = next_round_improvement_prob * improvement_if_next_round_helps

    remaining_gap_severity = normalize_gap_severity(parsed_gate.get("remaining_gap_severity"))
    if remaining_gap_severity is None:
        remaining_gap_severity = normalize_gap_severity(parsed_gate.get("residual_gap_type"))
    gap_rank = severity_rank(remaining_gap_severity)

    recommended_action = normalize_action(
        parsed_gate.get("recommended_action", parsed_gate.get("stop_decision"))
    )
    answer_supported = parsed_gate.get("answer_supported") is True
    support_score = parsed_gate.get("support_score")
    support_ok = support_score is not None and support_score >= args.stop_support_threshold

    answer_stability_ok = (
        isinstance(answer_stability, (int, float))
        and answer_stability >= args.stop_answer_stability_threshold
    )
    evidence_stability_ok = (
        isinstance(evidence_stability, (int, float))
        and evidence_stability >= args.stop_evidence_stability_threshold
    )
    stable_enough = round_num == 1 or (answer_stability_ok and evidence_stability_ok)

    continue_threshold = (
        args.continue_improvement_threshold_round1
        if round_num == 1
        else args.continue_improvement_threshold_round2
    )

    should_continue = False
    continue_reasons = []

    if gap_rank is not None and gap_rank >= severity_rank("medium"):
        should_continue = True
        continue_reasons.append("material_gap_remaining")
    if (
        next_round_improvement_prob is not None
        and next_round_improvement_prob >= continue_threshold
    ):
        should_continue = True
        continue_reasons.append("next_round_improvement_prob_high")
    if (
        expected_value_next_round is not None
        and expected_value_next_round > args.stop_expected_gain_threshold
    ):
        should_continue = True
        continue_reasons.append("expected_gain_above_stop_threshold")
    if recommended_action == "continue":
        should_continue = True
        continue_reasons.append("gate_recommends_continue")

    can_stop_on_quality = (
        current_correct_prob is not None
        and current_correct_prob >= args.stop_current_correct_threshold
        and support_ok
        and answer_supported
    )
    low_gap = gap_rank is not None and gap_rank <= severity_rank("low")
    low_expected_gain = (
        expected_value_next_round is not None
        and expected_value_next_round <= args.stop_expected_gain_threshold
    )

    should_stop = False
    stop_reason = "insufficient_stop_evidence"

    if can_stop_on_quality and low_gap and low_expected_gain and stable_enough:
        should_stop = True
        stop_reason = "high_correctness_low_expected_gain"
    elif (
        can_stop_on_quality
        and remaining_gap_severity == "none"
        and recommended_action == "stop"
        and stable_enough
    ):
        should_stop = True
        stop_reason = "gate_stop_with_no_remaining_gap"
    elif (
        round_num > 1
        and can_stop_on_quality
        and answer_stability_ok
        and evidence_stability_ok
        and (
            low_expected_gain
            or recommended_action == "stop"
        )
    ):
        should_stop = True
        stop_reason = "stable_answer_and_evidence"

    if should_continue:
        should_stop = False
        if len(continue_reasons) > 0:
            stop_reason = "continue:" + ",".join(continue_reasons)

    return {
        "should_stop": should_stop,
        "stop_reason": stop_reason,
        "current_correct_prob": current_correct_prob,
        "next_round_improvement_prob": next_round_improvement_prob,
        "improvement_if_next_round_helps": improvement_if_next_round_helps,
        "expected_value_next_round": expected_value_next_round,
        "remaining_gap_severity": remaining_gap_severity,
        "recommended_action": recommended_action,
        "answer_stability": answer_stability,
        "evidence_stability": evidence_stability,
    }


def build_stop_gate_prompt(test_item, current_round_state, previous_round_state=None, provisional_final_answer=None):
    prompt_lines = [
        "You are an adaptive stopping assessor for a multi-round clarification workflow.",
        "Your job is to estimate the marginal value of paying for one more clarification round.",
        "Return JSON only and do not include any explanation outside the JSON object.",
        "",
        f"Original question: {test_item['question']}",
        f"Current round: {current_round_state['round']}",
        "",
        "Current clarification question:",
        current_round_state["intermediate_question"],
        "",
        "Current pointed-back evidence:",
        current_round_state["pinned_context"],
        "",
        "Current clarification answer:",
        current_round_state["intermediate_answer"],
        "",
        "Current provisional final answer:",
        provisional_final_answer if provisional_final_answer is not None else "",
        "",
    ]

    if previous_round_state is not None:
        prompt_lines.extend([
            "Previous round provisional final answer:",
            previous_round_state.get("provisional_final_answer", ""),
            "",
            f"Answer stability from previous round to current round: {previous_round_state.get('answer_stability')}",
            f"Evidence stability from previous round to current round: {previous_round_state.get('evidence_stability')}",
            "",
        ])

    prompt_lines.extend([
        "Judge the following:",
        "1. How likely is the current provisional final answer already correct?",
        "2. Is there any unresolved information gap remaining, and how severe is it?",
        "3. If we pay for one more clarification round, how likely is it to improve correctness?",
        "4. If another round helps, how large would that improvement likely be?",
        "5. If a previous round exists, use the answer/evidence stability as supporting evidence.",
        "",
        "Return exactly one JSON object with this schema:",
        "{",
        '  "answer_supported": true or false,',
        '  "support_score": integer from 1 to 5,',
        '  "current_answer_correct_prob": number between 0.0 and 1.0,',
        '  "remaining_gap_severity": "none", "low", "medium", or "high",',
        '  "remaining_gap_type": "none" or a short label,',
        '  "residual_gap_text": "short explanation",',
        '  "another_round_useful": true or false,',
        '  "next_round_improvement_prob": number between 0.0 and 1.0,',
        '  "improvement_if_next_round_helps": number between 0.0 and 1.0,',
        '  "confidence": number between 0.0 and 1.0,',
        '  "recommended_action": "stop" or "continue",',
        '  "brief_rationale": "short explanation"'
        "}",
    ])

    return "\n".join(prompt_lines)


def build_initial_conversation(system_prompt, marked_context, test_item):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": marked_context + "\n\n" + test_item["question"]},
    ]


def build_provisional_final_answer(
    model,
    conversation,
    test_item,
    prompt_budget,
    idx,
    round_num,
    stage_outputs,
):
    final_prompt = build_final_prompt(test_item)
    provisional_conversation = clone_conversation(conversation)
    provisional_conversation.append({"role": "user", "content": final_prompt})
    conversation_before = clone_conversation(provisional_conversation)

    output = safe_generate(
        model,
        provisional_conversation,
        prompt_budget,
        idx,
        f"provisional_final_answer_round_{round_num}"
    )

    if output is None:
        stage_outputs.append(build_stage_record(
            stage_name=f"provisional_final_answer_round_{round_num}",
            round_idx=round_num,
            prompt_text=final_prompt,
            model_output=None,
            conversation_before=conversation_before,
            conversation_after=provisional_conversation,
        ))
        return None

    provisional_answer = output["output"]
    provisional_conversation.append({"role": "assistant", "content": provisional_answer})
    output_logged = copy.deepcopy(output)
    stage_outputs.append(build_stage_record(
        stage_name=f"provisional_final_answer_round_{round_num}",
        round_idx=round_num,
        prompt_text=final_prompt,
        model_output=output_logged,
        conversation_before=conversation_before,
        conversation_after=provisional_conversation,
    ))

    return {
        "prompt": final_prompt,
        "answer": provisional_answer,
        "generation": copy.deepcopy(output),
        "conversation_after": clone_conversation(provisional_conversation),
        "input_text": get_chat_text(conversation_before, model.tokenizer),
    }


def run_stop_gate(
    args,
    model,
    prompt_budget,
    idx,
    round_num,
    stage_outputs,
    provisional_final,
    current_round_state,
    previous_round_state=None,
):
    gate_prompt = build_stop_gate_prompt(
        test_item=current_round_state["test_item"],
        current_round_state=current_round_state,
        previous_round_state=previous_round_state,
        provisional_final_answer=provisional_final["answer"],
    )

    gate_conversation = clone_conversation(provisional_final["conversation_after"])
    gate_conversation.append({"role": "user", "content": gate_prompt})
    conversation_before = clone_conversation(gate_conversation)

    output = safe_generate(
        model,
        gate_conversation,
        prompt_budget,
        idx,
        f"stop_gate_round_{round_num}"
    )

    if output is None:
        stage_outputs.append(build_stage_record(
            stage_name=f"stop_gate_round_{round_num}",
            round_idx=round_num,
            prompt_text=gate_prompt,
            model_output=None,
            conversation_before=conversation_before,
            conversation_after=gate_conversation,
        ))
        return None

    parsed_gate = parse_stop_gate_output(output["output"])
    gate_conversation.append({"role": "assistant", "content": output["output"]})

    output_logged = copy.deepcopy(output)
    output_logged["parsed_output"] = parsed_gate

    stage_outputs.append(build_stage_record(
        stage_name=f"stop_gate_round_{round_num}",
        round_idx=round_num,
        prompt_text=gate_prompt,
        model_output=output_logged,
        conversation_before=conversation_before,
        conversation_after=gate_conversation,
    ))

    answer_stability = None
    evidence_stability = None
    if previous_round_state is not None:
        answer_stability = previous_round_state.get("answer_stability")
        evidence_stability = previous_round_state.get("evidence_stability")

    policy_decision = should_stop_after_round(
        args=args,
        round_num=round_num,
        parsed_gate=parsed_gate,
        answer_stability=answer_stability,
        evidence_stability=evidence_stability,
    )

    return {
        "round": round_num,
        "raw_output": output["output"],
        "parsed_output": parsed_gate,
        "should_stop": policy_decision["should_stop"],
        "stop_reason": policy_decision["stop_reason"],
        "policy_decision": policy_decision,
        "input_len": output.get("input_len"),
        "output_len": output.get("output_len"),
        "input_text": output.get("input_text"),
    }

def rebuild_metrics_from_results(results):
    metrics = defaultdict(list)
    for record in results:
        metric_values = record.get("metric_values", {})
        for k, v in metric_values.items():
            if isinstance(v, (int, float, np.bool_)):
                metrics[k].append(v)
        if record.get("input_len") is not None:
            metrics["input_len"].append(record.get("input_len"))
        if record.get("output_len") is not None:
            metrics["output_len"].append(record.get("output_len"))
    return metrics
def load_resume_results(args, output_path, expected_signature):
    resume_path = getattr(args, "resume_from_path", None) or output_path
    candidate_paths = [resume_path]
    if resume_path == output_path:
        candidate_paths.append(output_path + ".bak")

    for path in candidate_paths:
        if path is None or not os.path.exists(path):
            continue

        try:
            payload = load_json_payload(path)
        except Exception as e:
            logger.warning(f"failed to load resume payload from {path}: {e}")
            continue

        if not should_resume_from_payload(payload, expected_signature):
            logger.warning(
                f"resume payload at {path} does not match current run signature; ignoring it"
            )
            continue

        existing_results = payload.get("data", [])
        if not isinstance(existing_results, list):
            logger.warning(f"resume payload at {path} has non-list data field; ignoring it")
            continue

        logger.info(f"loaded {len(existing_results)} completed examples from compact results file {path}")
        return existing_results, path

    logger.info("no matching compact resume file found, starting from scratch")
    return [], None


def run_test(args, model, dataset, test_file, demo_file):
    logger.info(f"running test on {dataset} with test {test_file} and demo {demo_file}")

    system_prompt = (
        "You are an AI assistant specialized in long context reasoning. "
        "Analyze information thoroughly while maintaining clarity and focus. "
        "Track the full context of conversations, building connections between concepts "
        "and flagging when context review is needed. Break down complex problems into "
        "components, showing your reasoning steps and stating key assumptions. "
        "Structure your responses with clear headers and periodic summaries. "
        "Present evidence for your conclusions, acknowledge uncertainties, and request "
        "clarification when needed. Keep your analysis organized, explicit, and focused "
        "on addressing the core question."
    )

    random.seed(args.seed)
    data = load_data(args, dataset, test_file, demo_file)

    full_loaded_dataset = data["data"]
    raw_loaded = len(full_loaded_dataset)
    logger.info(f"loaded {raw_loaded} samples from {dataset}")

    selected_indices = determine_selected_indices(raw_loaded, args)
    args._subset_tag = build_subset_tag(args, selected_indices, raw_loaded)

    data = apply_selected_indices(data, selected_indices)

    no_explicit_subset = (
        getattr(args, "example_indices", None) is None
        and getattr(args, "example_indices_file", None) is None
        and getattr(args, "start_idx", None) is None
        and getattr(args, "end_idx", None) is None
    )

    is_grouped_longqa = any(x in dataset for x in ["nq", "triviaqa", "hotpotqa", "popqa"])

    if is_grouped_longqa and no_explicit_subset:
        current_rows = len(data["data"])
        if current_rows != raw_loaded:
            logger.warning(
                f"detected unexpected shrink for grouped QA dataset {dataset}: "
                f"raw_loaded={raw_loaded}, current_rows={current_rows}. "
                "Restoring full loaded dataset before resume."
            )
            data["data"] = full_loaded_dataset
            selected_indices = list(range(raw_loaded))
            args._subset_tag = "subsetall"

    effective_size = len(data["data"])
    logger.info(f"effective evaluation size for {dataset}: {effective_size}")
    logger.info(f"subset tag for this run: {args._subset_tag}")
    logger.info(
        f"dataset size debug -> raw_loaded={raw_loaded}, "
        f"selected_rows={len(selected_indices)}, "
        f"no_explicit_subset={no_explicit_subset}"
    )

    try:
        if hasattr(data["data"], "column_names") and "question" in data["data"].column_names:
            unique_questions = len(set(data["data"]["question"]))
            logger.info(
                f"dataset summary for {dataset}: unique_questions={unique_questions}, total_rows={effective_size}"
            )
    except Exception as e:
        logger.warning(f"could not compute unique question count: {e}")

    output_path, score_path, progress_path, trace_path = build_output_paths(args, dataset, test_file)

    if getattr(args, "complete_restart", False):
        cleanup_restart_files(output_path, progress_path, score_path, trace_path)

    dataloader = DataLoader(
        ItemDataset(data, model, model.tokenizer),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=args.num_workers if not args.debug else 0,
    )

    expected_signature = build_run_signature(args, dataset, test_file)
    results, loaded_from = load_resume_results(
        args=args,
        output_path=output_path,
        expected_signature=expected_signature,
    )
    metrics = rebuild_metrics_from_results(results)
    completed_ids = {
        record.get("example_id")
        for record in results
        if isinstance(record, dict) and record.get("example_id")
    }

    logger.info(
        f"resume debug -> output_path={output_path}, progress_path={progress_path}, "
        f"loaded_resume_count={len(results)}, loaded_from={loaded_from}, effective_size={effective_size}"
    )

    if len(completed_ids) >= effective_size:
        logger.info(
            f"resume file already contains {len(completed_ids)} completed examples, covering effective_size={effective_size}"
        )
        return output_path

    start_time = time.time()

    max_input_tokens = parse_max_len(model.max_length)
    gen_tokens = parse_max_len(model.generation_max_length)
    prompt_budget = max_input_tokens - gen_tokens

    save_every = getattr(args, "save_every", 1)
    if save_every < 1:
        save_every = 1

    logger.info(
        f"token budget -> max_input_tokens={max_input_tokens}, "
        f"generation_max_tokens={gen_tokens}, prompt_budget={prompt_budget}"
    )
    logger.info(f"compact results path: {output_path}")
    logger.info(f"full log path: {progress_path}")
    logger.info(f"event log path: {trace_path}")
    logger.info(f"using num_clarification_rounds={args.num_clarification_rounds}")
    logger.info(f"resuming from completed example count: {len(completed_ids)}")

    try:
        with torch.inference_mode():
            for idx, entry in enumerate(tqdm(dataloader, total=effective_size)):
                test_item = entry[0]
                source_idx = selected_indices[idx]
                identity = build_example_identity(dataset, source_idx, test_item)
                if identity["example_id"] in completed_ids:
                    continue
                example_start = time.time()

                context_budget = max(1024, prompt_budget - 1024)
                truncated_context = truncate_text_to_tokens(
                    test_item["context"],
                    model.tokenizer,
                    context_budget
                )

                tokenized_context = model.tokenizer(
                    truncated_context,
                    return_tensors="pt",
                    add_special_tokens=False
                )
                context_token_count = tokenized_context["input_ids"].size(1)

                context_len = 512
                context_pieces = [
                    model.tokenizer.decode(
                        tokenized_context["input_ids"][0][i:i + context_len],
                        skip_special_tokens=True
                    )
                    for i in range(0, len(tokenized_context["input_ids"][0]), context_len)
                ]
                marked_context = mark_context(context_pieces)

                conversation = build_initial_conversation(system_prompt, marked_context, test_item)

                clarification_rounds = []
                stage_outputs = []
                round_failed = False
                termination_check = None
                input_text_final_answer = None
                input_text_termination_check = None
                adaptive_stop_records = []
                provisional_answers_by_round = []
                stopped_early = False
                stopped_after_round = None
                num_rounds_used = 0

                for round_idx in range(args.num_clarification_rounds):
                    round_num = round_idx + 1

                    clarify_question_prompt = (
                        f'\nIn order to answer this question "{test_item["question"]}", '
                        f"ask one question about what do you want to know in order to better answer it."
                    )

                    conversation.append({"role": "user", "content": clarify_question_prompt})
                    conversation_before = clone_conversation(conversation)

                    output = safe_generate(
                        model,
                        conversation,
                        prompt_budget,
                        idx,
                        f"clarify_question_round_{round_num}"
                    )
                    if output is None:
                        append_failure_trace(
                            trace_path,
                            idx,
                            source_idx,
                            test_item,
                            f"clarify_question_round_{round_num}",
                            conversation,
                            stage_outputs,
                        )
                        round_failed = True
                        break

                    clarify_output = output["output"]
                    conversation.append({"role": "assistant", "content": clarify_output})
                    stage_outputs.append(build_stage_record(
                        stage_name=f"clarify_question_round_{round_num}",
                        round_idx=round_num,
                        prompt_text=clarify_question_prompt,
                        model_output=output,
                        conversation_before=conversation_before,
                        conversation_after=conversation,
                    ))

                    find_context_prompt = "Help me find relevant context to answer the previous clarifying quesiton."
                    conversation.append({
                        "role": "user",
                        "content": find_context_prompt
                    })
                    conversation_before = clone_conversation(conversation)

                    output = safe_generate(
                        model,
                        conversation,
                        prompt_budget,
                        idx,
                        f"find_relevant_context_round_{round_num}"
                    )
                    if output is None:
                        append_failure_trace(
                            trace_path,
                            idx,
                            source_idx,
                            test_item,
                            f"find_relevant_context_round_{round_num}",
                            conversation,
                            stage_outputs,
                        )
                        round_failed = True
                        break

                    pinned_context = output["output"]
                    conversation.append({"role": "assistant", "content": pinned_context})
                    stage_outputs.append(build_stage_record(
                        stage_name=f"find_relevant_context_round_{round_num}",
                        round_idx=round_num,
                        prompt_text=find_context_prompt,
                        model_output=output,
                        conversation_before=conversation_before,
                        conversation_after=conversation,
                    ))

                    answer_clarify_prompt = "Based on the relevant context, answer the previous clarifying question."
                    conversation.append({
                        "role": "user",
                        "content": answer_clarify_prompt
                    })
                    conversation_before = clone_conversation(conversation)

                    output = safe_generate(
                        model,
                        conversation,
                        prompt_budget,
                        idx,
                        f"answer_clarifying_question_round_{round_num}"
                    )
                    if output is None:
                        append_failure_trace(
                            trace_path,
                            idx,
                            source_idx,
                            test_item,
                            f"answer_clarifying_question_round_{round_num}",
                            conversation,
                            stage_outputs,
                        )
                        round_failed = True
                        break

                    intermediate_answer = output["output"]
                    conversation.append({"role": "assistant", "content": intermediate_answer})
                    stage_outputs.append(build_stage_record(
                        stage_name=f"answer_clarifying_question_round_{round_num}",
                        round_idx=round_num,
                        prompt_text=answer_clarify_prompt,
                        model_output=output,
                        conversation_before=conversation_before,
                        conversation_after=conversation,
                    ))

                    clarification_round_state = {
                        "round": round_num,
                        "intermediate_question": clarify_output,
                        "pinned_context": pinned_context,
                        "intermediate_answer": intermediate_answer,
                    }
                    clarification_rounds.append(clarification_round_state)
                    num_rounds_used = round_num

                if round_failed:
                    continue

                final_prompt = build_final_prompt(test_item)
                conversation.append({
                    "role": "user",
                    "content": final_prompt
                })

                conversation_before = clone_conversation(conversation)
                input_text_final_answer = get_chat_text(conversation, model.tokenizer)

                output = safe_generate(model, conversation, prompt_budget, idx, "final_answer")
                if output is None:
                    append_failure_trace(
                        trace_path,
                        idx,
                        source_idx,
                        test_item,
                        "final_answer",
                        conversation,
                        stage_outputs,
                        extra={"input_text_final_answer": input_text_final_answer},
                    )
                    continue

                final_answer_generation = copy.deepcopy(output)
                final_answer = output["output"]
                conversation.append({"role": "assistant", "content": final_answer})
                stage_outputs.append(build_stage_record(
                    stage_name="final_answer",
                    round_idx=None,
                    prompt_text=final_prompt,
                    model_output=final_answer_generation,
                    conversation_before=conversation_before,
                    conversation_after=conversation,
                ))

                termination_prompt = "Have you provided the correct answer?"
                conversation.append({"role": "user", "content": termination_prompt})
                conversation_before = clone_conversation(conversation)
                input_text_termination_check = get_chat_text(conversation, model.tokenizer)

                output = safe_generate(model, conversation, prompt_budget, idx, "termination_check")
                if output is None:
                    append_failure_trace(
                        trace_path,
                        idx,
                        source_idx,
                        test_item,
                        "termination_check",
                        conversation,
                        stage_outputs,
                        extra={"input_text_termination_check": input_text_termination_check},
                    )
                    continue

                termination_generation = copy.deepcopy(output)
                termination_check = output["output"]
                conversation.append({"role": "assistant", "content": termination_check})
                stage_outputs.append(build_stage_record(
                    stage_name="termination_check",
                    round_idx=None,
                    prompt_text=termination_prompt,
                    model_output=termination_generation,
                    conversation_before=conversation_before,
                    conversation_after=conversation,
                ))

                scoring_output = copy.deepcopy(final_answer_generation)
                scoring_output["output"] = final_answer

                if not args.use_chat_template:
                    prepend_text = data["system_template"].format(**test_item)
                    scoring_output["output"] = prepend_text + scoring_output["output"]

                mets, others = data["post_process"](scoring_output, test_item)
                scoring_output.update({**others, **mets})

                for k, v in mets.items():
                    metrics[k].append(v)

                metrics["input_len"].append(scoring_output["input_len"])
                metrics["output_len"].append(scoring_output["output_len"])

                compact_record = build_compact_result_record(
                    dataset=dataset,
                    test_item=test_item,
                    source_idx=source_idx,
                    scoring_output=scoring_output,
                    final_answer=final_answer,
                    termination_check=termination_check,
                    clarification_rounds=clarification_rounds,
                    adaptive_stop_records=adaptive_stop_records,
                    provisional_answers_by_round=provisional_answers_by_round,
                    stopped_early=stopped_early,
                    stopped_after_round=stopped_after_round,
                    num_rounds_used=num_rounds_used,
                    context_token_count=context_token_count,
                    context_pieces=context_pieces,
                )
                compact_record["num_clarification_rounds_requested"] = args.num_clarification_rounds
                compact_record["adaptive_stop_enabled"] = False
                compact_record["adaptive_stop_active_until_end"] = False
                results.append(compact_record)
                completed_ids.add(compact_record["example_id"])

                example_time = time.time() - example_start
                current_avg = compute_averaged_metrics(metrics, dataset)
                full_record = build_full_log_record(
                    compact_record=compact_record,
                    test_item=test_item,
                    input_text_final_answer=input_text_final_answer,
                    input_text_termination_check=input_text_termination_check,
                    full_conversation=conversation,
                    stage_outputs=stage_outputs,
                    example_time=example_time,
                )
                full_record["running_metrics"] = current_avg
                append_trace(progress_path, full_record)
                append_trace(trace_path, {
                    "status": "completed",
                    "example_id": compact_record["example_id"],
                    "example_idx": idx + 1,
                    "source_idx": source_idx,
                    "question_group_id": compact_record["question_group_id"],
                    "context_variant_id": compact_record["context_variant_id"],
                    "question": test_item["question"],
                    "num_clarification_rounds_used": num_rounds_used,
                    "stopped_early": stopped_early,
                    "stopped_after_round": stopped_after_round,
                    "final_answer_raw": final_answer,
                    "input_len": scoring_output["input_len"],
                    "output_len": scoring_output["output_len"],
                    "example_time_sec": round(example_time, 3),
                })

                if idx < 5 or args.debug:
                    logger.info(f"Example {idx + 1} (source_idx={source_idx}): ")
                    logger.info(f"Final-answer decoder input:\n{input_text_final_answer}\n")
                    logger.info(f"Termination-check decoder input:\n{input_text_termination_check}\n")
                    logger.info(f"Input length: {scoring_output['input_len']}")
                    logger.info(f"Question: {test_item['question']}\n")
                    logger.info(f"Answer: {test_item['answer']}")
                    logger.info(f"Final answer: {final_answer}")
                    logger.info(f"Termination check: {termination_check}")
                    logger.info(f"Parsed output: {scoring_output['parsed_output']}")
                    logger.info(f"Stopped early: {stopped_early} after round {stopped_after_round}")
                    logger.info(f"Example runtime: {example_time:.2f}s")
                    logger.info(f"Running averages after {len(results)} examples: {current_avg}")

                if (len(results) % save_every == 0) or (idx == effective_size - 1):
                    elapsed = time.time() - start_time
                    throughput = len(results) / elapsed if elapsed > 0 else 0.0
                    mem_usage = sum(
                        [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
                    )
                    logger.info(
                        f"saving partial progress at {len(results)} completed examples "
                        f"(elapsed={elapsed:.2f}s, throughput={throughput:.4f} samples/s)"
                    )
                    save_compact_results(
                        args=args,
                        output_path=output_path,
                        results=results,
                        metrics=metrics,
                        dataset=dataset,
                        mem_usage=mem_usage,
                        throughput=throughput,
                        is_partial=True,
                    )

                if args.debug:
                    import pdb
                    pdb.set_trace()

                output = None

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        throughput = len(results) / elapsed if elapsed > 0 else 0.0
        mem_usage = sum([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])

        logger.warning("KeyboardInterrupt received - saving progress before exit")
        save_compact_results(
            args=args,
            output_path=output_path,
            results=results,
            metrics=metrics,
            dataset=dataset,
            mem_usage=mem_usage,
            throughput=throughput,
            is_partial=True,
        )
        logger.warning(f"partial progress saved to {output_path}")
        return output_path

    end_time = time.time()
    mem_usage = sum([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
    throughput = len(results) / (end_time - start_time) if end_time > start_time else 0.0

    logger.info(f"Memory usage: {mem_usage/1000**3:.02f} GB")
    logger.info(f"Throughput: {throughput:.02f} samples/s")

    if args.count_tokens:
        if len(metrics["input_len"]) > 0:
            logger.info(
                f"----{dataset}----\n"
                f"Average input length: {np.mean(metrics['input_len']):.02f}, "
                f"std input length: {np.std(metrics['input_len']):.02f}, "
                f"max input length: {max(metrics['input_len'])}, "
                f"min input length: {min(metrics['input_len'])}\n"
                f"----returning----"
            )
        return output_path

    if len(results) == 0:
        logger.error("No results to evaluate, something went wrong, returning...")
        return output_path

    averaged_metrics = compute_averaged_metrics(metrics, dataset)

    logger.info("Averaged metrics:")
    for k, v in averaged_metrics.items():
        logger.info(f"{k}: {v:.02f}")

    if args.output_dir is not None:
        save_compact_results(
            args=args,
            output_path=output_path,
            results=results,
            metrics=metrics,
            dataset=dataset,
            mem_usage=mem_usage,
            throughput=throughput,
            is_partial=False,
        )
        if "alce" not in dataset:
            atomic_write_json(score_path, averaged_metrics, indent=4)
        logger.info(f"done, results are written to {output_path}")
        logger.info(f"score file written to {score_path}")
        logger.info(f"full log file written to {progress_path}")
        logger.info(f"event file written to {trace_path}")

    return output_path





def main():
    args = parse_arguments()

    logger.info(f"Arguments: {args}")
    assert args.model_name_or_path is not None

    if args.output_dir is None:
        logger.warning("no output directory specified, setting it to args.model_name_or_path but may cause error")
        args.output_dir = args.model_name_or_path
    os.makedirs(args.output_dir, exist_ok=True)

    if args.max_test_samples is None:
        logger.warning("max_test_samples is None; forcing it to 100 for safer evaluation")
        args.max_test_samples = 100

    if args.num_clarification_rounds < 1:
        logger.warning("num_clarification_rounds < 1; forcing it to 1")
        args.num_clarification_rounds = 1

    if not hasattr(args, "complete_restart"):
        args.complete_restart = False
    if not hasattr(args, "save_every"):
        args.save_every = 1
    if not hasattr(args, "start_idx"):
        args.start_idx = None
    if not hasattr(args, "end_idx"):
        args.end_idx = None
    if not hasattr(args, "example_indices"):
        args.example_indices = None
    if not hasattr(args, "example_indices_file"):
        args.example_indices_file = None

    if not args.do_sample:
        if args.temperature != 0.0:
            logger.warning("do_sample is set to false but temperature is not 0, do_sample will overwrite temperature")

    model = load_LLM(args)

    datasets = args.datasets.split(",")
    test_files = args.test_files.split(",")
    demo_files = args.demo_files.split(",")
    max_lengths = (
        [int(args.input_max_length)] * len(datasets)
        if isinstance(args.input_max_length, int) or len(str(args.input_max_length).split(",")) == 1
        else [int(l) for l in str(args.input_max_length).split(",")]
    )
    gen_lengths = (
        [int(args.generation_max_length)] * len(datasets)
        if isinstance(args.generation_max_length, int) or len(str(args.generation_max_length).split(",")) == 1
        else [int(l) for l in str(args.generation_max_length).split(",")]
    )
    assert len(test_files) == len(demo_files)

    for dataset, test_file, demo_file, max_length, gen_length in zip(
        datasets, test_files, demo_files, max_lengths, gen_lengths
    ):
        args.datasets = dataset
        args.test_files = test_file
        args.demo_files = demo_file
        args.input_max_length = max_length
        args.generation_max_length = gen_length
        model.max_length = max_length
        model.generation_max_length = gen_length

        logger.info(
            f"starting dataset={dataset} with input_max_length={max_length}, "
            f"generation_max_length={gen_length}, max_test_samples={args.max_test_samples}, "
            f"num_clarification_rounds={args.num_clarification_rounds}, "
            f"complete_restart={args.complete_restart}"
        )

        try:
            output_path = run_test(args, model, dataset, test_file, demo_file)

            if "alce" in dataset and not args.count_tokens and (not os.path.exists(output_path + ".score") or args.overwrite):
                import eval_alce
                logger.info("running eval_alce.py...")
                cli_args = ["--f", output_path]
                if "nocite" not in dataset:
                    cli_args.append("--citations")
                if "asqa" in dataset:
                    cli_args.append("--mauve")
                elif "eli5" in dataset:
                    cli_args += ["mauve", "--claims_nli"]
                eval_alce.main(cli_args)

        except Exception as e:
            logger.exception(f"Error in {dataset}: {e}, continuing...")


if __name__ == "__main__":
    main()
