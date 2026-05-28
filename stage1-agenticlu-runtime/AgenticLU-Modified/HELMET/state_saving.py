"""Progressive state saving helpers for the AgenticLU evaluation runtime."""

import copy
import hashlib
import json
import logging
import os
import re
import string

import numpy as np


logger = logging.getLogger(__name__)


def clone_conversation(conversation):
    return copy.deepcopy(conversation)


def build_stage_record(
    stage_name,
    round_idx,
    prompt_text,
    model_output,
    conversation_before,
    conversation_after,
):
    record = {
        "stage_name": stage_name,
        "round": round_idx,
        "prompt_text": prompt_text,
        "conversation_before": clone_conversation(conversation_before),
        "conversation_after": clone_conversation(conversation_after),
        "generation": None,
    }

    if model_output is not None:
        record["generation"] = {
            "raw_output_dict": copy.deepcopy(model_output),
            "text": model_output.get("output"),
            "parsed_output": model_output.get("parsed_output"),
            "input_len": model_output.get("input_len"),
            "output_len": model_output.get("output_len"),
            "input_text": model_output.get("input_text"),
        }

    return record


def append_trace(trace_path, payload):
    with open(trace_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def slugify_for_path(value, max_len=40):
    value = os.path.basename(str(value)) if value is not None else "na"
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    if value == "":
        value = "na"
    return value[:max_len]


def short_text_hash(text, length=12):
    if text is None:
        text = ""
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:length]


def normalize_text_for_identity(text):
    if text is None:
        return ""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def build_example_identity(dataset, source_idx, test_item):
    question = str(test_item.get("question", "")).strip()
    context = str(test_item.get("context", "")).strip()
    question_group_id = short_text_hash(normalize_text_for_identity(question))
    context_variant_id = short_text_hash(normalize_text_for_identity(context))
    example_id = f"{dataset}:{source_idx}:{question_group_id}:{context_variant_id}"
    return {
        "example_id": example_id,
        "question_group_id": question_group_id,
        "context_variant_id": context_variant_id,
    }


def append_failure_trace(
    trace_path,
    idx,
    source_idx,
    test_item,
    failed_stage,
    conversation,
    stage_outputs,
    extra=None,
):
    identity = build_example_identity("unknown", source_idx, test_item)
    payload = {
        "example_idx": idx + 1,
        "example_id": identity["example_id"],
        "question_group_id": identity["question_group_id"],
        "context_variant_id": identity["context_variant_id"],
        "source_idx": source_idx,
        "question": test_item.get("question"),
        "answer": test_item.get("answer"),
        "status": "skipped",
        "failed_stage": failed_stage,
        "full_conversation_so_far": clone_conversation(conversation),
        "stage_outputs_so_far": copy.deepcopy(stage_outputs),
    }
    if extra is not None:
        payload["extra"] = extra
    append_trace(trace_path, payload)


def compute_averaged_metrics(metrics, dataset):
    if len(metrics) == 0:
        return {}

    averaged_metrics = {}
    for key, values in metrics.items():
        if len(values) == 0:
            continue
        averaged_metrics[key] = np.mean(values) * (100 if "_len" not in key else 1)

    if "dialogre" in dataset and len(metrics.get("precision", [])) > 0:
        prec = np.average(metrics["precision"], weights=metrics["num_preds"]) if sum(metrics["num_preds"]) > 0 else 0
        rec = np.average(metrics["recall"], weights=metrics["num_labels"])
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        averaged_metrics["dialogre_precision"] = prec * 100
        averaged_metrics["dialogre_recall"] = rec * 100
        averaged_metrics["dialogre_f1"] = f1 * 100

    return averaged_metrics


def cleanup_restart_files(output_path, progress_path, score_path, trace_path):
    for path in [output_path, progress_path, score_path, trace_path]:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"deleted old file for complete restart: {path}")


def build_output_paths(args, dataset, test_file):
    tag = args.tag
    if dataset == "popqa":
        tag += f"-pop{args.popularity_threshold}"

    test_name = os.path.splitext(os.path.basename(test_file))[0]
    model_name = os.path.basename(str(args.model_name_or_path).rstrip("/"))
    subset_tag = getattr(args, "_subset_tag", "subsetall")
    subset_suffix = ""
    if subset_tag != "subsetall":
        subset_suffix = f"__{slugify_for_path(subset_tag, max_len=24)}"

    base_name = (
        f"{slugify_for_path(dataset, max_len=24)}"
        f"__{slugify_for_path(test_name, max_len=24)}"
        f"__{slugify_for_path(tag, max_len=24)}"
        f"__{slugify_for_path(model_name, max_len=24)}"
        f"__seed{args.seed}"
        f"__coc{args.num_clarification_rounds}"
        f"__ad0"
        f"{subset_suffix}"
    )

    output_path = os.path.join(args.output_dir, base_name + ".json")
    score_path = os.path.join(args.output_dir, base_name + ".metrics.json")
    progress_path = os.path.join(args.output_dir, base_name + ".full.jsonl")
    trace_path = os.path.join(args.output_dir, base_name + ".events.jsonl")
    return output_path, score_path, progress_path, trace_path


def atomic_write_json(path, payload, indent=None):
    tmp_path = path + ".tmp"
    bak_path = path + ".bak"

    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())

    if os.path.exists(path):
        try:
            os.replace(path, bak_path)
        except Exception as exc:
            logger.warning(f"could not rotate backup for {path}: {exc}")

    os.replace(tmp_path, path)


def load_json_payload(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_compact_result_record(
    dataset,
    test_item,
    source_idx,
    scoring_output,
    final_answer,
    termination_check,
    clarification_rounds,
    adaptive_stop_records,
    provisional_answers_by_round,
    stopped_early,
    stopped_after_round,
    num_rounds_used,
    context_token_count,
    context_pieces,
):
    identity = build_example_identity(dataset, source_idx, test_item)
    metric_values = {}
    for key, value in scoring_output.items():
        if key in {"output", "parsed_output", "input_len", "output_len", "input_text"}:
            continue
        if isinstance(value, (int, float, bool, str, list, dict)) or value is None:
            metric_values[key] = copy.deepcopy(value)

    return {
        "example_id": identity["example_id"],
        "dataset": dataset,
        "source_idx": source_idx,
        "question_group_id": identity["question_group_id"],
        "context_variant_id": identity["context_variant_id"],
        "question": test_item.get("question"),
        "answer": test_item.get("answer"),
        "answers": test_item.get("answers", test_item.get("answer")),
        "final_answer_raw": final_answer,
        "output": final_answer,
        "termination_check": termination_check,
        "parsed_output": copy.deepcopy(scoring_output.get("parsed_output")),
        "input_len": scoring_output.get("input_len"),
        "output_len": scoring_output.get("output_len"),
        "metric_values": metric_values,
        "num_clarification_rounds_requested": None,
        "num_clarification_rounds_used": num_rounds_used,
        "stopped_early": stopped_early,
        "stopped_after_round": stopped_after_round,
        "adaptive_stop_records": copy.deepcopy(adaptive_stop_records),
        "provisional_answers_by_round": copy.deepcopy(provisional_answers_by_round),
        "clarification_rounds": copy.deepcopy(clarification_rounds),
        "context_token_count": context_token_count,
        "num_context_chunks": len(context_pieces),
    }


def build_full_log_record(
    compact_record,
    test_item,
    input_text_final_answer,
    input_text_termination_check,
    full_conversation,
    stage_outputs,
    example_time,
):
    full_record = copy.deepcopy(compact_record)
    full_record["status"] = "completed"
    full_record["input_text_final_answer"] = input_text_final_answer
    full_record["input_text_termination_check"] = input_text_termination_check
    full_record["full_conversation"] = copy.deepcopy(full_conversation)
    full_record["stage_outputs"] = copy.deepcopy(stage_outputs)
    full_record["example_time_sec"] = round(example_time, 3)
    full_record["question"] = test_item.get("question")
    full_record["answer"] = test_item.get("answer")
    return full_record


def save_compact_results(args, output_path, results, metrics, dataset, mem_usage=None, throughput=None, is_partial=True):
    averaged_metrics = compute_averaged_metrics(metrics, dataset)
    payload = {
        "args": copy.deepcopy(args.__dict__),
        "num_completed": len(results),
        "metrics": {key: list(values) for key, values in metrics.items()},
        "averaged_metrics": averaged_metrics,
        "data": results,
        "memory_usage": mem_usage,
        "throughput": throughput,
        "is_partial": is_partial,
    }
    atomic_write_json(output_path, payload, indent=4)
    return averaged_metrics
