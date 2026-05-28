import copy
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_HELMET_DIR = os.path.join(
    CURRENT_DIR,
    "..",
    "..",
    "stage1-agenticlu-runtime",
    "AgenticLU-Modified",
    "HELMET",
)
if BASE_HELMET_DIR not in sys.path:
    sys.path.append(BASE_HELMET_DIR)

import eval_agent as base
from verify_arguments import parse_arguments
from verify_model_utils import load_LLM


def build_verify_run_signature(args, dataset, test_file):
    signature = base.build_run_signature(args, dataset, test_file)
    signature.update(
        {
            "source_output_dir": getattr(args, "source_output_dir", None),
            "source_output_path": getattr(args, "source_output_path", None),
            "verify_refine_enabled": getattr(args, "verify_refine_enabled", None),
            "verify_verifier_generation_max_length": getattr(args, "verify_verifier_generation_max_length", None),
            "verify_use_falsifier": getattr(args, "verify_use_falsifier", None),
            "verify_falsifier_generation_max_length": getattr(args, "verify_falsifier_generation_max_length", None),
        }
    )
    return signature


def build_source_run_paths(args, dataset, test_file):
    if getattr(args, "source_output_path", None):
        source_path = args.source_output_path
        if source_path.endswith(".jsonl"):
            return source_path, None, None, None
        source_dir = os.path.dirname(source_path)
        source_name = os.path.splitext(os.path.basename(source_path))[0]
        return source_path, source_dir, source_name, None

    source_dir = getattr(args, "source_output_dir", None) or args.output_dir
    source_args = copy.copy(args)
    source_args.output_dir = source_dir
    source_path, source_score_path, source_progress_path, source_trace_path = base.build_output_paths(
        source_args,
        dataset,
        test_file,
    )
    return source_path, source_score_path, source_progress_path, source_trace_path


def load_source_payload(path):
    if path is None or not os.path.exists(path):
        return None

    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return {
            "data": records,
            "source_format": "jsonl",
            "source_path": path,
        }

    payload = base._load_json_payload(path)
    payload["source_format"] = "json"
    payload["source_path"] = path
    return payload


def load_source_results(args, dataset, test_file):
    source_path, _, _, _ = build_source_run_paths(args, dataset, test_file)
    candidate_paths = [source_path]
    if source_path and not source_path.endswith(".jsonl"):
        candidate_paths.append(os.path.splitext(source_path)[0] + ".full.jsonl")
    elif source_path and source_path.endswith(".jsonl"):
        candidate_paths.append(os.path.splitext(source_path)[0] + ".json")

    for candidate in candidate_paths:
        if candidate is None or not os.path.exists(candidate):
            continue
        try:
            payload = load_source_payload(candidate)
        except Exception as exc:
            base.logger.warning(f"failed to load source payload from {candidate}: {exc}")
            continue
        data = payload.get("data", [])
        if not isinstance(data, list):
            base.logger.warning(f"source payload at {candidate} has non-list data; ignoring it")
            continue
        base.logger.info(f"loaded {len(data)} saved core records from {candidate}")
        return payload, candidate

    raise FileNotFoundError(
        f"could not find source output for dataset={dataset} test={test_file}; "
        f"checked {candidate_paths}"
    )


def should_resume_from_payload_verify(payload, expected_signature):
    if payload is None:
        return False
    saved_args = payload.get("args", {})
    saved_signature = build_verify_run_signature(
        type("Args", (), saved_args),
        expected_signature["dataset"],
        expected_signature["test_file"],
    )
    return saved_signature == expected_signature


def load_resume_results_verify(args, output_path, expected_signature):
    resume_path = getattr(args, "resume_from_path", None) or output_path
    candidate_paths = [resume_path]
    if resume_path == output_path:
        candidate_paths.append(output_path + ".bak")

    for path in candidate_paths:
        if path is None or not os.path.exists(path):
            continue
        try:
            payload = base._load_json_payload(path)
        except Exception as exc:
            base.logger.warning(f"failed to load verify resume payload from {path}: {exc}")
            continue
        if not should_resume_from_payload_verify(payload, expected_signature):
            base.logger.warning(f"verify resume payload at {path} does not match current run signature; ignoring it")
            continue
        existing_results = payload.get("data", [])
        if not isinstance(existing_results, list):
            continue
        base.logger.info(f"loaded {len(existing_results)} completed examples from compact verify results file {path}")
        return existing_results, path

    base.logger.info("no matching verify resume file found, starting from scratch")
    return [], None


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_append_trace(trace_path, payload):
    ensure_parent_dir(trace_path)
    base.append_trace(trace_path, payload)


def generate_with_stage_cap(model, conversation, prompt_budget, idx, stage_name, generation_cap):
    original_cap = getattr(model, "generation_max_length", None)
    try:
        if generation_cap is not None:
            model.generation_max_length = generation_cap
        return base.safe_generate(model, conversation, prompt_budget, idx, stage_name)
    finally:
        if generation_cap is not None and original_cap is not None:
            model.generation_max_length = original_cap


def build_answer_verifier_prompt(test_item, clarification_state, provisional_answer):
    lines = [
        "You are verifying a provisional answer in a long-context QA workflow.",
        f"Original question: {test_item['question']}",
        "",
        "Current clarification question:",
        clarification_state["intermediate_question"],
        "Current pinned evidence:",
        clarification_state["pinned_context"],
        "Current clarification answer:",
        clarification_state["intermediate_answer"],
        "Current provisional final answer:",
        provisional_answer,
        "",
        "Judge whether the current answer is directly supported, complete, and safe to keep as the best answer.",
        "Only recommend another refinement if there is a concrete missing piece or a real need to verify against alternatives.",
        "Do not invent ambiguity such as unspecified time periods, positions, or roles unless the evidence truly requires it.",
        "Return exactly one JSON object:",
        "{",
        '  "direct_support_prob": number between 0.0 and 1.0,',
        '  "answer_complete_prob": number between 0.0 and 1.0,',
        '  "regression_risk_prob": number between 0.0 and 1.0,',
        '  "refinement_expected_gain": number between 0.0 and 1.0,',
        '  "recommend_refine": true or false,',
        '  "missing_info_type": "none" or a short label,',
        '  "rationale": "short explanation"',
        "}",
    ]
    return "\n".join(lines)


def build_answer_falsifier_prompt(test_item, clarification_state, provisional_answer):
    lines = [
        "You are stress-testing a provisional answer in a long-context QA workflow.",
        "Your job is not to confirm the answer. Your job is to identify whether the current answer may still be wrong even if it looks coherent.",
        f"Original question: {test_item['question']}",
        "",
        "Current clarification question:",
        clarification_state["intermediate_question"],
        "Current pinned evidence:",
        clarification_state["pinned_context"],
        "Current clarification answer:",
        clarification_state["intermediate_answer"],
        "Current provisional final answer:",
        provisional_answer,
        "",
        "Focus on three risks:",
        "1. The answer may be unsupported by the right evidence even if the current chain sounds coherent.",
        "2. A plausible competing answer may fit the question better.",
        "3. The current evidence selection may have locked onto the wrong entity, date, role, or relation.",
        "Return exactly one JSON object:",
        "{",
        '  "false_answer_risk_prob": number between 0.0 and 1.0,',
        '  "alternative_answer_risk_prob": number between 0.0 and 1.0,',
        '  "evidence_selection_risk_prob": number between 0.0 and 1.0,',
        '  "rationale": "short explanation"',
        "}",
    ]
    return "\n".join(lines)


def parse_answer_verifier_output(raw_text):
    parsed = {
        "direct_support_prob": None,
        "answer_complete_prob": None,
        "regression_risk_prob": None,
        "missing_info_type": None,
        "refinement_expected_gain": None,
        "recommend_refine": None,
        "rationale": None,
        "parse_error": None,
        "raw_text": raw_text,
    }
    json_block = base.extract_json_block(raw_text)
    if json_block is None:
        parsed["parse_error"] = "no_json_found"
        return parsed
    try:
        data = json.loads(json_block)
    except Exception as exc:
        parsed["parse_error"] = f"json_decode_error: {exc}"
        return parsed

    parsed["direct_support_prob"] = base.clamp01(base.coerce_float(data.get("direct_support_prob")))
    parsed["answer_complete_prob"] = base.clamp01(base.coerce_float(data.get("answer_complete_prob")))
    parsed["regression_risk_prob"] = base.clamp01(base.coerce_float(data.get("regression_risk_prob")))
    parsed["missing_info_type"] = base.normalize_label(data.get("missing_info_type"))
    parsed["refinement_expected_gain"] = base.clamp01(base.coerce_float(data.get("refinement_expected_gain")))
    parsed["recommend_refine"] = base.coerce_bool(data.get("recommend_refine"))
    parsed["rationale"] = data.get("rationale")
    return parsed


def parse_answer_falsifier_output(raw_text):
    parsed = {
        "false_answer_risk_prob": None,
        "alternative_answer_risk_prob": None,
        "evidence_selection_risk_prob": None,
        "rationale": None,
        "parse_error": None,
        "raw_text": raw_text,
    }
    json_block = base.extract_json_block(raw_text)
    if json_block is None:
        parsed["parse_error"] = "no_json_found"
        return parsed
    try:
        data = json.loads(json_block)
    except Exception as exc:
        parsed["parse_error"] = f"json_decode_error: {exc}"
        return parsed

    parsed["false_answer_risk_prob"] = base.clamp01(base.coerce_float(data.get("false_answer_risk_prob")))
    parsed["alternative_answer_risk_prob"] = base.clamp01(base.coerce_float(data.get("alternative_answer_risk_prob")))
    parsed["evidence_selection_risk_prob"] = base.clamp01(base.coerce_float(data.get("evidence_selection_risk_prob")))
    parsed["rationale"] = data.get("rationale")
    return parsed


def run_answer_verifier(args, model, prompt_budget, idx, round_num, test_item, clarification_state, provisional_answer, stage_outputs):
    verifier_prompt = build_answer_verifier_prompt(
        test_item=test_item,
        clarification_state=clarification_state,
        provisional_answer=provisional_answer,
    )
    verifier_conversation = [
        {"role": "system", "content": "You are a careful evidence-based verifier."},
        {"role": "user", "content": verifier_prompt},
    ]
    conversation_before = base.clone_conversation(verifier_conversation)

    output = generate_with_stage_cap(
        model=model,
        conversation=verifier_conversation,
        prompt_budget=prompt_budget,
        idx=idx,
        stage_name=f"answer_verifier_round_{round_num}",
        generation_cap=getattr(args, "verify_verifier_generation_max_length", None),
    )
    if output is None:
        stage_outputs.append(
            base.build_stage_record(
                stage_name=f"answer_verifier_round_{round_num}",
                round_idx=round_num,
                prompt_text=verifier_prompt,
                model_output=None,
                conversation_before=conversation_before,
                conversation_after=verifier_conversation,
            )
        )
        return None

    parsed = parse_answer_verifier_output(output["output"])
    verifier_conversation.append({"role": "assistant", "content": output["output"]})
    output_logged = copy.deepcopy(output)
    output_logged["parsed_output"] = parsed

    stage_outputs.append(
        base.build_stage_record(
            stage_name=f"answer_verifier_round_{round_num}",
            round_idx=round_num,
            prompt_text=verifier_prompt,
            model_output=output_logged,
            conversation_before=conversation_before,
            conversation_after=verifier_conversation,
        )
    )

    return {
        "round": round_num,
        "raw_output": output["output"],
        "parsed_output": parsed,
        "input_len": output.get("input_len"),
        "output_len": output.get("output_len"),
        "input_text": output.get("input_text"),
    }


def run_answer_falsifier(args, model, prompt_budget, idx, round_num, test_item, clarification_state, provisional_answer, stage_outputs):
    if not getattr(args, "verify_use_falsifier", True):
        return None

    falsifier_prompt = build_answer_falsifier_prompt(
        test_item=test_item,
        clarification_state=clarification_state,
        provisional_answer=provisional_answer,
    )
    falsifier_conversation = [
        {"role": "system", "content": "You are a careful evidence-based falsifier."},
        {"role": "user", "content": falsifier_prompt},
    ]
    conversation_before = base.clone_conversation(falsifier_conversation)

    output = generate_with_stage_cap(
        model=model,
        conversation=falsifier_conversation,
        prompt_budget=prompt_budget,
        idx=idx,
        stage_name=f"answer_falsifier_round_{round_num}",
        generation_cap=getattr(args, "verify_falsifier_generation_max_length", None),
    )
    if output is None:
        stage_outputs.append(
            base.build_stage_record(
                stage_name=f"answer_falsifier_round_{round_num}",
                round_idx=round_num,
                prompt_text=falsifier_prompt,
                model_output=None,
                conversation_before=conversation_before,
                conversation_after=falsifier_conversation,
            )
        )
        return None

    parsed = parse_answer_falsifier_output(output["output"])
    falsifier_conversation.append({"role": "assistant", "content": output["output"]})
    output_logged = copy.deepcopy(output)
    output_logged["parsed_output"] = parsed

    stage_outputs.append(
        base.build_stage_record(
            stage_name=f"answer_falsifier_round_{round_num}",
            round_idx=round_num,
            prompt_text=falsifier_prompt,
            model_output=output_logged,
            conversation_before=conversation_before,
            conversation_after=falsifier_conversation,
        )
    )

    return {
        "round": round_num,
        "raw_output": output["output"],
        "parsed_output": parsed,
        "input_len": output.get("input_len"),
        "output_len": output.get("output_len"),
        "input_text": output.get("input_text"),
    }


def should_refine(args, verifier_record, falsifier_record, round_num):
    from stage3_common import to_band

    if verifier_record is None:
        return False, "no_verifier_output"
    verifier_parsed = verifier_record["parsed_output"]
    if verifier_parsed.get("parse_error") is not None:
        return False, verifier_parsed["parse_error"]

    d = to_band(verifier_parsed.get("direct_support_prob"))
    c = to_band(verifier_parsed.get("answer_complete_prob"))
    q = to_band(verifier_parsed.get("refinement_expected_gain"))
    v = verifier_parsed.get("recommend_refine") is True
    m = verifier_parsed.get("missing_info_type") not in {None, "none"}

    falsifier_parsed = falsifier_record["parsed_output"] if falsifier_record is not None else {}
    f = to_band(falsifier_parsed.get("false_answer_risk_prob"))
    a = to_band(falsifier_parsed.get("alternative_answer_risk_prob"))
    e = to_band(falsifier_parsed.get("evidence_selection_risk_prob"))

    if (
        d == "high"
        and c == "high"
        and q == "low"
        and f != "high"
        and a != "high"
        and e != "high"
    ):
        return False, "clean_stop"

    if f == "high":
        return True, "false_answer_risk_high"
    if a == "high":
        return True, "alternative_answer_risk_high"
    if e == "high":
        return True, "evidence_selection_risk_high"
    if v and q == "high":
        return True, "verifier_recommends_refine_with_high_gain"
    if m and q == "high":
        return True, "missing_info_type_with_high_gain"

    return False, "default_stop"


def run_test(args, model, dataset, test_file, demo_file):
    base.logger.info(
        f"running verify-refine on saved core records for {dataset} with test {test_file} and demo {demo_file}"
    )

    random.seed(args.seed)
    source_payload, source_path = load_source_results(args, dataset, test_file)
    source_results = source_payload.get("data", [])

    output_path, score_path, progress_path, trace_path = base.build_output_paths(args, dataset, test_file)
    if getattr(args, "complete_restart", False):
        base.cleanup_restart_files(output_path, progress_path, score_path, trace_path)

    ensure_parent_dir(output_path)

    expected_signature = build_verify_run_signature(args, dataset, test_file)
    results, loaded_from = load_resume_results_verify(args, output_path, expected_signature)
    completed_ids = {
        record.get("example_id")
        for record in results
        if isinstance(record, dict) and record.get("example_id")
    }
    metrics = base.rebuild_metrics_from_results(results)

    base.logger.info(
        f"verify resume debug -> output_path={output_path}, progress_path={progress_path}, "
        f"loaded_resume_count={len(results)}, loaded_from={loaded_from}, source_count={len(source_results)}"
    )

    max_input_tokens = base.parse_max_len(model.max_length)
    gen_tokens = base.parse_max_len(model.generation_max_length)
    prompt_budget = max_input_tokens - gen_tokens
    save_every = max(1, getattr(args, "save_every", 1))
    start_time = time.time()

    def update_metrics_from_record(record):
        metric_values = record.get("metric_values", {})
        for key, value in metric_values.items():
            if isinstance(value, (int, float, np.bool_)):
                metrics[key].append(value)
        if record.get("input_len") is not None:
            metrics["input_len"].append(record.get("input_len"))
        if record.get("output_len") is not None:
            metrics["output_len"].append(record.get("output_len"))

    try:
        for idx, source_record in enumerate(source_results):
            example_id = source_record.get("example_id")
            if example_id in completed_ids:
                continue

            example_start = time.time()
            clarification_rounds = copy.deepcopy(source_record.get("clarification_rounds") or [])
            provisional_answers_by_round = copy.deepcopy(source_record.get("provisional_answers_by_round") or [])
            if not provisional_answers_by_round and source_record.get("final_answer_raw") is not None:
                provisional_answers_by_round = [
                    {
                        "round": 1,
                        "final_answer": source_record.get("final_answer_raw"),
                        "input_text": None,
                    }
                ]

            test_item = {
                "question": source_record.get("question"),
                "answer": source_record.get("answer"),
                "answers": source_record.get("answers"),
            }

            verify_refine_records = []
            adaptive_stop_records = copy.deepcopy(source_record.get("adaptive_stop_records") or [])
            stage_outputs = []

            if clarification_rounds and provisional_answers_by_round:
                clarification_state = clarification_rounds[0]
                provisional_entry = provisional_answers_by_round[0]
                provisional_answer = provisional_entry.get("final_answer") or source_record.get("final_answer_raw")

                verifier_record = run_answer_verifier(
                    args=args,
                    model=model,
                    prompt_budget=prompt_budget,
                    idx=idx,
                    round_num=1,
                    test_item=test_item,
                    clarification_state=clarification_state,
                    provisional_answer=provisional_answer,
                    stage_outputs=stage_outputs,
                )
                falsifier_record = run_answer_falsifier(
                    args=args,
                    model=model,
                    prompt_budget=prompt_budget,
                    idx=idx,
                    round_num=1,
                    test_item=test_item,
                    clarification_state=clarification_state,
                    provisional_answer=provisional_answer,
                    stage_outputs=stage_outputs,
                )

                refine, refine_reason = should_refine(args, verifier_record, falsifier_record, 1)

                verify_refine_records.append(
                    {
                        "round": 1,
                        "refine_decision": refine,
                        "refine_reason": refine_reason,
                        "verifier_record": copy.deepcopy(verifier_record),
                        "falsifier_record": copy.deepcopy(falsifier_record),
                        "candidate_answer": provisional_answer,
                    }
                )

            final_answer = source_record.get("final_answer_raw") or source_record.get("output")
            final_answer_generation = {
                "output": final_answer,
                "input_len": source_record.get("input_len"),
                "output_len": source_record.get("output_len"),
                "input_text": None,
            }

            compact_record = copy.deepcopy(source_record)
            compact_record["output"] = final_answer
            compact_record["final_answer_raw"] = final_answer
            compact_record["termination_check"] = source_record.get("termination_check")
            compact_record["num_clarification_rounds_requested"] = source_record.get(
                "num_clarification_rounds_requested",
                len(clarification_rounds),
            )
            compact_record["num_clarification_rounds_used"] = len(clarification_rounds)
            compact_record["verify_refine_enabled"] = args.verify_refine_enabled
            compact_record["verify_refine_records"] = copy.deepcopy(verify_refine_records)
            compact_record["verify_source_output_path"] = source_path
            compact_record["verify_source_output_format"] = source_payload.get("source_format")
            compact_record["source_final_answer_raw"] = source_record.get("final_answer_raw")
            compact_record["source_num_rounds"] = len(clarification_rounds)

            results.append(compact_record)
            completed_ids.add(example_id)
            update_metrics_from_record(compact_record)

            example_time = time.time() - example_start
            current_avg = base.compute_averaged_metrics(metrics, dataset)

            full_record = copy.deepcopy(compact_record)
            full_record["status"] = "completed"
            full_record["example_time_sec"] = round(example_time, 3)
            full_record["running_metrics"] = current_avg
            full_record["source_output_path"] = source_path
            safe_append_trace(progress_path, full_record)
            safe_append_trace(
                trace_path,
                {
                    "status": "completed",
                    "example_id": compact_record["example_id"],
                    "example_idx": idx + 1,
                    "source_idx": compact_record.get("source_idx"),
                    "question": compact_record.get("question"),
                    "num_clarification_rounds_used": len(clarification_rounds),
                    "final_answer_raw": final_answer,
                    "example_time_sec": round(example_time, 3),
                    "source_output_path": source_path,
                },
            )

            if (len(results) % save_every == 0) or (idx == len(source_results) - 1):
                elapsed = time.time() - start_time
                throughput = len(results) / elapsed if elapsed > 0 else 0.0
                mem_usage = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
                base.save_compact_results(
                    args=args,
                    output_path=output_path,
                    results=results,
                    metrics=metrics,
                    dataset=dataset,
                    mem_usage=mem_usage,
                    throughput=throughput,
                    is_partial=True,
                )

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        throughput = len(results) / elapsed if elapsed > 0 else 0.0
        mem_usage = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
        base.save_compact_results(
            args=args,
            output_path=output_path,
            results=results,
            metrics=metrics,
            dataset=dataset,
            mem_usage=mem_usage,
            throughput=throughput,
            is_partial=True,
        )
        return output_path

    end_time = time.time()
    mem_usage = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
    throughput = len(results) / (end_time - start_time) if end_time > start_time else 0.0

    if len(results) > 0:
        averaged_metrics = base.compute_averaged_metrics(metrics, dataset)
        base.save_compact_results(
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
            base._atomic_write_json(score_path, averaged_metrics, indent=4)
    return output_path


def main():
    args = parse_arguments()
    base.logger.info(f"Arguments: {args}")
    assert args.model_name_or_path is not None

    if args.output_dir is None:
        args.output_dir = args.model_name_or_path
    os.makedirs(args.output_dir, exist_ok=True)
    if args.source_output_path is None and args.source_output_dir is None:
        raise ValueError(
            "verify-refine now consumes saved core outputs; provide --source_output_dir or --source_output_path"
        )

    if args.max_test_samples is None:
        args.max_test_samples = 100
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

    model = load_LLM(args)
    datasets = args.datasets.split(",")
    test_files = args.test_files.split(",")
    demo_files = args.demo_files.split(",")
    max_lengths = [int(args.input_max_length)] * len(datasets) if len(str(args.input_max_length).split(",")) == 1 else [int(x) for x in str(args.input_max_length).split(",")]
    gen_lengths = [int(args.generation_max_length)] * len(datasets) if len(str(args.generation_max_length).split(",")) == 1 else [int(x) for x in str(args.generation_max_length).split(",")]

    for dataset, test_file, demo_file, max_length, gen_length in zip(datasets, test_files, demo_files, max_lengths, gen_lengths):
        args.datasets = dataset
        args.test_files = test_file
        args.demo_files = demo_file
        args.input_max_length = max_length
        args.generation_max_length = gen_length
        model.max_length = max_length
        model.generation_max_length = gen_length
        run_test(args, model, dataset, test_file, demo_file)


if __name__ == "__main__":
    main()