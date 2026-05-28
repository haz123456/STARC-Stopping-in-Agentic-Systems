import argparse
from copy import deepcopy
from typing import Any, Dict, Optional

from agenticlu_model_utils import OpenAIModel
from _judging_common import (
    ROUND_LABELS,
    build_round_output_path,
    extract_gold_answers,
    extract_question,
    is_valid_verification,
    judge_one_answer,
    load_json,
    merge_existing_round_scores,
    populate_round_fields,
    recompute_round_metrics,
    save_json,
)


def judge_round_answers(
    input_file: str,
    output_file: Optional[str] = None,
    judge_model: str = "gpt-4o",
    temperature: float = 0.1,
    overwrite: bool = False,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    source_results = load_json(input_file)
    results = deepcopy(source_results)
    if max_examples is not None and isinstance(results.get("data"), list):
        results["data"] = results["data"][: min(max_examples, len(results["data"]))]

    output_path = output_file or build_round_output_path(input_file)
    if not overwrite:
        try:
            existing = load_json(output_path)
            results = merge_existing_round_scores(results, existing)
        except Exception:
            pass

    model = OpenAIModel(judge_model, temperature=temperature)
    judge_model_name = getattr(model, "model_name", judge_model)

    for example in results.get("data", []):
        populate_round_fields(example)
        question = extract_question(example)
        gold_answers = extract_gold_answers(example)
        if question is None or gold_answers is None:
            continue

        round_cache: Dict[str, Dict[str, Any]] = {}
        for label in ROUND_LABELS:
            field_name = "final_answer_raw" if label == "final" else f"{label}_answer"
            verification_key = f"gpt4_verification_{label}"
            raw_key = f"gpt4_verification_raw_output_{label}"
            answer = example.get(field_name)

            if answer is None:
                example[verification_key] = None
                continue
            if is_valid_verification(example.get(verification_key)) and not overwrite:
                continue

            cache_key = f"{question}|||{gold_answers}|||{answer}"
            cached = round_cache.get(cache_key)
            if cached is None:
                cached = judge_one_answer(model, judge_model_name, question, gold_answers, answer)
                round_cache[cache_key] = cached

            example[verification_key] = cached["verification"]
            example[raw_key] = cached["raw_output"]

        recompute_round_metrics(results)
        results["gpt4_eval_rounds_meta"] = {
            "judge_model": judge_model_name,
            "source_results_file": input_file,
            "output_file": output_path,
            "num_examples_in_output_file": len(results.get("data", [])),
        }
        save_json(output_path, results)

    recompute_round_metrics(results)
    save_json(output_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge the round answers in a compact results JSON file.")
    parser.add_argument("--input-file", required=True, help="Compact results JSON file.")
    parser.add_argument("--output-file", default=None, help="Optional judged output path.")
    parser.add_argument("--judge-model", default="gpt-4o", help="OpenAI judge model.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Judge temperature.")
    parser.add_argument("--overwrite", action="store_true", help="Rejudge already-scored examples.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap on the number of examples.")
    args = parser.parse_args()

    judge_round_answers(
        input_file=args.input_file,
        output_file=args.output_file,
        judge_model=args.judge_model,
        temperature=args.temperature,
        overwrite=args.overwrite,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
