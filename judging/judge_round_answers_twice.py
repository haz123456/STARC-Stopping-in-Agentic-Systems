import argparse
from typing import Any, Dict, List, Optional

from _judging_common import (
    ROUND_LABELS,
    build_round_disagreement_path,
    build_round_run_output_path,
    extract_question,
    load_json,
    populate_round_fields,
    save_json,
)
from judge_round_answers import judge_round_answers


def compare_round_judgements(run1_results: Dict[str, Any], run2_results: Dict[str, Any]) -> Dict[str, Any]:
    disagreements: List[Dict[str, Any]] = []
    run1_examples = run1_results.get("data", [])
    run2_examples = run2_results.get("data", [])

    for index, (run1_example, run2_example) in enumerate(zip(run1_examples, run2_examples)):
        populate_round_fields(run1_example)
        populate_round_fields(run2_example)
        question = extract_question(run1_example)

        for label in ROUND_LABELS:
            verification_key = f"gpt4_verification_{label}"
            run1_verification = run1_example.get(verification_key)
            run2_verification = run2_example.get(verification_key)
            run1_correct = None if not isinstance(run1_verification, dict) else run1_verification.get("correct_answer")
            run2_correct = None if not isinstance(run2_verification, dict) else run2_verification.get("correct_answer")
            if run1_correct == run2_correct:
                continue

            field_name = "final_answer_raw" if label == "final" else f"{label}_answer"
            disagreements.append(
                {
                    "example_idx": index,
                    "example_id": run1_example.get("example_id"),
                    "question": question,
                    "round_label": label,
                    "predicted_answer": run1_example.get(field_name),
                    "run1_verification": run1_verification,
                    "run2_verification": run2_verification,
                }
            )

    return {
        "num_examples": len(run1_examples),
        "num_disagreements": len(disagreements),
        "disagreements": disagreements,
    }


def judge_round_answers_twice(
    input_file: str,
    output_file_run1: Optional[str] = None,
    output_file_run2: Optional[str] = None,
    disagreements_file: Optional[str] = None,
    judge_model: str = "gpt-4o",
    temperature: float = 0.1,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    run1_output = output_file_run1 or build_round_run_output_path(input_file, "run1")
    run2_output = output_file_run2 or build_round_run_output_path(input_file, "run2")
    disagreement_output = disagreements_file or build_round_disagreement_path(input_file)

    run1_results = judge_round_answers(
        input_file=input_file,
        output_file=run1_output,
        judge_model=judge_model,
        temperature=temperature,
        overwrite=True,
        max_examples=max_examples,
    )
    run2_results = judge_round_answers(
        input_file=input_file,
        output_file=run2_output,
        judge_model=judge_model,
        temperature=temperature,
        overwrite=True,
        max_examples=max_examples,
    )

    disagreements = compare_round_judgements(load_json(run1_output), load_json(run2_output))
    disagreements["source_results_file"] = input_file
    disagreements["run1_output_file"] = run1_output
    disagreements["run2_output_file"] = run2_output
    save_json(disagreement_output, disagreements)
    return disagreements


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run per-round judging twice and write run 1, run 2, and disagreement outputs."
    )
    parser.add_argument("--input-file", required=True, help="Compact results JSON file.")
    parser.add_argument("--output-file-run1", default=None, help="Optional judged output path for run 1.")
    parser.add_argument("--output-file-run2", default=None, help="Optional judged output path for run 2.")
    parser.add_argument("--disagreements-file", default=None, help="Optional disagreement output path.")
    parser.add_argument("--judge-model", default="gpt-4o", help="OpenAI judge model.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Judge temperature.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap on the number of examples.")
    args = parser.parse_args()

    judge_round_answers_twice(
        input_file=args.input_file,
        output_file_run1=args.output_file_run1,
        output_file_run2=args.output_file_run2,
        disagreements_file=args.disagreements_file,
        judge_model=args.judge_model,
        temperature=args.temperature,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
