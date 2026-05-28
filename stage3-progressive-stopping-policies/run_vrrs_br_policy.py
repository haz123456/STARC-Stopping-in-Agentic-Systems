import argparse
import json
from vrrs_br_system import run_vrrs_br


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VRRS-BR on a VRRS decision JSONL file.")
    parser.add_argument("--input-file", required=True, help="VRRS decision file produced by run_vrrs_policy.py.")
    parser.add_argument("--labels-file", default=None, help="Optional *_gpt4eval_rounds.json file.")
    parser.add_argument("--output-file", default=None, help="Optional decision-file output path.")
    parser.add_argument("--summary-file", default=None, help="Optional summary JSON output path.")
    parser.add_argument("--api-key", default=None, help="OpenAI API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--model", default="gpt-4o", help="Model used for borderline review.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for borderline review.")
    parser.add_argument("--review-limit", type=int, default=None, help="Optional cap on borderline GPT reviews.")
    args = parser.parse_args()

    result = run_vrrs_br(
        input_file=args.input_file,
        labels_file=args.labels_file,
        output_file=args.output_file,
        summary_file=args.summary_file,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        review_limit=args.review_limit,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
