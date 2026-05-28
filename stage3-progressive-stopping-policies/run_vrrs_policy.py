import argparse
import json

from vrrs_system import run_vrrs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VRRS on a verify-refine augmented full JSONL file.")
    parser.add_argument("--input-file", required=True, help="Verify-refine augmented *.full.jsonl file.")
    parser.add_argument("--labels-file", default=None, help="Optional *_gpt4eval_rounds.json file.")
    parser.add_argument("--output-file", default=None, help="Optional decision-file output path.")
    parser.add_argument("--summary-file", default=None, help="Optional summary JSON output path.")
    args = parser.parse_args()

    result = run_vrrs(
        input_file=args.input_file,
        labels_file=args.labels_file,
        output_file=args.output_file,
        summary_file=args.summary_file,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
