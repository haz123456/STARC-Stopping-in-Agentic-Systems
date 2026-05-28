import argparse
import ast
import os

import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="verify-then-refine evaluation on saved core run outputs"
    )
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    parser.add_argument("--tag", type=str, default="verify-refine", help="tag to add to the output file")

    # model setting
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true", help="whether to use vllm engine")

    # data paths
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--demo_files", type=str, default=None)
    parser.add_argument("--test_files", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="path to save the predictions")
    parser.add_argument(
        "--source_output_dir",
        type=str,
        default=None,
        help="directory containing the saved core-run outputs to verify and refine",
    )
    parser.add_argument(
        "--source_output_path",
        type=str,
        default=None,
        help="explicit path to a saved core-run compact JSON or full JSONL file",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_depths", type=int, default=10)
    parser.add_argument("--resume_from_path", type=str, default=None)

    # dataset specific settings
    parser.add_argument("--popularity_threshold", type=int, default=3)

    # evaluation settings
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--input_max_length", type=str, default="4096")
    parser.add_argument("--num_clarification_rounds", type=int, default=3)

    # verify/refine settings
    parser.add_argument("--verify_refine_enabled", type=ast.literal_eval, choices=[True, False], default=True)
    parser.add_argument("--verify_verifier_generation_max_length", type=int, default=256)
    parser.add_argument("--verify_use_falsifier", type=ast.literal_eval, choices=[True, False], default=True)
    parser.add_argument("--verify_falsifier_generation_max_length", type=int, default=256)

    # generation settings
    parser.add_argument("--do_sample", type=ast.literal_eval, choices=[True, False], default=False)
    parser.add_argument("--generation_max_length", type=str, default="10")
    parser.add_argument("--generation_min_length", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--stop_newline", type=ast.literal_eval, choices=[True, False], default=False)

    # model specific settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--no_torch_compile", action="store_true")
    parser.add_argument("--use_chat_template", type=ast.literal_eval, choices=[True, False], default=False)
    parser.add_argument("--rope_theta", type=int, default=None)

    # misc
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--count_tokens", action="store_true")

    # prompting settings
    parser.add_argument(
        "--prompting_method",
        type=str,
        choices=[None, "stepbystep", "fact&reflect", "plan&solve", "longrag"],
        default=None,
    )

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    if hasattr(args, "stop_new_line") and not hasattr(args, "stop_newline"):
        args.stop_newline = args.stop_new_line
    elif hasattr(args, "stop_new_line") and getattr(args, "stop_new_line") is not None:
        args.stop_newline = args.stop_new_line

    if args.output_dir is None:
        args.output_dir = f"output/{os.path.basename(args.model_name_or_path)}"

    if args.rope_theta is not None:
        args.output_dir = args.output_dir + f"-override-rope{args.rope_theta}"

    return args