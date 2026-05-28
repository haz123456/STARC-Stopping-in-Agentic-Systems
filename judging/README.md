# Judging

This folder contains the cleaned judging scripts used to score AgenticLU outputs with ChatGPT.

## Public scripts

- `judge_final_answers.py`
- `judge_round_answers.py`
- `judge_round_answers_twice.py`

## Setup

Install the OpenAI client:

```bash
pip install openai
```

Optional token counting:

```bash
pip install tiktoken
```

Set your API key before running the scripts:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

If you are using PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
```

## Purpose

- `judge_final_answers.py` scores the final answer in a compact results JSON file.
- `judge_round_answers.py` scores round 1, round 2, and final answers in the same file.
- `judge_round_answers_twice.py` runs the round judging twice and writes:
  - run 1 output
  - run 2 output
  - disagreement output

The scripts take a path to a compact results JSON file in the same format as the files produced in the output folders.

## Helpers

- `agenticlu_model_utils.py`
- `_judging_common.py`

## Example

```bash
python judge_final_answers.py --input-file /path/to/results.json
python judge_round_answers.py --input-file /path/to/results.json
python judge_round_answers_twice.py --input-file /path/to/results.json
```

You can also choose the judge model explicitly but the default is gpt-4o, the same as AgenticLU project:

```bash
python judge_round_answers.py \
  --input-file /path/to/results.json \
  --judge-model gpt-4o
```




# Output Files

## API setup

Before running the judging scripts, set `OPENAI_API_KEY`.

Example:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
```

## Final-answer judging

Input:

- compact results JSON file

Output:

- `*_gpt4eval.json`

## Per-round judging

Input:

- compact results JSON file

Output:

- `*_gpt4eval_rounds.json`

This file stores:

- `gpt4_verification_round1`
- `gpt4_verification_round2`
- `gpt4_verification_final`

## Double per-round judging

Input:

- compact results JSON file

Outputs:

- `*_gpt4eval_rounds_run1.json`
- `*_gpt4eval_rounds_run2.json`
- `*_gpt4eval_rounds_disagreements.json`

The disagreement file lists examples and round labels where run 1 and run 2 produced different boolean correctness judgements.

