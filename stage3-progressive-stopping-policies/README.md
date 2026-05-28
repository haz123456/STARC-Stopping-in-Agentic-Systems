# Stage 3 Progressive Stopping Policies

This folder is the cleaned submission copy of the stage 3 offline replay code.

It contains three round-1 stopping policies and follows the pipeline used in the thesis:

1. AgenticLU live runtime produces a saved output file.
2. `verify_refine/` reads that file and writes a new verify-refine augmented `*.full.jsonl` file.
3. `VRB` and `VRRS` both read that same verify-refine augmented file and each write a decision file.
4. `VRRS-BR` reads the `VRRS` decision file and only reviews the borderline rows.

## Layout

- `vrb_system.py`
- `vrrs_system.py`
- `vrrs_br_system.py`
- `stage3_common.py`
- `run_vrb_policy.py`
- `run_vrrs_policy.py`
- `run_vrrs_br_policy.py`
- `compare_stopping_policies.py`
- `compile_statistical_analysis.py`
- `verify_refine/`

## Usage

```bash
python run_vrb_policy.py \
  --input-file /path/to/run.full.jsonl \
  --labels-file /path/to/run_gpt4eval_rounds.json
```

```bash
python run_vrrs_policy.py \
  --input-file /path/to/run.full.jsonl \
  --labels-file /path/to/run_gpt4eval_rounds.json
```

```bash
python run_vrrs_br_policy.py \
  --input-file /path/to/run__vrrs_decisions.jsonl \
  --labels-file /path/to/run_gpt4eval_rounds.json \
  --model gpt-4o
```

```bash
python compare_stopping_policies.py --files \
  /path/to/run__vrb_decisions.jsonl \
  /path/to/run__vrrs_decisions.jsonl \
  /path/to/run__vrrs_br_decisions.jsonl
```

By default the three policy runners now write into `output/`.

The kept replay output files also live in `output/`:

- `replay_vrb_decisions.jsonl`
- `replay_vrrs_decisions.jsonl`
- `replay_vrrs_br_decisions.jsonl`
- `replay_round1_labels.jsonl`

To rebuild the policy tables and significance outputs:

```bash
python compile_statistical_analysis.py
```

This writes JSON and CSV outputs to `output/statistics/`.
