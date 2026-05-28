# Runtime Changes

The stage 1 keeps the upstream AgenticLU code in place and only documents the thesis runtime additions.

## Scope

The runtime modification was limited to end-of-round state saving for the live AgenticLU run.

In the submission copy, the save machinery is grouped in:

- `AgenticLU-Modified/HELMET/state_saving.py`

This module is called by:

- `AgenticLU-Modified/HELMET/eval_agent.py`

This saved enough information for later offline replay:

- `clarification_rounds`
- `provisional_answers_by_round`
- `num_clarification_rounds_used`
- `stopped_after_round`
- `adaptive_stop_records` when present
- the full per-example `*.full.jsonl` log record


and more!



## Main files

- `AgenticLU-Modified/HELMET/eval_agent.py`
- `AgenticLU-Modified/HELMET/state_saving.py`

The main saved-output helpers are:

- `build_compact_result_record(...)`
- `build_full_log_record(...)`
- `append_trace(...)`
- `save_compact_results(...)`


