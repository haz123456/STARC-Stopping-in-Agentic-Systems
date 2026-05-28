# File Guide

## Runtime root

- `AgenticLU-Modified/README.md`
- `AgenticLU-Modified/HELMET/`
- `AgenticLU-Modified/long_context_llm/`
- `AgenticLU-Modified/openrlhf/`

## Files most relevant to the thesis runtime

- `AgenticLU-Modified/HELMET/eval_agent.py`
- `AgenticLU-Modified/HELMET/state_saving.py`
- `AgenticLU-Modified/HELMET/arguments.py`

## Output note

The live runtime writes compact JSON results and per-example `*.full.jsonl` records.

The `*.full.jsonl` files are the stage 1 artefacts later consumed by the verify-refine step and then by the stage 3 offline stopping policies.
