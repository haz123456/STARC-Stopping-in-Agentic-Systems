# Stage 1 AgenticLU Runtime

This folder contains the cleaned submission copy of the modified AgenticLU runtime used in stage 1.

## What is included

- `AgenticLU-Modified/`
- `docs/`

## Purpose

The runtime was used to run AgenticLU live and save the clarification state for later offline analysis.

The thesis specific runtime change here is the round state logging written into the `*.full.jsonl` output files. The live run is driven by `HELMET/eval_agent.py`, and the progressive save/write helpers are collected in `HELMET/state_saving.py`.

The implementation saved:

- each clarification round state
- the provisional answer after each round
- stopping metadata such as rounds used and stop point

## Submission cleanup

The submission copy was reorganised so the state-saving logic sits in one helper module called by `eval_agent.py`. The saved format was not changed.

