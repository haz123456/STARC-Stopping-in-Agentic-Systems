# verify_refine

This subfolder contains the saved-output verify-refine replay used by the stage 3 baseline specifically the VRB policy, and later used in the VRRS and VRRS-BR.

This is LLM output, so as stated in the report it is subject to run by run vairation. As such, this system has been seperated from the main policy run files, and it should be run once on the saved AgenticLU states, so that all policy runs are using the same generation, thus any differnces in policy runs can be attributed to policy attributes rather than random black box LLM generations.



Entry point:

- `run_verify_refine.py`

Notes:

- it depends on `../../stage1-agenticlu-runtime/AgenticLU-Modified/HELMET`
- it replays from saved outputs rather than rerunning the live clarification pipeline
- no outputs or credentials files are included in this copy
