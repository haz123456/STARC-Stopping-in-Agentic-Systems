# STARC: State Aware Risk Control for the Stopping Decision in Agentic Systems

This repository accompanies my honours thesis *State Aware Risk Control (STARC) for the Stopping Decision in Agentic Systems*, submitted in fulfilment of INFO4002 at the School of Computer Science, The University of Sydney.

**Author:** Harry Kember  
**Supervisor:** Ying Zhou  
**Submitted:** June 2026  
**Degree:** Bachelor of Advanced Computing (Computational Data Science) and Bachelor of Science (Mathematics)

---

## Abstract

Large Language Models are increasingly deployed as agentic question-answering systems that clarify, gather evidence, and refine an answer over several rounds rather than producing it in a single pass. Clarification is widely assumed to be beneficial, but this is not always so. Further clarification can repair a weak answer, but it can equally waste computation, entrench a wrong reasoning path, or push a correct answer off course. Existing approaches make the stop or continue decision from a model's confidence in its answer, a signal that is known to be unreliable.

This thesis proposes that the decision should be made not from the model's confidence in its answer, but from the evidence carried in the agentic system state that produced it. STARC is an architecture which evaluates the agentic state and produces a stop or continue decision. It is implemented through three progressively layered policies: the Verification and Refinement Baseline (VRB), Verification and Refinement with Risk Signals (VRRS), and Verification and Refinement with Risk Signals and Borderline Review (VRRS-BR).

The strongest policy, VRRS-BR, raised continue recall from 45% to 67% over the simplest policy. This gain does not come at the expense of overall judgement quality: balanced accuracy rose from 66% to 74% and the Matthews correlation coefficient from 0.35 to 0.45 alongside it. Knowing when to stop in agentic question-answering is a judgement to be read from the agentic system state, not from confidence in the answer.

---

## What this repository contains

This codebase produces the empirical results reported in the thesis. An instrumented agentic question answering runtime saves the full system state at the first clarification boundary, and a family of stopping policies, culminating in STARC, is then evaluated offline against those saved states.

The pipeline is organised into four stages:

```
.
├── stage1-agenticlu-runtime/                Instrumented AgenticLU runtime; produces saved round state files
├── stage2-round2-case-study/                Round-two case study evaluation (the four mechanism analysis)
├── stage3-progressive-stopping-policies/    Offline STARC stopping policy replay, retained outputs, and statistical compiler
└── judging/                                 OpenAI answer judging method
```

---

## Pipeline overview

The thesis frames the stop or continue choice in agentic QA as a control problem at the first clarification boundary. The codebase reflects that framing directly.

1. **Stage 1, Runtime instrumentation.** A modified AgenticLU runtime executes the Chain of Clarifications (CoC) workflow on the Llama-3.1-8B-Instruct base model at a 128k token context, and saves the full state at the first clarification boundary: the provisional answer, the pinned context, the clarification question, and the surrounding trace. The boundary becomes a structured object that can be read, classified, and compared.
2. **Stage 2, Round-two case study.** A second clarification round is run over the saved states, and the outcomes are classified into the four mechanisms reported in the thesis: repair of a wrong answer, preservation of an already correct one, persistent failure, and degradation of a correct answer.
3. **Stage 3, STARC stopping policy replay.** The three policies that implement STARC, namely the Verification and Refinement Baseline (VRB), Verification and Refinement with Risk Signals (VRRS), and Verification and Refinement with Risk Signals and Borderline Review (VRRS-BR), are evaluated offline against the saved states. The folder also holds the retained replay output subset and the statistical compiler that produces the tables and confidence intervals reported in Chapter 5 of the thesis.
4. **Judging.** Wherever an answer must be scored against ground truth, Stages 1 to 3 invoke the OpenAI judging method in `judging/`.

---

## Datasets

The evaluation suite is drawn from four open domain QA datasets, used in the long context setting via prompt packing of distractor passages:

- **Natural Questions**
- **PopQA**
- **TriviaQA**
- **HotpotQA**

The exact dataset versions, sampling, and preprocessing are documented in Chapter 3 of the thesis.

---

## Requirements

- Python 3.12
- A Google Colab environment with an A100 GPU, for Stage 1 (the AgenticLU runtime is configured to run the Llama-3.1-8B-Instruct base model at a 128k token context, and the A100 is the smallest configuration on which that fits)
- An OpenAI API key, for the `judging/` answer correctness judge

Stage-specific Python dependencies are pinned in each stage's `requirements.txt`. A `.env.example` is provided in each stage that needs credentials; copy it to `.env` and fill in the values.

---

## Reproducing the headline results

The thesis reports that STARC, in its full VRRS-BR form, raises continue recall from 44.66% to 66.99%, balanced accuracy from 66.10% to 73.99%, and Matthews correlation coefficient from 0.346 to 0.450 against the VRB baseline. These numbers are produced by running Stage 1 end-to-end on all four datasets, then running the Stage 3 statistical compiler over the resulting replay outputs.

The retained replay output subset is checked into `stage3-progressive-stopping-policies/` so that the statistical compiler can be run again without re-executing the LLM dependent stages. Re-running from scratch requires access to the backbone model and the judge API, and is significantly more expensive and introduces run bias.

---

## Repository structure (detail)

```
judging/
    OpenAI answer judging method evaluation scripts. Used by all stages
    wherever an answer must be scored against ground truth.

stage1-agenticlu-runtime/
    Modified AgenticLU runtime. Executes the Chain of Clarifications
    workflow on Llama-3.1-8B-Instruct at a 128k token context, and
    produces the saved first-round state files consumed downstream.

stage2-round2-case-study/
    Round-two case study report only. The pipeline used to classify
    outcomes into the four mechanisms (repair, preservation, persistent
    failure, degradation).

stage3-progressive-stopping-policies/
    Offline stopping-policy replay code (VRB, VRRS, VRRS-BR), the
    retained replay-output subset, and the statistical compiler that
    produces the tables and confidence intervals in Chapter 5. Consumes
    saved outputs from Stage 1 and the later verify-refine augmented
    files.
```

---

## Relationship to the thesis

| Thesis chapter or section                       | Code location                           |
|-------------------------------------------------|-----------------------------------------|
| Chapter 3, Background: AgenticLU and CoC        | `stage1-agenticlu-runtime/`             |
| Chapter 4.1 to 4.2, Round-state instrumentation | `stage1-agenticlu-runtime/`             |
| Chapter 4.3, Round-two case study               | `stage2-round2-case-study/`             |
| Chapter 4.4, VRB, VRRS, VRRS-BR                 | `stage3-progressive-stopping-policies/` |
| Chapter 4.5, Answer-equivalence judge           | `judging/`                              |
| Chapter 5, Results (tables and intervals)       | `stage3-progressive-stopping-policies/` |

---

## Citation

If you refer to this work, please cite the thesis:

```
@thesis{kember2026starc,
  author = {Harry Kember},
  title  = {State Aware Risk Control (STARC) for the Stopping Decision in Agentic Question-Answering},
  school = {The University of Sydney, School of Computer Science},
  year   = {2026},
  type   = {Honours Thesis (INFO4002)},
  note   = {Supervisor: Ying Zhou}
}
```

---

## Acknowledgements

This work was supervised by Ying Zhou.

---

## Licence

This repository is released under the MIT Licence. See `LICENSE` for the full text.