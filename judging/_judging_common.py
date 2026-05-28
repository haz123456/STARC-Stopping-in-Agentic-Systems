import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

try:
    import tiktoken
except ImportError:
    tiktoken = None


JUDGE_PROMPT = """Please verify the following answer:
Question: {question}
Ground Truth Answers: {ground_truth}
Predicted Answer: {answer}

Your task is to determine whether the predicted answer correctly matches the ground truth.
Focus on overall correctness and provide a detailed explanation in the following JSON format:

{{
  "explanation": "Justification",
  "confidence": 0.0,
  "correct_answer": true
}}

Rules:
- "correct_answer" must be true if the predicted answer is overall correct, otherwise false.
- "confidence" must be a number between 0 and 1.
- Return JSON only.
"""


ROUND_LABELS = ("round1", "round2", "final")


MODEL_PRICING_USD_PER_1M = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "pricing_note": "OpenAI official GPT-4o text pricing"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "pricing_note": "OpenAI official GPT-4o-mini text pricing"},
    "azure/gpt-4o-2024-05-13": {
        "input": 2.50,
        "output": 10.00,
        "pricing_note": "Estimated using OpenAI GPT-4o pricing; Azure pricing may differ",
    },
}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)


def normalize_answer_text(text: Any, prefix: str = "Answer:") -> Optional[str]:
    if text is None:
        return None
    text = str(text).strip()
    text = re.sub(rf"^\s*{re.escape(prefix)}\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("**", "")
    text = re.sub(r"\s+\.", ".", text)
    return " ".join(text.split())


def parse_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    matches = re.findall(r"\{.*\}", text, re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except Exception:
        return None


def is_valid_verification(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("correct_answer"), bool):
        return False
    try:
        confidence = float(obj.get("confidence"))
    except Exception:
        return False
    if not (0.0 <= confidence <= 1.0):
        return False
    return isinstance(obj.get("explanation"), str)


def extract_question(example: Dict[str, Any]) -> Optional[str]:
    for key in ("question", "query", "prompt"):
        if example.get(key) is not None:
            return str(example[key])
    return None


def format_correct_answers(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " | ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_gold_answers(example: Dict[str, Any]) -> Optional[str]:
    for key in ("answers", "answer", "gold_answers", "gold_answer", "reference_answer", "references"):
        if example.get(key) is not None:
            return format_correct_answers(example[key])
    return None


def extract_final_answer(example: Dict[str, Any]) -> Optional[str]:
    for key in ("final_answer_raw", "final_answer", "output", "prediction", "pred", "response"):
        if example.get(key) is not None:
            return normalize_answer_text(example[key])
    return None


def extract_round_answer(example: Dict[str, Any], round_idx: int) -> Optional[str]:
    provisionals = example.get("provisional_answers_by_round") or []
    for record in provisionals:
        if record.get("round") == round_idx:
            return normalize_answer_text(record.get("final_answer"))
    return None


def populate_round_fields(example: Dict[str, Any]) -> None:
    example["round1_answer"] = extract_round_answer(example, 1)
    example["round2_answer"] = extract_round_answer(example, 2)


def canonical_model_name(model_name: Optional[str]) -> str:
    if not model_name:
        return ""
    name = model_name.lower().strip()
    if "azure/gpt-4o-2024-05-13" in name:
        return "azure/gpt-4o-2024-05-13"
    if "gpt-4o-mini" in name:
        return "gpt-4o-mini"
    if "gpt-4o" in name:
        return "gpt-4o"
    return name


def get_model_pricing(model_name: Optional[str]) -> Dict[str, Any]:
    canonical = canonical_model_name(model_name)
    if canonical in MODEL_PRICING_USD_PER_1M:
        return MODEL_PRICING_USD_PER_1M[canonical]
    if "gpt-4o-mini" in canonical:
        return MODEL_PRICING_USD_PER_1M["gpt-4o-mini"]
    if "gpt-4o" in canonical:
        return MODEL_PRICING_USD_PER_1M["gpt-4o"]
    return {"input": 0.0, "output": 0.0, "pricing_note": "No known pricing configured"}


def estimate_tokens_with_tiktoken(text: Optional[str], model_name: Optional[str]) -> Optional[int]:
    if not text:
        return 0
    if tiktoken is None:
        return None
    clean_name = (model_name or "").replace("azure/", "")
    try:
        try:
            encoding = tiktoken.encoding_for_model(clean_name)
        except Exception:
            encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except Exception:
        return None


def rough_token_estimate(text: Optional[str]) -> int:
    if not text:
        return 0
    return max(1, int(round(len(text) / 4.0)))


def extract_token_counts(
    judge_response: Optional[Dict[str, Any]],
    prompt: str,
    raw_output: Optional[str],
    model_name: Optional[str],
) -> Tuple[int, int]:
    input_tokens = None
    output_tokens = None
    if isinstance(judge_response, dict):
        for key in ("input_len", "prompt_tokens", "input_tokens"):
            if isinstance(judge_response.get(key), int):
                input_tokens = judge_response[key]
                break
        for key in ("output_len", "completion_tokens", "output_tokens"):
            if isinstance(judge_response.get(key), int):
                output_tokens = judge_response[key]
                break
    if input_tokens is None:
        input_tokens = estimate_tokens_with_tiktoken(prompt, model_name)
    if output_tokens is None:
        output_tokens = estimate_tokens_with_tiktoken(raw_output, model_name)
    if input_tokens is None:
        input_tokens = rough_token_estimate(prompt)
    if output_tokens is None:
        output_tokens = rough_token_estimate(raw_output)
    return int(input_tokens), int(output_tokens)


def estimate_cost_usd(input_tokens: int, output_tokens: int, pricing: Dict[str, Any]) -> float:
    return (input_tokens / 1_000_000.0) * pricing["input"] + (output_tokens / 1_000_000.0) * pricing["output"]


def round_cost(value: float) -> float:
    return round(float(value), 10)


def judge_one_answer(
    model,
    judge_model_name: str,
    question: str,
    gold_answers: str,
    answer: str,
) -> Dict[str, Any]:
    pricing = get_model_pricing(judge_model_name)
    prompt = JUDGE_PROMPT.format(question=question, ground_truth=gold_answers, answer=answer)
    judge_response = model.generate(prompt=prompt)
    raw_output = judge_response.get("output") if judge_response is not None else None
    parsed = parse_json(raw_output)
    prompt_tokens, completion_tokens = extract_token_counts(
        judge_response=judge_response,
        prompt=prompt,
        raw_output=raw_output,
        model_name=judge_model_name,
    )
    return {
        "verification": parsed if is_valid_verification(parsed) else None,
        "raw_output": raw_output,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_total_cost_usd": round_cost(
            estimate_cost_usd(prompt_tokens, completion_tokens, pricing)
        ),
    }


def build_final_output_path(input_path: str) -> str:
    root, ext = os.path.splitext(input_path)
    return f"{root}_gpt4eval{ext}"


def build_round_output_path(input_path: str) -> str:
    root, ext = os.path.splitext(input_path)
    return f"{root}_gpt4eval_rounds{ext}"


def build_round_run_output_path(input_path: str, run_label: str) -> str:
    root, ext = os.path.splitext(input_path)
    return f"{root}_gpt4eval_rounds_{run_label}{ext}"


def build_round_disagreement_path(input_path: str) -> str:
    root, ext = os.path.splitext(input_path)
    return f"{root}_gpt4eval_rounds_disagreements{ext}"


def merge_existing_final_scores(source_results: Dict[str, Any], existing_results: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(existing_results, dict):
        return source_results
    if not isinstance(source_results.get("data"), list) or not isinstance(existing_results.get("data"), list):
        return source_results
    if len(source_results["data"]) != len(existing_results["data"]):
        return source_results

    merged = deepcopy(source_results)
    for idx, example in enumerate(merged["data"]):
        old_example = existing_results["data"][idx]
        same_example = False
        if example.get("example_id") is not None and old_example.get("example_id") is not None:
            same_example = example.get("example_id") == old_example.get("example_id")
        else:
            same_example = extract_question(example) == extract_question(old_example)
        if same_example and is_valid_verification(old_example.get("gpt4_verification")):
            example["gpt4_verification"] = old_example["gpt4_verification"]
            if "gpt4_verification_raw_output" in old_example:
                example["gpt4_verification_raw_output"] = old_example["gpt4_verification_raw_output"]
    return merged


def merge_existing_round_scores(source_results: Dict[str, Any], existing_results: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(existing_results, dict):
        return source_results
    if not isinstance(source_results.get("data"), list) or not isinstance(existing_results.get("data"), list):
        return source_results
    if len(source_results["data"]) != len(existing_results["data"]):
        return source_results

    merged = deepcopy(source_results)
    for idx, example in enumerate(merged["data"]):
        old_example = existing_results["data"][idx]
        same_example = False
        if example.get("example_id") is not None and old_example.get("example_id") is not None:
            same_example = example.get("example_id") == old_example.get("example_id")
        else:
            same_example = extract_question(example) == extract_question(old_example)
        if not same_example:
            continue
        for label in ROUND_LABELS:
            verification_key = f"gpt4_verification_{label}"
            raw_key = f"gpt4_verification_raw_output_{label}"
            if is_valid_verification(old_example.get(verification_key)):
                example[verification_key] = old_example[verification_key]
                if raw_key in old_example:
                    example[raw_key] = old_example[raw_key]
    return merged


def recompute_final_metrics(results: Dict[str, Any]) -> None:
    correct = 0.0
    confidence = 0.0
    count = 0
    for example in results.get("data", []):
        verification = example.get("gpt4_verification")
        if is_valid_verification(verification):
            correct += 1.0 if verification["correct_answer"] else 0.0
            confidence += float(verification["confidence"])
            count += 1
    results.setdefault("averaged_metrics", {})
    results["averaged_metrics"]["gpt-4-accuracy"] = (100.0 * correct / count) if count else None
    results["averaged_metrics"]["accuracy"] = results["averaged_metrics"]["gpt-4-accuracy"]
    results["averaged_metrics"]["gpt-4-avg-confidence"] = (confidence / count) if count else None


def recompute_round_metrics(results: Dict[str, Any]) -> None:
    results.setdefault("averaged_metrics", {})
    for label in ROUND_LABELS:
        correctness = []
        confidence = []
        for example in results.get("data", []):
            verification = example.get(f"gpt4_verification_{label}")
            if is_valid_verification(verification):
                correctness.append(1.0 if verification["correct_answer"] else 0.0)
                confidence.append(float(verification["confidence"]))
        results["averaged_metrics"][f"gpt-4-accuracy-{label}"] = (
            100.0 * sum(correctness) / len(correctness) if correctness else None
        )
        results["averaged_metrics"][f"gpt-4-avg-confidence-{label}"] = (
            sum(confidence) / len(confidence) if confidence else None
        )
