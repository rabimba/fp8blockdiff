"""Real dataset loaders and lossless quality evaluators for Phase 1 & Phase 3:
1. Code synthesis: openai/openai_humaneval (with MBPP backup)
2. Math reasoning: openai/gsm8k (with exact numeric answer extractor & grader)
3. Open-ended dialogue / summarization: HuggingFaceH4/ultrachat_200k

Zero synthetic dummy prompt strings: every prompt is loaded from real Hugging Face datasets.
"""

from __future__ import annotations

import dataclasses
import random
import re
from typing import Any


@dataclasses.dataclass(frozen=True)
class EvalPrompt:
    """Single real evaluation prompt with ground-truth reference metadata."""

    prompt_id: str
    suite: str
    prompt_text: str
    reference_answer: str | None = None
    canonical_test_code: str | None = None
    entry_point: str | None = None


def _require_datasets_lib() -> Any:
    try:
        import datasets

        return datasets
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' library is required to load real HumanEval, GSM8K, and "
            "UltraChat benchmarks. Run `pip install datasets>=2.20.0` or `scripts/phase0_discover.py`."
        ) from exc


def load_code_synthesis_prompts(
    warmup_count: int = 8,
    eval_count: int = 50,
    seed: int = 42,
) -> tuple[list[EvalPrompt], list[EvalPrompt]]:
    """Load real Python code synthesis prompts from openai/openai_humaneval."""
    datasets = _require_datasets_lib()
    ds = datasets.load_dataset("openai/openai_humaneval", split="test")
    rows = list(ds)
    rng = random.Random(seed)
    rng.shuffle(rows)

    needed = warmup_count + eval_count
    if len(rows) < needed:
        mbpp = datasets.load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        for item in mbpp:
            rows.append(
                {
                    "task_id": f"MBPP/{item['task_id']}",
                    "prompt": f"{item['prompt']}\n\n# Reference signature context:\n{item['code'].splitlines()[0]}\n",
                    "canonical_solution": item["code"],
                    "test": "\n".join(item.get("test_list", [])),
                    "entry_point": None,
                }
            )
            if len(rows) >= needed:
                break

    prompts: list[EvalPrompt] = []
    for row in rows[:needed]:
        raw_prompt = str(row["prompt"]).strip()
        formatted = (
            "Complete the following Python function accurately and concisely. "
            "Output valid Python code implementing the function body:\n\n"
            f"{raw_prompt}\n"
        )
        prompts.append(
            EvalPrompt(
                prompt_id=str(row["task_id"]),
                suite="code_synthesis",
                prompt_text=formatted,
                reference_answer=str(row.get("canonical_solution", "")),
                canonical_test_code=str(row.get("test", "")),
                entry_point=row.get("entry_point"),
            )
        )
    return prompts[:warmup_count], prompts[warmup_count:needed]


def load_math_reasoning_prompts(
    warmup_count: int = 8,
    eval_count: int = 50,
    seed: int = 42,
) -> tuple[list[EvalPrompt], list[EvalPrompt]]:
    """Load real grade-school math reasoning problems from openai/gsm8k."""
    datasets = _require_datasets_lib()
    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    rows = list(ds)
    rng = random.Random(seed)
    rng.shuffle(rows)

    needed = warmup_count + eval_count
    prompts: list[EvalPrompt] = []
    for idx, row in enumerate(rows[:needed]):
        question = str(row["question"]).strip()
        answer_raw = str(row["answer"])
        gold_num = extract_gsm8k_gold_number(answer_raw)
        formatted = (
            "Solve the following math problem step by step. "
            "End your response with the final numeric answer formatted as `#### <number>`.\n\n"
            f"Question: {question}\nAnswer:\n"
        )
        prompts.append(
            EvalPrompt(
                prompt_id=f"GSM8K/{idx}",
                suite="math_reasoning",
                prompt_text=formatted,
                reference_answer=gold_num,
            )
        )
    return prompts[:warmup_count], prompts[warmup_count:needed]


def load_dialogue_summarization_prompts(
    warmup_count: int = 8,
    eval_count: int = 50,
    min_chars: int = 600,
    max_chars: int = 6000,
    seed: int = 42,
) -> tuple[list[EvalPrompt], list[EvalPrompt]]:
    """Load real multi-turn / long-context prompts from HuggingFaceH4/ultrachat_200k."""
    datasets = _require_datasets_lib()
    # Stream so we do not waste disk downloading unused shards
    ds_iter = datasets.load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True
    )
    needed = warmup_count + eval_count
    pool_target = max(needed * 3, 180)
    candidates: list[EvalPrompt] = []

    for idx, row in enumerate(ds_iter):
        messages = row.get("messages") or []
        user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
        assistant_turns = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
        if not user_turns:
            continue
        # Combine up to two user turns + prior context for realistic long-context dialogue/summarization
        if len(messages) >= 3:
            context_block = (
                f"User: {messages[0].get('content', '')}\n\n"
                f"Assistant: {messages[1].get('content', '')}\n\n"
                f"User: {messages[2].get('content', '')}\n\nAssistant:"
            )
        else:
            context_block = f"User: {user_turns[0]}\n\nAssistant:"

        if min_chars <= len(context_block) <= max_chars:
            candidates.append(
                EvalPrompt(
                    prompt_id=f"UltraChat/{row.get('prompt_id', idx)}",
                    suite="dialogue_summarization",
                    prompt_text=context_block,
                    reference_answer=assistant_turns[0] if assistant_turns else None,
                )
            )
            if len(candidates) >= pool_target:
                break

    if len(candidates) < needed:
        raise RuntimeError(
            f"Collected only {len(candidates)} dialogue prompts matching length filter "
            f"[{min_chars}, {max_chars}], needed {needed}."
        )

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:needed]
    return selected[:warmup_count], selected[warmup_count:needed]


def load_phase1_smoke_16_prompts(seed: int = 42) -> list[EvalPrompt]:
    """Load a balanced 16-prompt smoke set across Code (6), Math (5), and Dialogue (5)."""
    _, code_eval = load_code_synthesis_prompts(warmup_count=0, eval_count=6, seed=seed)
    _, math_eval = load_math_reasoning_prompts(warmup_count=0, eval_count=5, seed=seed)
    _, chat_eval = load_dialogue_summarization_prompts(warmup_count=0, eval_count=5, seed=seed)
    return code_eval + math_eval + chat_eval


_GSM8K_HASH_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_LAST_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_gsm8k_gold_number(answer_text: str) -> str:
    match = _GSM8K_HASH_RE.search(answer_text)
    if match:
        return match.group(1).replace(",", "")
    nums = _LAST_NUM_RE.findall(answer_text)
    return nums[-1].replace(",", "") if nums else ""


def extract_predicted_gsm8k_number(generated_text: str) -> str:
    match = _GSM8K_HASH_RE.search(generated_text)
    if match:
        return match.group(1).replace(",", "")
    nums = _LAST_NUM_RE.findall(generated_text)
    return nums[-1].replace(",", "") if nums else ""


def grade_gsm8k_prediction(generated_text: str, gold_number: str) -> bool:
    pred = extract_predicted_gsm8k_number(generated_text)
    if not pred or not gold_number:
        return False
    try:
        return abs(float(pred) - float(gold_number)) < 1e-4
    except ValueError:
        return pred.strip() == gold_number.strip()


def evaluate_humaneval_functional_pass(
    prompt_obj: EvalPrompt, generated_text: str, timeout_sec: float = 4.0
) -> bool:
    """Execute extracted Python completion against HumanEval canonical unit tests in a child process."""
    import subprocess
    import sys

    if not prompt_obj.canonical_test_code or not prompt_obj.entry_point:
        return False

    code_body = _extract_python_code(generated_text)
    # Reconstruct original function header from prompt_text
    marker = "implementing the function body:\n\n"
    header = (
        prompt_obj.prompt_text.split(marker, 1)[1]
        if marker in prompt_obj.prompt_text
        else ""
    )
    if f"def {prompt_obj.entry_point}" in code_body:
        full_prog = f"{code_body}\n\n{prompt_obj.canonical_test_code}\ncheck({prompt_obj.entry_point})\n"
    else:
        full_prog = f"{header}\n{code_body}\n\n{prompt_obj.canonical_test_code}\ncheck({prompt_obj.entry_point})\n"

    try:
        res = subprocess.run(
            [sys.executable, "-I", "-c", full_prog],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _extract_python_code(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced[0]
    return text
