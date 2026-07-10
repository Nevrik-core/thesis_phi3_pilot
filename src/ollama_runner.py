import re
import time
from typing import Any

import requests

from resource_monitor import ResourceMonitor


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "phi4-mini"

OLLAMA_PROCESS_TERMS = (
    "ollama",
    "llama",
    "runner",
)


def ns_to_sec(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000


def strip_xml_like_tag_block(text: str, tag: str) -> str:
    pattern = rf"<\s*{tag}\s*>.*?<\s*/\s*{tag}\s*>"
    return re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)


def extract_tag_content(text: str, tag: str) -> str | None:
    pattern = rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)

    if not match:
        return None

    value = match.group(1).strip()
    return value if value else None


def normalize_clean_text(text: str) -> str:
    out = text.strip()

    out = re.sub(r"\s+", " ", out).strip()
    out = out.strip(" \n\r\t")

    return out


def extract_final_answer_from_content(content: str) -> tuple[str, str]:
    """
    Returns:
    - cleaned final answer
    - extraction source label

    This is intentionally conservative:
    1. Prefer explicit <answer>...</answer>.
    2. Remove <think>...</think>.
    3. Remove leftover answer tags if present.
    4. Return cleaned text.
    """
    if not isinstance(content, str):
        return "", "empty_content"

    raw = content.strip()

    if not raw:
        return "", "empty_content"

    answer_content = extract_tag_content(raw, "answer")
    if answer_content:
        return normalize_clean_text(answer_content), "content_answer_tag"

    without_think = strip_xml_like_tag_block(raw, "think")

    without_think = re.sub(
        r"<\s*/?\s*answer\s*>",
        " ",
        without_think,
        flags=re.IGNORECASE,
    )

    cleaned = normalize_clean_text(without_think)

    if cleaned:
        return cleaned, "content_cleaned"

    return "", "content_only_thinking_or_tags"


def extract_final_answer_from_thinking(thinking: str) -> tuple[str, str]:
    """
    Fallback for models that put everything into message.thinking and leave
    message.content empty.

    This is only a rescue path for smoke / problematic reasoning models.
    It prefers <answer>...</answer>, then tries to find a final answer-like line.
    """
    if not isinstance(thinking, str):
        return "", "empty_thinking"

    raw = thinking.strip()

    if not raw:
        return "", "empty_thinking"

    answer_content = extract_tag_content(raw, "answer")
    if answer_content:
        return normalize_clean_text(answer_content), "thinking_answer_tag"

    # Remove embedded thinking tags if the model included them in thinking text.
    cleaned = strip_xml_like_tag_block(raw, "think")
    cleaned = normalize_clean_text(cleaned)

    if not cleaned:
        cleaned = normalize_clean_text(raw)

    # Try to find explicit final answer phrases.
    patterns = [
        r"(?:final answer|answer|відповідь)\s*[:\-]\s*(.+)$",
        r"(?:the answer is|correct answer is)\s+(.+)$",
        r"(?:відповідь є|правильна відповідь)\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)

        if match:
            value = normalize_clean_text(match.group(1))
            if value:
                return value, "thinking_final_phrase"

    # Very conservative short-output rescue.
    # Useful when thinking is just "A" or "1", but avoids returning long CoT as answer.
    if len(cleaned) <= 40:
        return cleaned, "thinking_short_fallback"

    return "", "thinking_not_used_long"


def clean_ollama_message_text(
    content: str,
    thinking: str,
    allow_thinking_fallback: bool = True,
) -> dict[str, Any]:
    content_answer, content_source = extract_final_answer_from_content(content)

    if content_answer:
        return {
            "clean_text": content_answer,
            "clean_text_source": content_source,
            "used_thinking_fallback": 0,
        }

    if allow_thinking_fallback:
        thinking_answer, thinking_source = extract_final_answer_from_thinking(thinking)

        if thinking_answer:
            return {
                "clean_text": thinking_answer,
                "clean_text_source": thinking_source,
                "used_thinking_fallback": 1,
            }

    return {
        "clean_text": "",
        "clean_text_source": content_source,
        "used_thinking_fallback": 0,
    }


def call_ollama_chat(
    prompt: str,
    model: str = OLLAMA_MODEL_NAME,
    temperature: float = 0.0,
    max_new_tokens: int = 32,
    num_ctx: int = 4096,
    num_gpu: int | None = None,
    timeout: int = 300,
    prompt_prefix: str = "",
    ollama_think: bool | None = None,
    allow_thinking_fallback: bool = True,
) -> dict[str, Any]:
    url = f"{OLLAMA_BASE_URL}/api/chat"

    final_prompt = f"{prompt_prefix}{prompt}" if prompt_prefix else prompt

    options = {
        "temperature": temperature,
        "num_predict": max_new_tokens,
        "num_ctx": num_ctx,
        "seed": 42,
    }

    if num_gpu is not None:
        options["num_gpu"] = num_gpu

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": final_prompt,
            }
        ],
        "stream": False,
        "options": options,
    }

    if ollama_think is not None:
        payload["think"] = ollama_think

    with ResourceMonitor(
        external_process_terms=OLLAMA_PROCESS_TERMS,
        poll_interval_sec=0.25,
    ) as monitor:
        wall_t0 = time.perf_counter()
        response = requests.post(url, json=payload, timeout=timeout)
        wall_t1 = time.perf_counter()

    response.raise_for_status()
    data = response.json()

    resource_metrics = monitor.metrics().to_dict(
        current_label="client_process",
        external_label="model_process",
    )

    message = data.get("message", {}) or {}

    raw_content = (message.get("content") or "").strip()
    raw_thinking = (message.get("thinking") or "").strip()

    clean_result = clean_ollama_message_text(
        content=raw_content,
        thinking=raw_thinking,
        allow_thinking_fallback=allow_thinking_fallback,
    )

    clean_text = clean_result["clean_text"]

    prompt_eval_count = data.get("prompt_eval_count")
    prompt_eval_duration = ns_to_sec(data.get("prompt_eval_duration"))

    eval_count = data.get("eval_count")
    eval_duration = ns_to_sec(data.get("eval_duration"))

    total_duration = ns_to_sec(data.get("total_duration"))
    load_duration = ns_to_sec(data.get("load_duration"))

    prompt_tokens_per_sec = None
    if prompt_eval_count and prompt_eval_duration and prompt_eval_duration > 0:
        prompt_tokens_per_sec = prompt_eval_count / prompt_eval_duration

    generation_tokens_per_sec = None
    if eval_count and eval_duration and eval_duration > 0:
        generation_tokens_per_sec = eval_count / eval_duration

    return {
        # Backward-compatible field used by existing eval scripts.
        "text": clean_text,

        # Explicit text fields for debugging.
        "clean_text": clean_text,
        "clean_text_source": clean_result["clean_text_source"],
        "used_thinking_fallback": clean_result["used_thinking_fallback"],
        "raw_content": raw_content,
        "thinking": raw_thinking,
        "has_thinking": int(bool(raw_thinking)),

        "model": data.get("model", model),
        "requested_num_gpu": num_gpu,
        "requested_ollama_think": ollama_think,

        "wall_time_sec": wall_t1 - wall_t0,
        "total_duration_sec": total_duration,
        "load_duration_sec": load_duration,

        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_sec": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration_sec": eval_duration,

        "prompt_tokens_per_sec": prompt_tokens_per_sec,
        "generation_tokens_per_sec": generation_tokens_per_sec,

        **resource_metrics,

        "raw_response": data,
    }