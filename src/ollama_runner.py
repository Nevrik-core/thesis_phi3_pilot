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


def call_ollama_chat(
    prompt: str,
    model: str = OLLAMA_MODEL_NAME,
    temperature: float = 0.0,
    max_new_tokens: int = 32,
    num_ctx: int = 4096,
    num_gpu: int | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """
    Викликає локальний Ollama /api/chat у нестрімінговому режимі
    та повертає текст відповіді + технічні метрики + resource metrics.

    Важливо:
    - client_process_* = Python script/client RAM
    - model_process_* = Ollama/llama/runner RAM
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"

    options = {
        "temperature": temperature,
        "num_predict": max_new_tokens,
        "num_ctx": num_ctx,
        "seed": 42,
    }

    if num_gpu is not None:
        options["num_gpu"] = num_gpu

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "options": options,
    }

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

    message = data.get("message", {})
    text = message.get("content", "").strip()

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
        "text": text,
        "model": data.get("model", model),
        "requested_num_gpu": num_gpu,

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