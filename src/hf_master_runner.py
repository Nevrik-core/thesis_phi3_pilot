import time
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "microsoft/Phi-4-mini-instruct"

_model = None
_tokenizer = None
_load_duration_sec = None
_rss_mb_after_load = None


def get_rss_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def load_master_model():
    global _model, _tokenizer, _load_duration_sec, _rss_mb_after_load

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer, _load_duration_sec, _rss_mb_after_load

    print("Loading master model...")

    t0 = time.perf_counter()

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    _model.eval()

    t1 = time.perf_counter()

    _load_duration_sec = t1 - t0
    _rss_mb_after_load = get_rss_mb()

    print("Master model loaded.")

    return _model, _tokenizer, _load_duration_sec, _rss_mb_after_load


def call_hf_master_chat(
    prompt: str,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    model, tokenizer, load_duration_sec, rss_mb_after_load = load_master_model()

    mem_before_gen_mb = get_rss_mb()

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_token_count = int(inputs["input_ids"].shape[1])

    t0 = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    t1 = time.perf_counter()

    total_gen_time_sec = t1 - t0

    generated_ids = output_ids[0][prompt_token_count:]
    generated_token_count = int(generated_ids.shape[0])

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    mem_after_gen_mb = get_rss_mb()

    effective_total_tokens_per_sec = None
    total_token_count = prompt_token_count + generated_token_count
    if total_gen_time_sec > 0:
        effective_total_tokens_per_sec = total_token_count / total_gen_time_sec

    effective_generation_tokens_per_sec = None
    if total_gen_time_sec > 0:
        effective_generation_tokens_per_sec = generated_token_count / total_gen_time_sec

    return {
        "text": text,
        "wall_time_sec": total_gen_time_sec,
        "load_duration_sec": load_duration_sec,
        "total_duration_sec": total_gen_time_sec,
        "prompt_token_count": prompt_token_count,
        "generated_token_count": generated_token_count,
        "effective_total_tokens_per_sec": effective_total_tokens_per_sec,
        "effective_generation_tokens_per_sec": effective_generation_tokens_per_sec,
        "rss_mb_after_load": rss_mb_after_load,
        "rss_mb_before_gen": mem_before_gen_mb,
        "rss_mb_after_gen": mem_after_gen_mb,
    }