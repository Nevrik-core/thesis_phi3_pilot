# Stage 1 Q4 100-sample run: stage1_q4_100_v2

## Scope

This run evaluates Stage 1 Q4_K_M / Q4 packaged models on 100-sample subsets.

Benchmarks:

- QA: 100 Ukrainian + 100 English examples
- BELEBELE: 100 Ukrainian + 100 English examples
- UAlign: 100 examples per task/language pair

Runtime:

- Ollama local API
- CPU-only requested with `num_gpu=0`
- temperature=0.0
- num_ctx=2048

Important:

- Thinking/reasoning models use larger generation budgets inside the eval scripts.
- `prompt_prefix` and `ollama_think` are taken from `config.py` metadata.
- Outputs include `raw_content`, `thinking`, `clean_text_source`, and `used_thinking_fallback` for later debugging.

## Models

1. `stage1_phi4_mini_instruct_q4_k_m` — Phi-4-mini-instruct Q4_K_M (instruct, think=None)
2. `stage1_phi4_mini_reasoning_q4_k_m` — Phi-4-mini-reasoning Q4_K_M (reasoning_only, think=None)
3. `stage1_qwen3_4b_instruct_2507_q4_k_m` — Qwen3-4B-Instruct-2507 Q4_K_M (instruct, think=False)
4. `stage1_qwen3_4b_thinking_2507_q4_k_m` — Qwen3-4B-Thinking-2507 Q4_K_M (reasoning_only, think=True)
5. `stage1_qwen3_5_4b_q4_k_m` — Qwen3.5-4B Q4_K_M (default_or_hybrid, think=False)
6. `stage1_gemma3_4b_it_q4_k_m` — Gemma-3-4B-it Q4_K_M (instruct, think=None)
7. `stage1_mamaylm_gemma3_4b_it_q4_k_m` — MamayLM-Gemma-3-4B-IT-v1.0 Q4_K_M (instruct, think=None)
8. `stage1_hunyuan_4b_instruct_no_think_q4_k_m` — Hunyuan-4B-Instruct Q4_K_M /no_think (no_think, think=False)
9. `stage1_hunyuan_4b_instruct_think_q4_k_m` — Hunyuan-4B-Instruct Q4_K_M /think (think, think=True)
10. `stage1_minicpm3_4b_q4_k_m` — MiniCPM3-4B Q4_K_M (instruct, think=None)
11. `stage1_llm_jp_3_3_7b_instruct_q4_k_m` — llm-jp-3-3.7b-instruct Q4_K_M (instruct, think=None)
12. `stage1_nemotron3_nano_4b_no_think_q4_k_m` — NVIDIA-Nemotron-3-Nano-4B Q4_K_M /no_think (no_think, think=False)
13. `stage1_nemotron3_nano_4b_think_q4_k_m` — NVIDIA-Nemotron-3-Nano-4B Q4_K_M /think (think, think=True)