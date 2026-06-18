from ollama_runner import call_ollama_chat

from config import OLLAMA_NUM_GPU


def main():
    prompt = """Прочитай контекст і дай коротку точну відповідь на запитання.
Відповідай лише самою відповіддю, без пояснень.

Контекст:
Нормани були народом, який у X і XI століттях дав назву Нормандії, регіону у Франції.

Запитання:
У якій країні розташована Нормандія?

Відповідь:"""

    result = call_ollama_chat(
        prompt=prompt,
        model="phi4-mini",
        temperature=0.0,
        max_new_tokens=16,
        num_ctx=2048,
        num_gpu=0,
    )

    print("\n=== MODEL OUTPUT ===")
    print(result["text"])

    print("\n=== METRICS ===")
    print(f"wall_time_sec: {result['wall_time_sec']:.3f}")
    print(f"total_duration_sec: {result['total_duration_sec']}")
    print(f"load_duration_sec: {result['load_duration_sec']}")
    print(f"prompt_eval_count: {result['prompt_eval_count']}")
    print(f"eval_count: {result['eval_count']}")
    print(f"prompt_tokens_per_sec: {result['prompt_tokens_per_sec']}")
    print(f"generation_tokens_per_sec: {result['generation_tokens_per_sec']}")
    print(f"client_process_peak_rss_mb: {result.get('client_process_peak_rss_mb')}")
    print(f"model_process_rss_before_mb: {result.get('model_process_rss_before_mb')}")
    print(f"model_process_rss_after_mb: {result.get('model_process_rss_after_mb')}")
    print(f"model_process_peak_rss_mb: {result.get('model_process_peak_rss_mb')}")
    print(f"model_process_count_before: {result.get('model_process_count_before')}")
    print(f"model_process_count_after: {result.get('model_process_count_after')}")
    print(f"system_used_memory_before_mb: {result.get('system_used_memory_before_mb')}")
    print(f"system_used_memory_after_mb: {result.get('system_used_memory_after_mb')}")
    print(f"system_used_memory_peak_mb: {result.get('system_used_memory_peak_mb')}")


if __name__ == "__main__":
    main()