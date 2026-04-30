from hf_master_runner import call_hf_master_chat


def main():
    prompt = """Прочитай контекст і дай коротку точну відповідь на запитання.
Відповідай лише самою відповіддю, без пояснень.

Контекст:
Нормани були народом, який у X і XI століттях дав назву Нормандії, регіону у Франції.

Запитання:
У якій країні розташована Нормандія?

Відповідь:"""

    result = call_hf_master_chat(
        prompt=prompt,
        max_new_tokens=16,
    )

    print("\n=== MASTER MODEL OUTPUT ===")
    print(result["text"])

    print("\n=== MASTER METRICS ===")
    print(f"wall_time_sec: {result['wall_time_sec']:.3f}")
    print(f"load_duration_sec: {result['load_duration_sec']:.3f}")
    print(f"prompt_token_count: {result['prompt_token_count']}")
    print(f"generated_token_count: {result['generated_token_count']}")
    print(f"effective_total_tokens_per_sec: {result['effective_total_tokens_per_sec']}")
    print(f"effective_generation_tokens_per_sec: {result['effective_generation_tokens_per_sec']}")
    print(f"rss_mb_after_load: {result['rss_mb_after_load']:.2f}")
    print(f"rss_mb_before_gen: {result['rss_mb_before_gen']:.2f}")
    print(f"rss_mb_after_gen: {result['rss_mb_after_gen']:.2f}")


if __name__ == "__main__":
    main()