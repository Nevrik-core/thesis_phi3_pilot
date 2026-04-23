from ollama_runner import call_ollama_chat


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


if __name__ == "__main__":
    main()