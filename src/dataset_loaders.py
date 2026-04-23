import json
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download


def _flatten_squad_like_json(raw_obj: dict) -> list[dict]:
    data = raw_obj["data"] if isinstance(raw_obj, dict) and "data" in raw_obj else raw_obj

    rows = []

    for article in data:
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph["context"]

            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", []) or []

                answer_texts = [a.get("text", "") for a in answers]
                answer_starts = [a.get("answer_start", -1) for a in answers]

                rows.append(
                    {
                        "id": qa.get("id", ""),
                        "title": title,
                        "context": context,
                        "question": qa.get("question", ""),
                        "answers": {
                            "text": answer_texts,
                            "answer_start": answer_starts,
                        },
                    }
                )

    return rows


def load_ua_squad_validation_subset(n: int = 20) -> Dataset:
    """
    Завантажує validation split для UA-SQuAD напряму з HF Hub
    і самостійно перетворює вкладений SQuAD-like JSON у плоский формат.
    """
    file_path = hf_hub_download(
        repo_id="FIdo-AI/ua-squad",
        repo_type="dataset",
        filename="val.json",
    )

    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = _flatten_squad_like_json(raw)
    ds = Dataset.from_list(rows)

    return ds.select(range(min(n, len(ds))))


def load_en_squad_validation_subset(n: int = 20) -> Dataset:
    ds = load_dataset("squad", split="validation")
    return ds.select(range(min(n, len(ds))))