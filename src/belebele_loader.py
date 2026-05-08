from datasets import Dataset, load_dataset

from config import (
    BELEBELE_DATASET_NAME,
    BELEBELE_UK_LANG,
    BELEBELE_EN_LANG,
)


LETTER_BY_NUM = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
}


def _normalize_belebele_example(example: dict, lang_code: str) -> dict:
    correct_num = int(example["correct_answer_num"])
    correct_letter = LETTER_BY_NUM[correct_num]

    return {
        "id": str(example.get("question_number", "")),
        "lang_code": lang_code,
        "passage": example["flores_passage"],
        "question": example["question"],
        "choice_a": example["mc_answer1"],
        "choice_b": example["mc_answer2"],
        "choice_c": example["mc_answer3"],
        "choice_d": example["mc_answer4"],
        "correct_answer_num": correct_num,
        "correct_letter": correct_letter,
        "source": example.get("link", ""),
        "dialect": example.get("dialect", lang_code),
    }


def load_belebele_subset(lang_code: str, n: int = 20) -> Dataset:
    ds = load_dataset(BELEBELE_DATASET_NAME, lang_code, split="test")

    n = min(n, len(ds))
    ds = ds.select(range(n))

    rows = [_normalize_belebele_example(ex, lang_code) for ex in ds]
    return Dataset.from_list(rows)


def load_belebele_uk_en_subsets(n: int = 20) -> tuple[Dataset, Dataset]:
    uk_ds = load_belebele_subset(BELEBELE_UK_LANG, n)
    en_ds = load_belebele_subset(BELEBELE_EN_LANG, n)

    return uk_ds, en_ds