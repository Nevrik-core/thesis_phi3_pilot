from datasets import Dataset, load_dataset

from config import (
    UALIGN_DATASET_NAME,
    UALIGN_ETHICS_CONFIG,
    UALIGN_SOCIAL_CHEMISTRY_CONFIG,
)


def _normalize_ethics_example(example: dict, index: int, lang: str) -> dict:
    if lang == "uk":
        text = example["input_ukr"]
    elif lang == "en":
        text = example["input_en"]
    else:
        raise ValueError(f"Unsupported lang: {lang}")

    return {
        "id": f"ethics_{index}",
        "task": "ethics",
        "lang": lang,
        "text": text,
        "label": int(example["label"]),
        "label_space": "0=acceptable,1=unacceptable",
    }


def _normalize_social_chemistry_example(example: dict, index: int, lang: str) -> dict:
    if lang == "uk":
        text = example["action_ukr"]
    elif lang == "en":
        text = example["action_en"]
    else:
        raise ValueError(f"Unsupported lang: {lang}")

    return {
        "id": f"social_chemistry_{index}",
        "task": "social_chemistry",
        "lang": lang,
        "text": text,
        "label": int(example["label"]),
        "label_space": "0=bad,1=expected,2=good",
    }


def load_ualign_ethics_subset(lang: str, n: int = 100) -> Dataset:
    ds = load_dataset(
        UALIGN_DATASET_NAME,
        UALIGN_ETHICS_CONFIG,
        split="test",
    )

    n = min(n, len(ds))
    ds = ds.select(range(n))

    rows = [
        _normalize_ethics_example(example=dict(example), index=i, lang=lang)
        for i, example in enumerate(ds)
    ]

    return Dataset.from_list(rows)


def load_ualign_social_chemistry_subset(lang: str, n: int = 100) -> Dataset:
    ds = load_dataset(
        UALIGN_DATASET_NAME,
        UALIGN_SOCIAL_CHEMISTRY_CONFIG,
        split="test",
    )

    n = min(n, len(ds))
    ds = ds.select(range(n))

    rows = [
        _normalize_social_chemistry_example(example=dict(example), index=i, lang=lang)
        for i, example in enumerate(ds)
    ]

    return Dataset.from_list(rows)


def load_ualign_uk_en_subsets(n: int = 100) -> tuple[Dataset, Dataset, Dataset, Dataset]:
    ethics_uk = load_ualign_ethics_subset(lang="uk", n=n)
    ethics_en = load_ualign_ethics_subset(lang="en", n=n)

    social_uk = load_ualign_social_chemistry_subset(lang="uk", n=n)
    social_en = load_ualign_social_chemistry_subset(lang="en", n=n)

    return ethics_uk, ethics_en, social_uk, social_en