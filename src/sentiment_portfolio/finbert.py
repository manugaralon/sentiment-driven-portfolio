import functools
import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

PROBS = ["p_pos", "p_neg", "p_neu"]
Scorer = Callable[[list[str]], pd.DataFrame]  # texts -> DataFrame with the PROBS columns, same order


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def label_columns(id2label: dict[int, str]) -> list[str]:
    """Output column of each logit, read from the model config (never hard-coded indices)."""
    column = {"positive": "p_pos", "negative": "p_neg", "neutral": "p_neu"}
    return [column[id2label[i].lower()] for i in range(len(id2label))]


def finbert_scorer(model_name: str, revision: str, max_length: int = 64, batch_size: int = 16) -> Scorer:
    """FinBERT as a function texts -> class probabilities. The model loads on first use."""

    @functools.cache
    def load():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision).eval()
        return tokenizer, model, label_columns(model.config.id2label)

    def score(texts: list[str]) -> pd.DataFrame:
        import torch

        tokenizer, model, columns = load()
        by_length = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # less padding per batch
        probs = [None] * len(texts)
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = by_length[start : start + batch_size]
                encoded = tokenizer(
                    [texts[i] for i in batch],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                for i, row in zip(batch, model(**encoded).logits.softmax(dim=-1).numpy(), strict=True):
                    probs[i] = row
        return pd.DataFrame(probs, columns=columns)[PROBS]

    return score


def score_headlines(
    texts: pd.Series, cache_path: Path, scorer: Scorer, chunk_size: int = 1000
) -> pd.DataFrame:
    """Probabilities and score (P_pos - P_neg) for each unique text, computing only the ones not
    in the cache. The cache is saved after every chunk, so an interrupted run resumes."""
    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
    else:
        cache = pd.DataFrame({"key": pd.Series(dtype=str), **{c: pd.Series(dtype=float) for c in PROBS}})
    unique = pd.Series(texts.unique())
    keys = unique.map(text_key)
    todo = unique[~keys.isin(cache["key"])]
    print(f"{len(unique)} unique texts, {len(unique) - len(todo)} cached, {len(todo)} to score")

    t0 = time.time()
    for start in range(0, len(todo), chunk_size):
        chunk = todo.iloc[start : start + chunk_size]
        new = scorer(chunk.tolist()).assign(key=chunk.map(text_key).to_numpy())
        cache = pd.concat([cache, new], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(cache_path)
        done = start + len(chunk)
        print(f"  {done}/{len(todo)} scored ({done / (time.time() - t0):.1f} texts/s)")

    result = pd.DataFrame({"text": unique, "key": keys}).merge(cache, on="key", how="left")
    result["score"] = result["p_pos"] - result["p_neg"]
    return result
