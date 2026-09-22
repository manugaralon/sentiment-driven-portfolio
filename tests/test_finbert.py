import pandas as pd
import pytest

from sentiment_portfolio.config import load_config
from sentiment_portfolio.finbert import finbert_scorer, label_columns, score_headlines


def fake_scorer(calls: list):
    def score(texts):
        calls.append(list(texts))
        n = len(texts)
        return pd.DataFrame({"p_pos": [0.7] * n, "p_neg": [0.2] * n, "p_neu": [0.1] * n})

    return score


def test_label_columns_follow_the_model_config():
    # The reference repo hard-coded pos=probs[1]; FinBERT's config says 1 is "negative".
    assert label_columns({0: "positive", 1: "negative", 2: "neutral"}) == ["p_pos", "p_neg", "p_neu"]
    assert label_columns({0: "neutral", 1: "positive", 2: "negative"}) == ["p_neu", "p_pos", "p_neg"]


def test_each_unique_text_is_scored_once_and_rerun_hits_the_cache(tmp_path):
    calls = []
    path = tmp_path / "cache.parquet"
    first = score_headlines(pd.Series(["a", "b", "a", "c"]), path, fake_scorer(calls), chunk_size=2)
    assert calls == [["a", "b"], ["c"]]
    assert first["score"].tolist() == pytest.approx([0.5, 0.5, 0.5])

    calls.clear()
    score_headlines(pd.Series(["a", "c", "d"]), path, fake_scorer(calls))
    assert calls == [["d"]]


def test_interrupted_run_resumes_from_the_last_saved_chunk(tmp_path):
    path = tmp_path / "cache.parquet"
    seen = []
    ok = fake_scorer(seen)

    def killed_on_second_chunk(texts):
        if len(seen) == 1:
            raise RuntimeError("killed")
        return ok(texts)

    texts = pd.Series(["a", "b", "c", "d"])
    with pytest.raises(RuntimeError):
        score_headlines(texts, path, killed_on_second_chunk, chunk_size=2)
    calls = []
    score_headlines(texts, path, fake_scorer(calls), chunk_size=2)
    assert calls == [["c", "d"]]


@pytest.mark.slow
def test_real_finbert_gets_obvious_cases_right():
    cfg = load_config()["finbert"]
    probs = finbert_scorer(cfg["model"], cfg["revision"])(
        [
            "Apple beats earnings estimates and raises its full-year guidance",
            "Boeing shares plunge after the company reports a record quarterly loss",
            "Duke Energy will hold its annual shareholder meeting on May 5",
        ]
    )
    score = probs["p_pos"] - probs["p_neg"]
    assert score[0] > 0.5
    assert score[1] < -0.5
    assert probs.loc[2].idxmax() == "p_neu"
