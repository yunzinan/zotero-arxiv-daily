"""Tests for LocalReranker — requires sentence-transformers, marked slow."""

import numpy as np
import pytest

from zotero_arxiv_daily.reranker.local import LocalReranker


@pytest.mark.slow
def test_local_reranker(config, monkeypatch):
    class StubTensor:
        def __init__(self, value):
            self._value = value

        def numpy(self):
            return self._value

    class StubSentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, **_kwargs):
            return np.array([[len(text)] for text in texts], dtype=float)

        def similarity(self, left, right):
            return StubTensor(left @ right.T)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", StubSentenceTransformer)
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)
