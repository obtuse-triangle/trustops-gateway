from __future__ import annotations

from pathlib import Path

import pytest

from eval.retriever import (
    FetchResult,
    MarkdownDocument,
    fetch_materials,
    load_markdown_corpus,
    score_2gram_overlap,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "eval" / "corpus"


# ---------------------------------------------------------------------------
# load_markdown_corpus
# ---------------------------------------------------------------------------


def test_load_markdown_corpus_returns_all_five():
    docs = load_markdown_corpus(CORPUS_DIR)
    assert len(docs) == 5
    assert all(isinstance(d, MarkdownDocument) for d in docs)


def test_load_markdown_corpus_doc_ids_match_stems():
    docs = load_markdown_corpus(CORPUS_DIR)
    doc_ids = {d.doc_id for d in docs}
    expected = {
        "argocd-k3s-sync",
        "retention-policy",
        "refund-cancellation",
        "support-ticket",
        "transaction-retention",
    }
    assert doc_ids == expected


def test_load_markdown_corpus_sorted():
    docs = load_markdown_corpus(CORPUS_DIR)
    doc_ids = [d.doc_id for d in docs]
    assert doc_ids == sorted(doc_ids)


def test_load_markdown_corpus_extracts_title():
    docs = load_markdown_corpus(CORPUS_DIR)
    by_id = {d.doc_id: d for d in docs}
    assert "보관 기한 정책 개정 안내" in by_id["retention-policy"].title


def test_load_markdown_corpus_nonexistent_dir():
    docs = load_markdown_corpus(Path("/nonexistent"))
    assert docs == []


# ---------------------------------------------------------------------------
# score_2gram_overlap
# ---------------------------------------------------------------------------


def test_score_identical_strings():
    assert score_2gram_overlap("보관 기한", "보관 기한") == pytest.approx(1.0)


def test_score_empty_query():
    assert score_2gram_overlap("", "아무 내용") == 0.0


def test_score_no_overlap():
    assert score_2gram_overlap("abc", "def") == 0.0


def test_score_partial_overlap():
    score = score_2gram_overlap("보관 기한", "보관 정책")
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# fetch_materials – gold-doc correctness
# ---------------------------------------------------------------------------


def test_fetch_returns_gold_doc():
    """Query '보관 기한' must return retention-policy as top result."""
    results = fetch_materials("보관 기한", top_k=5, corpus_dir=CORPUS_DIR)
    assert len(results) > 0
    assert results[0].doc_id == "retention-policy"
    assert results[0].score > 0.0


# ---------------------------------------------------------------------------
# fetch_materials – determinism
# ---------------------------------------------------------------------------


def test_fetch_deterministic():
    """Ten identical calls return identical doc_id order and passage text."""
    query = "로그인 장애"
    first: list[FetchResult] | None = None
    for _ in range(10):
        results = fetch_materials(query, top_k=5, corpus_dir=CORPUS_DIR)
        if first is None:
            first = results
        else:
            assert len(results) == len(first)
            for r, f in zip(results, first):
                assert r.doc_id == f.doc_id
                assert r.passage == f.passage


# ---------------------------------------------------------------------------
# fetch_materials – top_k
# ---------------------------------------------------------------------------


def test_fetch_limits_top_k():
    results = fetch_materials("보관", top_k=2, corpus_dir=CORPUS_DIR)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# fetch_materials – empty query
# ---------------------------------------------------------------------------


def test_fetch_empty_query_returns_empty():
    results = fetch_materials("", top_k=5, corpus_dir=CORPUS_DIR)
    assert results == []


def test_fetch_whitespace_query_returns_empty():
    results = fetch_materials("   ", top_k=5, corpus_dir=CORPUS_DIR)
    assert results == []


# ---------------------------------------------------------------------------
# fetch_materials – result shape
# ---------------------------------------------------------------------------


def test_fetch_result_dataclass_shape():
    results = fetch_materials("환불", top_k=1, corpus_dir=CORPUS_DIR)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, FetchResult)
    assert isinstance(r.doc_id, str) and r.doc_id
    assert isinstance(r.title, str)
    assert isinstance(r.passage, str) and r.passage
    assert isinstance(r.score, float) and r.score >= 0.0
    assert isinstance(r.path, Path) and r.path.exists()


def test_fetch_all_docs_scored_descending():
    results = fetch_materials("환불 취소 규정", top_k=5, corpus_dir=CORPUS_DIR)
    assert len(results) == 5
    scores = [r.score for r in results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
