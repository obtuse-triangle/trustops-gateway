from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MarkdownDocument:
    doc_id: str
    title: str
    content: str
    path: Path


@dataclass
class FetchResult:
    doc_id: str
    title: str
    passage: str
    score: float
    path: Path


def _ngrams(text: str, n: int = 2) -> Counter:
    """Character n-gram counter (default: bigrams)."""
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def score_2gram_overlap(query: str, text: str) -> float:
    """Score similarity via character 2-gram overlap (for Korean compatibility)."""
    q = _ngrams(query.lower())
    t = _ngrams(text.lower())
    intersection = sum((q & t).values())
    total = sum(q.values())
    return intersection / total if total > 0 else 0.0


def _extract_title(content: str) -> str:
    """Extract the first # heading as title; fall back to empty string."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def load_markdown_corpus(corpus_dir: Path) -> list[MarkdownDocument]:
    """Load all .md files from *corpus_dir* into MarkdownDocument list.

    File stem (without extension) becomes ``doc_id``.
    Returns documents sorted by ``doc_id`` for deterministic ordering.
    """
    if not corpus_dir.is_dir():
        return []

    docs: list[MarkdownDocument] = []
    for md_path in sorted(corpus_dir.iterdir(), key=lambda p: p.stem):
        if md_path.suffix.lower() != ".md":
            continue
        content = md_path.read_text(encoding="utf-8")
        doc_id = md_path.stem
        title = _extract_title(content)
        docs.append(
            MarkdownDocument(
                doc_id=doc_id,
                title=title,
                content=content,
                path=md_path.resolve(),
            )
        )
    return docs


def fetch_materials(
    query: str,
    top_k: int,
    corpus_dir: Path,
) -> list[FetchResult]:
    """Score corpus documents against *query* and return the top-k results.

    Documents are scored by 2-gram overlap between the query and
    ``title + \"\\n\" + content``.  Results are sorted by score descending
    (ties broken by ``doc_id`` ascending) for deterministic output.
    """
    if not query.strip():
        return []

    docs = load_markdown_corpus(corpus_dir)
    scored: list[tuple[float, str, MarkdownDocument]] = []
    for doc in docs:
        text = f"{doc.title}\n{doc.content}"
        score = score_2gram_overlap(query, text)
        # Use negative score for descending sort, then doc_id for tie-break
        scored.append((-score, doc.doc_id, doc))

    scored.sort()

    results: list[FetchResult] = []
    for _neg_score, _doc_id, doc in scored[:top_k]:
        results.append(
            FetchResult(
                doc_id=doc.doc_id,
                title=doc.title,
                passage=doc.content,
                score=-_neg_score,
                path=doc.path,
            )
        )
    return results
