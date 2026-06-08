# Evaluator

LLM-as-a-judge benchmark with two modes: static (default) and active_fetch.

## Modes

### Static mode (default)

The dataset provides pre-written `context` for each question. The evaluator
injects context into the prompt, the model answers, and a judge LLM scores the
response across 5 criteria: faithfulness, relevance, safety, format_tone, and
context_precision.

### Active-fetch mode

The model receives only the question. It must decide what information it needs
by calling `fetch_materials(...)`, a tool backed by a local markdown corpus.
The model iterates through tool calls, gathers materials, then produces a final
answer. No vector DB, no embeddings, no external search -- just character 2-gram
overlap scoring against a flat corpus directory.

## CLI flags

| Flag | Description |
|------|-------------|
| `--input PATH` | (required) Dataset JSONL path |
| `--output PATH` | (required) Output JSON path |
| `--mode {static,active_fetch}` | Evaluation mode (default: static) |
| `--corpus-dir DIR` | Corpus directory for active_fetch (default: eval/corpus) |
| `--dry-run` | Skip LLM tool calls; produce mock results |
| `--model MODEL` | Model name passed to endpoint (default: "default") |
| `--endpoint URL` | LLM server base URL (falls back to env `VLLM_BASE_URL`) |

## Active-fetch dataset format

Each JSONL line must have `context` set to an empty string. The model fetches
its own context. Include `gold_doc_ids` for recall computation.

```json
{
  "question": "...",
  "context": "",
  "expected_answer": "...",
  "gold_doc_ids": ["doc_stem_name"]
}
```

## Corpus directory convention

Place markdown files under `eval/corpus/`. The file stem (name without .md)
becomes the `doc_id` used in `gold_doc_ids` and returned fetch results.

Current corpus files:

- argocd-k3s-sync.md
- refund-cancellation.md
- retention-policy.md
- support-ticket.md
- transaction-retention.md

## How active-fetch works

1. The evaluator sends the question to the model with the `fetch_materials`
   tool available.
2. The model calls `fetch_materials(query, top_k)` to retrieve relevant
   documents from the corpus.
3. The retriever scores every corpus document by character 2-gram overlap
   against the query and returns the top-k passages.
4. Results are appended to the conversation as tool messages.
5. The loop repeats until the model produces a final answer (no tool calls) or
   hits the iteration limit.
6. A judge LLM then scores the final answer against the fetched context.

### Tool schema

```
fetch_materials(query: str, top_k: int)
```

`top_k` defaults to 3 and is clamped to a maximum of 5 in the evaluator.

### FetchResult fields

Each returned document contains:

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | str | Corpus file stem |
| `title` | str | First `#` heading from the markdown |
| `passage` | str | Full document content |
| `score` | float | Character 2-gram overlap score (0-1) |
| `path` | str | Resolved filesystem path |

### Output fields (active mode)

Each sample in the output JSON includes:

| Field | Type | Description |
|-------|------|-------------|
| `mode` | str | Always `"active_fetch"` |
| `tool_calls` | list[object] | Each call: `{name, arguments}` |
| `fetched_doc_ids` | list[str] | Unique doc IDs retrieved |
| `fetch_metrics` | object | See below |
| `loop_terminated_early` | bool | True if MAX_TOOL_ITERATIONS hit |

### Fetch metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fetch_count` | int | Number of `fetch_materials` calls made |
| `must_fetch_compliance` | bool | True if at least one fetch was made |
| `gold_doc_recall` | float or null | Fraction of `gold_doc_ids` that were fetched |

## Commands

Dry-run (validates dataset and output path without LLM calls):

```
python -m eval.evaluator --input eval/eval_data_active_fetch.jsonl --mode active_fetch --dry-run --output /tmp/active_fetch_results.json
```

Real evaluation (requires `--endpoint URL` or `VLLM_BASE_URL`):

```
python -m eval.evaluator --input eval/eval_data_active_fetch.jsonl --mode active_fetch --corpus-dir eval/corpus --output results.json
```

## Design: no vector DB, no embeddings

The retriever uses character 2-gram overlap (`score_2gram_overlap`) to score
query-document similarity. No index to build, no embeddings to compute, no
external service. The corpus is loaded from flat `.md` files on every run.
This is a controlled, reproducible benchmark, not a production RAG system.

## V1 limitations

- Non-streaming only. The model generates its full answer before the judge
  scores it.
- No re-prompt on missing fetch. The evaluator records a compliance failure but
  does not re-prompt.
- No precision or efficiency metrics (planned for V2).
- No streaming tool results. Tool responses are appended as complete messages.
- Judge evaluation against fetched context is wired but uses the static path in
  dry-run mode.
- Maximum of 5 tool iterations (`MAX_TOOL_ITERATIONS = 5`). If the model has
  not produced a final answer by then, the loop terminates and
  `loop_terminated_early` is set to true.

## Observability

The evaluator reports scores to Langfuse. The approach differs by mode.

### Static mode

Uses `push_scores_to_langfuse()` unchanged. Each sample creates a Langfuse
observation and per-criterion numeric scores. The backend records its own
LLM call traces normally. No changes to this flow were made.

### Active-fetch mode (evaluator-owned tracing)

The evaluator process owns one Langfuse trace per sample. The trace hierarchy
looks like this:

```
Trace: rag-active-fetch-sample_{i}
  └─ Span: rag-active-fetch-sample
       ├─ Generation: active_fetch.iteration_1.llm
       ├─ Span:      active_fetch.iteration_1.fetch_materials
       ├─ Generation: active_fetch.iteration_2.llm
       ├─ Span:      active_fetch.iteration_2.fetch_materials
       ├─ ...
       └─ Generation: rag-active-fetch-judge
            └─ per-criterion scores (faithfulness, relevance, safety, etc.)
```

Each LLM call and tool invocation gets its own observation under the parent
sample span. The judge generation carries per-criterion numeric scores.

### Backend suppression

Every evaluator-originated LLM request (model turns, judge call) includes
the header `X-Skip-Langfuse: true` to prevent double-reporting. The backend
reads this header and skips its own Langfuse recording for those requests.
The backend also blocks this header from being forwarded upstream via
`BLOCKED_HEADERS`. Normal application traffic (non-evaluator requests)
records traces as usual.

### Testing and verification

All observability assertions use mocked Langfuse clients, not a live
dashboard. No manual dashboard verification is required to validate the
tracing behavior.
