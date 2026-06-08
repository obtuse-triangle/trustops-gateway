from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eval.evaluator import (
    CRITERIA,
    CriterionResult,
    FETCH_MATERIALS_TOOL,
    EvalSample,
    MAX_TOOL_ITERATIONS,
    SampleResult,
    JudgeResponseParseError,
    _build_parser,
    _evaluate_active_fetch_sample_with_tracing,
    build_active_fetch_output_record,
    build_batch_judge_prompt,
    call_llm,
    call_llm_messages,
    check_tool_call_capability,
    compute_fetch_metrics,
    evaluate_sample,
    evaluate_sample_active_fetch,
    load_dataset,
    parse_batch_judge_response,
    run_evaluation,
)
from eval.retriever import FetchResult

EVAL_DATA_PATH = Path(__file__).parent.parent / "eval" / "eval_data.jsonl"
ACTIVE_FETCH_DATA_PATH = Path(__file__).parent.parent / "eval" / "eval_data_active_fetch.jsonl"

_BATCH_PASS_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": '{"faithfulness":true,"relevance":true,"safety":true,"format_tone":true,"context_precision":true}'
            },
            "logprobs": {
                "content": []
            },
        }
    ]
}

_REASONING_ONLY_BATCH_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "",
                "reasoning_content": '<think type="reasoning">analysis</think >{"faithfulness":true,"relevance":false,"safety":true,"format_tone":true,"context_precision":false}',
            },
            "logprobs": {"content": []},
        }
    ]
}

_INVALID_JSON_BATCH_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "",
                "reasoning_content": '{"faithfulness":true,"relevance":true',
            },
            "logprobs": {"content": []},
        }
    ]
}

_MISSING_KEY_BATCH_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": '{"faithfulness":true,"relevance":true,"safety":true,"format_tone":true}',
            },
            "logprobs": {"content": []},
        }
    ]
}

_STRING_BOOL_BATCH_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": '{"faithfulness":"true","relevance":"false","safety":"true","format_tone":"false","context_precision":"true"}',
            },
            "logprobs": {"content": []},
        }
    ]
}

_EXTRA_KEYS_BATCH_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": '{"faithfulness":true,"relevance":false,"safety":true,"format_tone":false,"context_precision":true,"reasoning":"ignored"}',
            },
            "logprobs": {"content": []},
        }
    ]
}


def test_load_dataset():
    samples = load_dataset(str(EVAL_DATA_PATH))
    assert len(samples) == 5
    for sample in samples:
        assert isinstance(sample, EvalSample)
        assert sample.question
        assert sample.context
        assert sample.expected_answer


def test_load_dataset_field_mapping():
    samples = load_dataset(str(EVAL_DATA_PATH))
    first = samples[0]
    assert "K3s" in first.question or "ArgoCD" in first.question
    assert len(first.context) > 10
    assert len(first.expected_answer) > 10


def test_load_dataset_all_korean():
    samples = load_dataset(str(EVAL_DATA_PATH))
    for sample in samples:
        assert any(ord(c) > 0x3000 for c in sample.question), (
            f"Expected Korean characters in question: {sample.question!r}"
        )


def test_dataset_answer_token_overlap_is_low():
    samples = load_dataset(str(EVAL_DATA_PATH))
    for sample in samples:
        context_tokens = set(re.findall(r"[A-Za-z가-힣0-9_-]+", sample.context.lower()))
        answer_tokens = set(re.findall(r"[A-Za-z가-힣0-9_-]+", sample.expected_answer.lower()))
        overlap_ratio = len(context_tokens & answer_tokens) / len(answer_tokens)
        assert overlap_ratio < 0.30


def test_dataset_context_nontrivial_length():
    samples = load_dataset(str(EVAL_DATA_PATH))
    for sample in samples:
        assert len(sample.context) >= 50
        assert len(sample.expected_answer) >= 20


def test_dataset_question_asks_something():
    samples = load_dataset(str(EVAL_DATA_PATH))
    for sample in samples:
        assert "?" in sample.question or sample.question.endswith("요?")


def test_load_dataset_backward_compat():
    samples = load_dataset(str(EVAL_DATA_PATH))
    for sample in samples:
        assert isinstance(sample, EvalSample)
        assert sample.gold_doc_ids == []


def test_load_dataset_active_fetch():
    samples = load_dataset(str(ACTIVE_FETCH_DATA_PATH))
    assert len(samples) == 5
    for sample in samples:
        assert isinstance(sample, EvalSample)
        assert sample.context == ""
        assert len(sample.gold_doc_ids) > 0
        assert isinstance(sample.gold_doc_ids[0], str)


async def test_run_evaluation_writes_commit_and_workflow_metadata(tmp_path: Path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"question":"What is K3s?","context":"K3s is a lightweight Kubernetes distribution.","expected_answer":"K3s is lightweight Kubernetes."}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "results.json"

    await run_evaluation(
        str(dataset_path),
        llm_endpoint="http://localhost:8080",
        output_path=str(output_path),
        model="test-model",
        commit_sha="abc123",
        workflow_uid="wf-789",
        dry_run=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    record = payload[0]
    assert record["commit_sha"] == "abc123"
    assert record["workflow_uid"] == "wf-789"
    assert record["dataset_path"] == str(dataset_path)
    assert record["model_id"] == "test-model"
    assert record["scores"] == {criterion: 1.0 for criterion in CRITERIA}
    assert record["confidence"] == {criterion: 1.0 for criterion in CRITERIA}
    assert record["overall_score"] == 1.0
    assert record["passed"] is True


async def test_run_evaluation_malformed_input_fails_without_artifact(tmp_path: Path):
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text("{not json}\n", encoding="utf-8")
    output_path = tmp_path / "results.json"

    with pytest.raises(Exception):
        await run_evaluation(
            str(dataset_path),
            llm_endpoint="http://localhost:8080",
            output_path=str(output_path),
            dry_run=True,
        )

    assert not output_path.exists()


def test_batch_judge_prompt_contains_all_criteria():
    prompt = build_batch_judge_prompt(
        question="test question",
        context="test context",
        expected_answer="test answer",
        response="test response",
    )
    for criterion in CRITERIA:
        assert criterion in prompt


def test_batch_judge_prompt_contains_independence_instruction():
    prompt = build_batch_judge_prompt(
        question="What is K3s?",
        context="K3s is a lightweight Kubernetes distribution.",
        expected_answer="K3s is lightweight Kubernetes.",
        response="K3s is Kubernetes.",
    )
    assert "independently" in prompt or "independent" in prompt
    assert "What is K3s?" in prompt


def test_batch_judge_prompt_contains_criterion_rubric():
    prompt = build_batch_judge_prompt(
        question="What is K3s?",
        context="K3s is a lightweight Kubernetes distribution.",
        expected_answer="K3s is lightweight Kubernetes.",
        response="K3s is Kubernetes.",
    )
    assert "PASS only if" in prompt
    assert "FAIL if" in prompt


def test_batch_judge_prompt_faithfulness_rubric():
    prompt = build_batch_judge_prompt(
        question="What is K3s?",
        context="K3s is a lightweight Kubernetes distribution.",
        expected_answer="K3s is lightweight Kubernetes.",
        response="K3s is Kubernetes.",
    )
    assert (
        "hallucinat" in prompt.lower()
        or "unsupported" in prompt.lower()
        or "context에 없는" in prompt
    )


def test_batch_judge_prompt_safety_rubric():
    prompt = build_batch_judge_prompt(
        question="What is K3s?",
        context="K3s is a lightweight Kubernetes distribution.",
        expected_answer="K3s is lightweight Kubernetes.",
        response="K3s is Kubernetes.",
    )
    assert (
        "harmful" in prompt.lower()
        or "offensive" in prompt.lower()
        or "privacy" in prompt.lower()
        or "over-refus" in prompt.lower()
    )


def test_batch_judge_parse_valid_response():
    parsed = parse_batch_judge_response(_BATCH_PASS_RESPONSE)
    assert parsed == {criterion: True for criterion in CRITERIA}


def test_batch_judge_parse_missing_key_raises():
    with pytest.raises(JudgeResponseParseError):
        parse_batch_judge_response(_MISSING_KEY_BATCH_RESPONSE)


def test_batch_judge_parse_truncated_json_raises():
    with pytest.raises(JudgeResponseParseError):
        parse_batch_judge_response(_INVALID_JSON_BATCH_RESPONSE)


def test_batch_judge_parse_think_tags_stripped():
    parsed = parse_batch_judge_response(_REASONING_ONLY_BATCH_RESPONSE)
    assert parsed["faithfulness"] is True
    assert parsed["relevance"] is False
    assert parsed["context_precision"] is False


def test_batch_judge_parse_string_booleans_coerced():
    parsed = parse_batch_judge_response(_STRING_BOOL_BATCH_RESPONSE)
    assert parsed == {
        "faithfulness": True,
        "relevance": False,
        "safety": True,
        "format_tone": False,
        "context_precision": True,
    }


def test_batch_judge_parse_extra_keys_ignored():
    parsed = parse_batch_judge_response(_EXTRA_KEYS_BATCH_RESPONSE)
    assert parsed == {
        "faithfulness": True,
        "relevance": False,
        "safety": True,
        "format_tone": False,
        "context_precision": True,
    }


async def test_evaluate_sample_returns_result():
    sample = EvalSample(
        question="What is K3s?",
        context="K3s is a lightweight Kubernetes distribution.",
        expected_answer="K3s is lightweight Kubernetes.",
    )
    call_count = 0

    async def mock_call(prompt, *, endpoint, **kwargs):
        nonlocal call_count
        call_count += 1
        return _BATCH_PASS_RESPONSE

    semaphore = asyncio.Semaphore(1)
    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        result = await evaluate_sample(
            sample, 0, llm_endpoint="http://localhost:8080", semaphore=semaphore
        )

    assert isinstance(result, SampleResult)
    assert result.sample_id == "sample_0"
    assert [c.criterion for c in result.criteria] == CRITERIA
    assert call_count == 2


async def test_unexpected_token():
    sample = EvalSample(
        question="test",
        context="test ctx",
        expected_answer="test ans",
    )

    async def mock_call(prompt, *, endpoint, **kwargs):
        return _INVALID_JSON_BATCH_RESPONSE

    semaphore = asyncio.Semaphore(1)
    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        with pytest.raises(JudgeResponseParseError):
            await evaluate_sample(
                sample, 0, llm_endpoint="http://localhost:8080", semaphore=semaphore
            )


async def test_rate_limiting():
    active_count = [0]
    max_concurrent = [0]

    async def mock_call(prompt, *, endpoint, **kwargs):
        active_count[0] += 1
        max_concurrent[0] = max(max_concurrent[0], active_count[0])
        await asyncio.sleep(0.02)
        active_count[0] -= 1
        return _BATCH_PASS_RESPONSE

    semaphore = asyncio.Semaphore(1)
    sample = EvalSample(question="test", context="ctx", expected_answer="ans")

    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        tasks = [
            evaluate_sample(
                sample, i, llm_endpoint="http://localhost:8080", semaphore=semaphore
            )
            for i in range(2)
        ]
        await asyncio.gather(*tasks)

    assert max_concurrent[0] == 1


async def test_evaluate_sample_index_preserved():
    sample = EvalSample(question="q", context="c", expected_answer="e")

    async def mock_call(prompt, *, endpoint, **kwargs):
        return _BATCH_PASS_RESPONSE

    semaphore = asyncio.Semaphore(1)
    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        result = await evaluate_sample(
            sample, 42, llm_endpoint="http://localhost:8080", semaphore=semaphore
        )

    assert result.sample_id == "sample_42"


async def test_evaluate_sample_all_criteria_scored():
    sample = EvalSample(question="q", context="c", expected_answer="e")

    async def mock_call(prompt, *, endpoint, **kwargs):
        return _BATCH_PASS_RESPONSE

    semaphore = asyncio.Semaphore(1)
    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        result = await evaluate_sample(
            sample, 0, llm_endpoint="http://localhost:8080", semaphore=semaphore
        )

    for criterion in CRITERIA:
        item = next(c for c in result.criteria if c.criterion == criterion)
        assert item.criterion == criterion
        assert item.passed is True
        assert item.confidence == 1.0
        assert item.score == 1.0


async def test_evaluate_sample_batch_uses_reasoning_only_response():
    sample = EvalSample(question="q", context="c", expected_answer="e")
    call_count = 0

    async def mock_call(prompt, *, endpoint, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _BATCH_PASS_RESPONSE
        return _REASONING_ONLY_BATCH_RESPONSE

    semaphore = asyncio.Semaphore(1)
    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        result = await evaluate_sample(
            sample, 0, llm_endpoint="http://localhost:8080", semaphore=semaphore
        )

    assert call_count == 2
    assert [c.passed for c in result.criteria] == [True, False, True, True, False]


async def test_evaluate_sample_batch_makes_two_calls():
    sample = EvalSample(question="q", context="c", expected_answer="e")
    call_count = 0

    async def mock_call(prompt, *, endpoint, **kwargs):
        nonlocal call_count
        call_count += 1
        return _BATCH_PASS_RESPONSE

    semaphore = asyncio.Semaphore(1)
    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        await evaluate_sample(sample, 0, llm_endpoint="http://localhost:8080", semaphore=semaphore)

    assert call_count == 2


async def test_evaluate_sample_batch_retry_on_parse_failure():
    sample = EvalSample(question="q", context="c", expected_answer="e")
    call_count = 0

    async def mock_call(prompt, *, endpoint, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _BATCH_PASS_RESPONSE
        if call_count == 2:
            return _INVALID_JSON_BATCH_RESPONSE
        return _BATCH_PASS_RESPONSE

    semaphore = asyncio.Semaphore(1)
    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        result = await evaluate_sample(sample, 0, llm_endpoint="http://localhost:8080", semaphore=semaphore)

    assert call_count == 3
    assert all(c.passed for c in result.criteria)


async def test_evaluate_sample_batch_hard_error_after_retry():
    sample = EvalSample(question="q", context="c", expected_answer="e")
    call_count = 0

    async def mock_call(prompt, *, endpoint, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _BATCH_PASS_RESPONSE
        return _INVALID_JSON_BATCH_RESPONSE

    semaphore = asyncio.Semaphore(1)
    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        with pytest.raises(JudgeResponseParseError):
            await evaluate_sample(sample, 0, llm_endpoint="http://localhost:8080", semaphore=semaphore)

    assert call_count == 3


async def test_call_llm_signature_unchanged():
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"choices": [{"message": {"content": "ok"}}]})

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("eval.evaluator.httpx.AsyncClient", return_value=mock_client):
        result = await call_llm("test prompt", endpoint="http://test", model="m")

    call_kwargs = mock_client.post.call_args[1]
    payload = call_kwargs["json"]
    assert payload["messages"] == [{"role": "user", "content": "test prompt"}]
    assert "tools" not in payload
    assert result == {"choices": [{"message": {"content": "ok"}}]}


async def test_call_llm_messages_sends_tools():
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"choices": [{"message": {"content": "ok"}}]})

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)

    messages = [{"role": "user", "content": "test"}]
    with patch("eval.evaluator.httpx.AsyncClient", return_value=mock_client):
        result = await call_llm_messages(
            messages,
            endpoint="http://test",
            model="m",
            tools=[FETCH_MATERIALS_TOOL],
            tool_choice="auto",
        )

    call_kwargs = mock_client.post.call_args[1]
    payload = call_kwargs["json"]
    assert "tools" in payload
    assert payload["tools"] == [FETCH_MATERIALS_TOOL]
    assert "tool_choice" in payload
    assert payload["tool_choice"] == "auto"
    assert payload["messages"] == [{"role": "user", "content": "test"}]
    assert result == {"choices": [{"message": {"content": "ok"}}]}


async def test_call_llm_passes_extra_headers():
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"choices": [{"message": {"content": "ok"}}]})

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("eval.evaluator.httpx.AsyncClient", return_value=mock_client):
        result = await call_llm(
            "test prompt",
            endpoint="http://test",
            model="m",
            extra_headers={"X-Skip-Langfuse": "true"},
        )

    call_kwargs = mock_client.post.call_args[1]
    assert "headers" in call_kwargs
    assert call_kwargs["headers"]["X-Skip-Langfuse"] == "true"
    assert result == {"choices": [{"message": {"content": "ok"}}]}


async def test_call_llm_without_extra_headers_unchanged():
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"choices": [{"message": {"content": "ok"}}]})

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("eval.evaluator.httpx.AsyncClient", return_value=mock_client):
        result = await call_llm("test prompt", endpoint="http://test", model="m")

    call_kwargs = mock_client.post.call_args[1]
    assert "headers" not in call_kwargs
    assert result == {"choices": [{"message": {"content": "ok"}}]}


async def test_call_llm_messages_passes_extra_headers():
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"choices": [{"message": {"content": "ok"}}]})

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)

    messages = [{"role": "user", "content": "test"}]
    with patch("eval.evaluator.httpx.AsyncClient", return_value=mock_client):
        result = await call_llm_messages(
            messages,
            endpoint="http://test",
            model="m",
            extra_headers={"X-Skip-Langfuse": "true"},
        )

    call_kwargs = mock_client.post.call_args[1]
    assert "headers" in call_kwargs
    assert call_kwargs["headers"]["X-Skip-Langfuse"] == "true"
    assert result == {"choices": [{"message": {"content": "ok"}}]}


async def test_call_llm_messages_without_extra_headers_unchanged():
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"choices": [{"message": {"content": "ok"}}]})

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)

    messages = [{"role": "user", "content": "test"}]
    with patch("eval.evaluator.httpx.AsyncClient", return_value=mock_client):
        result = await call_llm_messages(
            messages,
            endpoint="http://test",
            model="m",
        )

    call_kwargs = mock_client.post.call_args[1]
    assert "headers" not in call_kwargs
    assert result == {"choices": [{"message": {"content": "ok"}}]}


async def test_tool_call_capability_failure_message():
    mock_response = {"choices": [{"message": {"content": "I can help"}}]}

    with patch("eval.evaluator.call_llm_messages", AsyncMock(return_value=mock_response)):
        with pytest.raises(RuntimeError) as exc_info:
            await check_tool_call_capability(endpoint="http://test")

    err = str(exc_info.value)
    assert "tool calling" in err.lower()
    assert "active_fetch" in err


async def test_active_fetch_loop_terminates():
    sample = EvalSample(question="test q", context="", expected_answer="test a")

    tool_calls_response = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "tool_calls": [{
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "fetch_materials",
                        "arguments": '{"query": "test", "top_k": 3}',
                    },
                }],
            },
        }]
    }

    stop_response = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "final answer"},
        }]
    }

    mock_fetch_result = FetchResult(
        doc_id="doc1", title="t", passage="p", score=1.0, path=Path("/tmp/d"),
    )

    with patch("eval.evaluator.call_llm_messages", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [tool_calls_response, stop_response]
        with patch("eval.evaluator.fetch_materials", return_value=[mock_fetch_result]):
            result = await evaluate_sample_active_fetch(sample, 0, llm_endpoint="http://test")

    assert mock_llm.call_count == 2
    assert result["llm_answer"] == "final answer"
    assert result["fetch_count"] == 1
    assert result["fetched_doc_ids"] == ["doc1"]
    assert result["loop_terminated_early"] is False


async def test_active_fetch_max_iterations():
    sample = EvalSample(question="q", context="", expected_answer="e")

    tool_calls_response = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "tool_calls": [{
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "fetch_materials",
                        "arguments": '{"query": "test", "top_k": 3}',
                    },
                }],
            },
        }]
    }

    mock_fetch = MagicMock(return_value=[
        FetchResult(doc_id="doc1", title="t", passage="p", score=1.0, path=Path("/tmp/d")),
    ])

    with patch("eval.evaluator.call_llm_messages", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [tool_calls_response] * MAX_TOOL_ITERATIONS
        with patch("eval.evaluator.fetch_materials", mock_fetch):
            result = await evaluate_sample_active_fetch(sample, 0, llm_endpoint="http://test")

    assert result["loop_terminated_early"] is True
    assert result["fetch_count"] == MAX_TOOL_ITERATIONS
    assert mock_llm.call_count == MAX_TOOL_ITERATIONS


async def test_active_fetch_rejects_unknown_tool():
    sample = EvalSample(question="q", context="", expected_answer="e")

    evil_tool_response = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "tool_calls": [{
                    "id": "evil1",
                    "type": "function",
                    "function": {
                        "name": "evil_tool",
                        "arguments": '{"malicious": true}',
                    },
                }],
            },
        }]
    }

    stop_response = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "safe answer"},
        }]
    }

    with patch("eval.evaluator.call_llm_messages", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [evil_tool_response, stop_response]
        with patch("eval.evaluator.fetch_materials") as mock_fetch:
            result = await evaluate_sample_active_fetch(sample, 0, llm_endpoint="http://test")

    assert result["fetch_count"] == 0
    assert mock_fetch.call_count == 0
    assert result["llm_answer"] == "safe answer"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "evil_tool"


def test_active_fetch_no_fetch_compliance():
    metrics = compute_fetch_metrics(
        fetch_count=0, fetched_doc_ids=[], gold_doc_ids=["doc_a"]
    )
    assert metrics["must_fetch_compliance"] is False
    assert metrics["gold_doc_recall"] == 0.0
    assert metrics["fetch_count"] == 0


def test_active_fetch_gold_doc_recall():
    metrics = compute_fetch_metrics(
        fetch_count=2,
        fetched_doc_ids=["doc_a", "doc_b"],
        gold_doc_ids=["doc_a", "doc_c"],
    )
    assert metrics["gold_doc_recall"] == 0.5
    assert metrics["must_fetch_compliance"] is True
    assert metrics["fetch_count"] == 2


def test_active_fetch_gold_doc_recall_no_gold():
    metrics = compute_fetch_metrics(
        fetch_count=1, fetched_doc_ids=["doc_a"], gold_doc_ids=[]
    )
    assert metrics["gold_doc_recall"] is None
    assert metrics["must_fetch_compliance"] is True
    assert metrics["fetch_count"] == 1


def test_active_fetch_output_schema():
    af_result = {
        "sample_id": "sample_0",
        "question": "test?",
        "expected_answer": "test answer",
        "llm_answer": "final answer",
        "tool_calls": [{"name": "fetch_materials", "arguments": '{"query": "test"}'}],
        "fetched_doc_ids": ["doc_a"],
        "fetched_text": "some content",
        "fetch_count": 1,
        "loop_terminated_early": False,
    }
    record = build_active_fetch_output_record(
        af_result,
        gold_doc_ids=["doc_a"],
        commit_sha="abc123",
        workflow_uid="wf-001",
        dataset_path="test_dataset",
        model="test-model",
    )
    assert record["mode"] == "active_fetch"
    assert record["fetched_doc_ids"] == ["doc_a"]
    assert record["tool_calls"] == af_result["tool_calls"]
    assert record["loop_terminated_early"] is False
    assert record["llm_answer"] == "final answer"
    assert record["sample_id"] == "sample_0"
    assert record["commit_sha"] == "abc123"
    assert record["workflow_uid"] == "wf-001"
    assert record["dataset_path"] == "test_dataset"
    assert record["model_id"] == "test-model"

    metrics = record["fetch_metrics"]
    assert "fetch_count" in metrics
    assert "must_fetch_compliance" in metrics
    assert "gold_doc_recall" in metrics
    assert metrics["fetch_count"] == 1
    assert metrics["must_fetch_compliance"] is True
    assert metrics["gold_doc_recall"] == 1.0


def test_static_output_schema_unchanged():
    sample_result = SampleResult(
        sample_id="sample_0",
        question="test?",
        context="test context",
        expected_answer="test answer",
        llm_answer="final answer",
        criteria=[
            CriterionResult(criterion="faithfulness", passed=True, confidence=1.0, score=1.0),
            CriterionResult(criterion="relevance", passed=True, confidence=1.0, score=1.0),
            CriterionResult(criterion="safety", passed=True, confidence=1.0, score=1.0),
            CriterionResult(criterion="format_tone", passed=False, confidence=0.5, score=0.0),
            CriterionResult(criterion="context_precision", passed=True, confidence=1.0, score=1.0),
        ],
    )
    scores = {c.criterion: c.score for c in sample_result.criteria}
    confidence = {c.criterion: c.confidence for c in sample_result.criteria}
    overall_score = sum(scores.values()) / len(scores) if scores else 0.0
    record = {
        "sample_id": sample_result.sample_id,
        "commit_sha": None,
        "workflow_uid": None,
        "dataset_path": "test",
        "model_id": "test",
        "scores": scores,
        "confidence": confidence,
        "overall_score": overall_score,
        "passed": all(c.passed for c in sample_result.criteria),
        "llm_answer": sample_result.llm_answer,
    }

    assert "mode" not in record
    assert "tool_calls" not in record
    assert "fetched_doc_ids" not in record
    assert "fetch_metrics" not in record
    assert "loop_terminated_early" not in record


def test_mode_defaults_to_static():
    parser = _build_parser()
    args = parser.parse_args(["--input", "test.jsonl", "--output", "test.json"])
    assert args.mode == "static"
    assert args.corpus_dir == "eval/corpus"


async def test_active_fetch_runs_capability_check(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"question":"What is K3s?","context":"","expected_answer":"lightweight k8s"}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "results.json"

    tool_calls_response = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "tool_calls": [{
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "fetch_materials",
                        "arguments": '{"query": "test", "top_k": 3}',
                    },
                }],
            },
        }]
    }
    stop_response = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "final answer"},
        }]
    }

    with patch("eval.evaluator.check_tool_call_capability", new_callable=AsyncMock) as mock_check:
        with patch("eval.evaluator.call_llm_messages", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [tool_calls_response, stop_response]
            with patch("eval.evaluator.fetch_materials", return_value=[]):
                with patch("eval.evaluator.call_llm", new_callable=AsyncMock) as mock_call_llm:
                    mock_call_llm.return_value = _BATCH_PASS_RESPONSE
                    await run_evaluation(
                        str(dataset_path),
                        llm_endpoint="http://localhost:8080",
                        output_path=str(output_path),
                        model="test-model",
                        dry_run=False,
                        mode="active_fetch",
                        corpus_dir=tmp_path,
                    )

    mock_check.assert_awaited_once()


async def test_run_evaluation_static_regression(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"question":"What is K3s?","context":"K3s is a lightweight Kubernetes distribution.","expected_answer":"K3s is lightweight Kubernetes."}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "results.json"

    async def mock_call(prompt: str, *, endpoint: str, **kwargs: object) -> dict:
        return _BATCH_PASS_RESPONSE

    with patch("eval.evaluator.call_llm", side_effect=mock_call) as mock_cl:
        with patch("eval.evaluator.evaluate_sample_active_fetch") as mock_active:
            await run_evaluation(
                str(dataset_path),
                llm_endpoint="http://localhost:8080",
                output_path=str(output_path),
                model="test-model",
                dry_run=False,
            )

    mock_active.assert_not_called()
    assert mock_cl.call_count == 2

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    record = payload[0]
    assert "scores" in record
    assert "confidence" in record
    assert "overall_score" in record
    assert "passed" in record
    assert "llm_answer" in record
    assert "mode" not in record
    assert "fetched_doc_ids" not in record
    assert "fetch_metrics" not in record
    assert "tool_calls" not in record
    assert "loop_terminated_early" not in record


async def test_run_evaluation_active_fetch_e2e(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"question":"What is K3s?","context":"","expected_answer":"lightweight K3s","gold_doc_ids":["doc1"]}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "results.json"

    tool_calls_response = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "tool_calls": [{
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "fetch_materials",
                        "arguments": '{"query": "test", "top_k": 3}',
                    },
                }],
            },
        }]
    }

    stop_response = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "final answer"},
        }]
    }

    mock_fetch_result = FetchResult(
        doc_id="doc1", title="t", passage="p", score=1.0, path=Path("/tmp/d"),
    )

    with patch("eval.evaluator.call_llm_messages", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [
            tool_calls_response,  # check_tool_call_capability
            tool_calls_response,  # evaluate_sample_active_fetch iteration 0 (tool call)
            stop_response,        # evaluate_sample_active_fetch iteration 1 (stop)
        ]
        with patch("eval.evaluator.fetch_materials", return_value=[mock_fetch_result]):
            with patch("eval.evaluator.call_llm", new_callable=AsyncMock) as mock_call_llm:
                mock_call_llm.return_value = _BATCH_PASS_RESPONSE
                await run_evaluation(
                    str(dataset_path),
                    llm_endpoint="http://localhost:8080",
                    output_path=str(output_path),
                    model="test-model",
                    dry_run=False,
                    mode="active_fetch",
                    corpus_dir=str(tmp_path),
                )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    record = payload[0]

    assert record["mode"] == "active_fetch"
    assert record["fetched_doc_ids"] == ["doc1"]
    assert record["tool_calls"] == [
        {"name": "fetch_materials", "arguments": '{"query": "test", "top_k": 3}'},
    ]
    assert record["loop_terminated_early"] is False
    assert record["llm_answer"] == "final answer"

    metrics = record["fetch_metrics"]
    assert metrics["fetch_count"] == 1
    assert metrics["must_fetch_compliance"] is True
    assert metrics["gold_doc_recall"] == 1.0

    # Verify judge criteria are now included in active_fetch output
    assert "scores" in record
    assert record["scores"] == {criterion: 1.0 for criterion in CRITERIA}
    assert "confidence" in record
    assert record["confidence"] == {criterion: 1.0 for criterion in CRITERIA}
    assert "overall_score" in record
    assert record["overall_score"] == 1.0
    assert "passed" in record
    assert record["passed"] is True

    assert record["model_id"] == "test-model"
    assert record["dataset_path"] == str(dataset_path)


async def test_active_fetch_creates_langfuse_trace():
    sample = EvalSample(question="test q", context="", expected_answer="test a")

    tool_calls_response = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "tool_calls": [{
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "fetch_materials",
                        "arguments": '{"query": "test", "top_k": 3}',
                    },
                }],
            },
        }]
    }
    stop_response = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "final answer"},
        }]
    }

    mock_fetch_result = FetchResult(
        doc_id="doc1", title="t", passage="p", score=1.0, path=Path("/tmp/d"),
    )

    langfuse_client = MagicMock()
    obs_mock = MagicMock()
    obs_mock.id = "obs-001"
    langfuse_client.start_observation.return_value = obs_mock

    with patch("eval.evaluator.call_llm_messages", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [tool_calls_response, stop_response]
        with patch("eval.evaluator.fetch_materials", return_value=[mock_fetch_result]):
            result = await evaluate_sample_active_fetch(
                sample, 0,
                llm_endpoint="http://test",
                langfuse_client=langfuse_client,
                active_fetch_trace_name="test-trace",
                active_fetch_parent_observation_id="parent-obs-1",
            )

    assert result["llm_answer"] == "final answer"
    assert result["fetch_count"] == 1

    # start_observation: iteration_1 LLM + iteration_1 fetch + iteration_2 LLM (stop)
    assert langfuse_client.start_observation.call_count == 3

    # Inspect each call
    calls = langfuse_client.start_observation.call_args_list

    # First call: iteration_1 LLM generation
    assert calls[0][1]["name"] == "active_fetch.iteration_1.llm"
    assert calls[0][1]["as_type"] == "generation"
    # input captures messages reference (mutated in later loop iterations)
    assert calls[0][1]["input"]["messages"][0] == {"role": "user", "content": "test q"}
    assert calls[0][1]["metadata"]["parent_observation_id"] == "parent-obs-1"
    assert calls[0][1]["metadata"]["active_fetch_trace_name"] == "test-trace"
    assert calls[0][1]["metadata"]["iteration"] == 1

    # Second call: iteration_1 fetch_materials span
    assert calls[1][1]["name"] == "active_fetch.iteration_1.fetch_materials"
    assert calls[1][1]["as_type"] == "span"
    assert calls[1][1]["input"]["query"] == "test"
    assert calls[1][1]["input"]["top_k"] == 3
    assert calls[1][1]["input"]["tool_call_id"] == "tc1"
    assert calls[1][1]["metadata"]["parent_observation_id"] == "parent-obs-1"
    assert calls[1][1]["metadata"]["active_fetch_trace_name"] == "test-trace"
    assert calls[1][1]["metadata"]["iteration"] == 1

    # Third call: iteration_2 LLM generation (stop response - still traced)
    assert calls[2][1]["name"] == "active_fetch.iteration_2.llm"
    assert calls[2][1]["as_type"] == "generation"
    assert calls[2][1]["metadata"]["iteration"] == 2


async def test_active_fetch_langfuse_none_no_crash():
    sample = EvalSample(question="test q", context="", expected_answer="test a")

    tool_calls_response = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "tool_calls": [{
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "fetch_materials",
                        "arguments": '{"query": "test", "top_k": 3}',
                    },
                }],
            },
        }]
    }
    stop_response = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "final answer"},
        }]
    }

    mock_fetch_result = FetchResult(
        doc_id="doc1", title="t", passage="p", score=1.0, path=Path("/tmp/d"),
    )

    with patch("eval.evaluator.call_llm_messages", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [tool_calls_response, stop_response]
        with patch("eval.evaluator.fetch_materials", return_value=[mock_fetch_result]):
            result = await evaluate_sample_active_fetch(
                sample, 0,
                llm_endpoint="http://test",
                langfuse_client=None,
                suppress_backend_langfuse=True,
            )

    assert result["llm_answer"] == "final answer"
    assert result["fetch_count"] == 1
    assert result["fetched_doc_ids"] == ["doc1"]
    assert result["loop_terminated_early"] is False

    # Verify skip header was passed in both calls
    assert mock_llm.call_count == 2
    for call_args in mock_llm.call_args_list:
        assert call_args[1]["extra_headers"] == {"X-Skip-Langfuse": "true"}


async def test_active_fetch_no_tool_call_still_traced():
    sample = EvalSample(question="direct answer q", context="", expected_answer="direct a")

    stop_response = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "direct answer"},
        }]
    }

    langfuse_client = MagicMock()
    obs_mock = MagicMock()
    obs_mock.id = "obs-001"
    langfuse_client.start_observation.return_value = obs_mock

    with patch("eval.evaluator.call_llm_messages", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [stop_response]
        with patch("eval.evaluator.fetch_materials") as mock_fetch:
            result = await evaluate_sample_active_fetch(
                sample, 0,
                llm_endpoint="http://test",
                langfuse_client=langfuse_client,
                active_fetch_trace_name="test-trace",
                active_fetch_parent_observation_id="parent-obs-1",
            )

    assert result["llm_answer"] == "direct answer"
    assert result["fetch_count"] == 0
    mock_fetch.assert_not_called()

    # At least one LLM observation was created
    assert langfuse_client.start_observation.call_count == 1
    calls = langfuse_client.start_observation.call_args_list
    assert calls[0][1]["name"] == "active_fetch.iteration_1.llm"
    assert calls[0][1]["as_type"] == "generation"

    # No fetch observation since there were no tool calls
    obs_mock.update.assert_called_once()
    obs_mock.end.assert_called_once()


# ---------------------------------------------------------------------------
# Active-fetch tracing helper tests
# ---------------------------------------------------------------------------


async def test_active_fetch_judge_uses_skip_header():
    """Verify that judge call_llm receives X-Skip-Langfuse header."""
    sample = EvalSample(question="test q", context="", expected_answer="test a")
    mock_af_result = {
        "sample_id": "sample_0",
        "question": "test q",
        "expected_answer": "test a",
        "llm_answer": "final answer",
        "tool_calls": [],
        "fetched_doc_ids": [],
        "fetched_text": "some context",
        "fetch_count": 0,
        "loop_terminated_early": False,
    }

    langfuse_client = MagicMock()
    parent_obs = MagicMock()
    parent_obs.id = "parent-obs-001"
    langfuse_client.start_observation.return_value = parent_obs

    with patch("eval.evaluator._propagate_attributes", return_value=MagicMock()):
        with patch(
            "eval.evaluator.evaluate_sample_active_fetch", new_callable=AsyncMock
        ) as mock_active:
            mock_active.return_value = mock_af_result
            with patch(
                "eval.evaluator.call_llm", new_callable=AsyncMock
            ) as mock_call_llm:
                mock_call_llm.return_value = _BATCH_PASS_RESPONSE

                await _evaluate_active_fetch_sample_with_tracing(
                    sample,
                    0,
                    llm_endpoint="http://test",
                    evaluation_langfuse_client=langfuse_client,
                )

    assert mock_call_llm.call_count == 1
    call_kwargs = mock_call_llm.call_args[1]
    assert call_kwargs.get("extra_headers") == {"X-Skip-Langfuse": "true"}


async def test_active_fetch_records_judge_observation():
    """Verify that a judge observation is created under the same trace context."""
    sample = EvalSample(question="test q", context="", expected_answer="test a")
    mock_af_result = {
        "sample_id": "sample_0",
        "question": "test q",
        "expected_answer": "test a",
        "llm_answer": "final answer",
        "tool_calls": [],
        "fetched_doc_ids": [],
        "fetched_text": "some context",
        "fetch_count": 0,
        "loop_terminated_early": False,
    }

    langfuse_client = MagicMock()
    parent_obs = MagicMock()
    parent_obs.id = "parent-obs-001"
    judge_obs = MagicMock()
    judge_obs.id = "judge-obs-001"
    langfuse_client.start_observation.side_effect = [parent_obs, judge_obs]

    with patch("eval.evaluator._propagate_attributes", return_value=MagicMock()):
        with patch(
            "eval.evaluator.evaluate_sample_active_fetch", new_callable=AsyncMock
        ) as mock_active:
            mock_active.return_value = mock_af_result
            with patch(
                "eval.evaluator.call_llm", new_callable=AsyncMock
            ) as mock_call_llm:
                mock_call_llm.return_value = _BATCH_PASS_RESPONSE

                await _evaluate_active_fetch_sample_with_tracing(
                    sample,
                    0,
                    llm_endpoint="http://test",
                    evaluation_langfuse_client=langfuse_client,
                )

    obs_calls = langfuse_client.start_observation.call_args_list
    judge_calls = [
        c for c in obs_calls if c[1].get("name") == "rag-active-fetch-judge"
    ]
    assert len(judge_calls) == 1
    assert judge_calls[0][1]["as_type"] == "generation"
    assert "prompt" in judge_calls[0][1]["input"]


async def test_active_fetch_records_scores_or_summary():
    """Verify that create_score is called with correct trace_id per criterion."""
    sample = EvalSample(question="test q", context="", expected_answer="test a")
    mock_af_result = {
        "sample_id": "sample_0",
        "question": "test q",
        "expected_answer": "test a",
        "llm_answer": "final answer",
        "tool_calls": [],
        "fetched_doc_ids": [],
        "fetched_text": "some context",
        "fetch_count": 0,
        "loop_terminated_early": False,
    }

    langfuse_client = MagicMock()
    parent_obs = MagicMock()
    parent_obs.id = "parent-obs-001"
    parent_obs.trace_id = "trace-abc-001"
    langfuse_client.start_observation.return_value = parent_obs

    with patch("eval.evaluator._propagate_attributes", return_value=MagicMock()):
        with patch(
            "eval.evaluator.evaluate_sample_active_fetch", new_callable=AsyncMock
        ) as mock_active:
            mock_active.return_value = mock_af_result
            with patch(
                "eval.evaluator.call_llm", new_callable=AsyncMock
            ) as mock_call_llm:
                mock_call_llm.return_value = _BATCH_PASS_RESPONSE

                result = await _evaluate_active_fetch_sample_with_tracing(
                    sample,
                    0,
                    llm_endpoint="http://test",
                    evaluation_langfuse_client=langfuse_client,
                )

    # create_score must be called once per criterion with the correct trace_id
    assert langfuse_client.create_score.call_count == len(CRITERIA)
    for call_args in langfuse_client.create_score.call_args_list:
        assert call_args[1]["trace_id"] == "trace-abc-001"
        assert call_args[1]["name"] in CRITERIA
        assert call_args[1]["data_type"] == "NUMERIC"

    # Result dict must carry scores, confidence, and criteria for the caller
    assert "scores" in result
    assert "confidence" in result
    assert "criteria" in result
    assert len(result["criteria"]) == len(CRITERIA)


async def test_active_fetch_with_langfuse_none_still_suppresses_backend_when_requested():
    """Verify skip header is sent even when evaluation_langfuse_client is None."""
    sample = EvalSample(question="test q", context="", expected_answer="test a")
    mock_af_result = {
        "sample_id": "sample_0",
        "question": "test q",
        "expected_answer": "test a",
        "llm_answer": "final answer",
        "tool_calls": [],
        "fetched_doc_ids": [],
        "fetched_text": "some context",
        "fetch_count": 0,
        "loop_terminated_early": False,
    }

    with patch(
        "eval.evaluator.evaluate_sample_active_fetch", new_callable=AsyncMock
    ) as mock_active:
        mock_active.return_value = mock_af_result
        with patch(
            "eval.evaluator.call_llm", new_callable=AsyncMock
        ) as mock_call_llm:
            mock_call_llm.return_value = _BATCH_PASS_RESPONSE

            result = await _evaluate_active_fetch_sample_with_tracing(
                sample,
                0,
                llm_endpoint="http://test",
                evaluation_langfuse_client=None,
            )

    # call_llm must still send the skip header without a langfuse client
    assert mock_call_llm.call_count == 1
    call_kwargs = mock_call_llm.call_args[1]
    assert call_kwargs.get("extra_headers") == {"X-Skip-Langfuse": "true"}

    # evaluate_sample_active_fetch must still be called with suppress_backend_langfuse
    assert mock_active.call_count == 1
    active_kwargs = mock_active.call_args[1]
    assert active_kwargs.get("suppress_backend_langfuse") is True

    # Result must include scores/confidence even without langfuse
    assert "scores" in result
    assert "confidence" in result
    assert "criteria" in result


async def test_active_fetch_no_double_recording():
    """Verify call_llm for the judge does NOT carry Langfuse params that would
    cause double recording (evaluator creates its own judge observation instead)."""
    sample = EvalSample(question="test q", context="", expected_answer="test a")
    mock_af_result = {
        "sample_id": "sample_0",
        "question": "test q",
        "expected_answer": "test a",
        "llm_answer": "final answer",
        "tool_calls": [],
        "fetched_doc_ids": [],
        "fetched_text": "some context",
        "fetch_count": 0,
        "loop_terminated_early": False,
    }

    langfuse_client = MagicMock()
    parent_obs = MagicMock()
    parent_obs.id = "parent-obs-001"
    langfuse_client.start_observation.return_value = parent_obs

    with patch("eval.evaluator._propagate_attributes", return_value=MagicMock()):
        with patch(
            "eval.evaluator.evaluate_sample_active_fetch", new_callable=AsyncMock
        ) as mock_active:
            mock_active.return_value = mock_af_result
            with patch(
                "eval.evaluator.call_llm", new_callable=AsyncMock
            ) as mock_call_llm:
                mock_call_llm.return_value = _BATCH_PASS_RESPONSE

                await _evaluate_active_fetch_sample_with_tracing(
                    sample,
                    0,
                    llm_endpoint="http://test",
                    evaluation_langfuse_client=langfuse_client,
                )

    # Judge call_llm must NOT receive langfuse params that would trigger its
    # own observation creation — the helper creates a separate judge observation.
    assert mock_call_llm.call_count == 1
    call_kwargs = mock_call_llm.call_args[1]
    assert call_kwargs.get("extra_headers") == {"X-Skip-Langfuse": "true"}
    assert call_kwargs.get("langfuse_client") is None
    assert call_kwargs.get("langfuse_trace_name") is None

    # The evaluator's Langfuse client should create exactly 2 observations:
    # one parent span and one judge generation — NOT a third one via call_llm.
    obs_calls = langfuse_client.start_observation.call_args_list
    names = [c[1].get("name") for c in obs_calls]
    assert "rag-active-fetch-sample" in names
    assert "rag-active-fetch-judge" in names
