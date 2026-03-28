from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from eval.evaluator import (
    CRITERIA,
    EvalSample,
    SampleResult,
    JudgeResponseParseError,
    build_batch_judge_prompt,
    parse_batch_judge_response,
    evaluate_sample,
    load_dataset,
)

EVAL_DATA_PATH = Path(__file__).parent.parent / "eval" / "eval_data.jsonl"

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
