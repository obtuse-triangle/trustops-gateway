from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from eval.evaluator import (
    CRITERIA,
    EvalSample,
    SampleResult,
    build_judge_prompt,
    evaluate_sample,
    load_dataset,
)

EVAL_DATA_PATH = Path(__file__).parent.parent / "eval" / "eval_data.jsonl"

_PASS_RESPONSE = {
    "choices": [
        {
            "message": {"content": "Pass"},
            "logprobs": {
                "content": [
                    {
                        "token": "Pass",
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": "Pass", "logprob": -0.1},
                            {"token": "Fail", "logprob": -2.5},
                        ],
                    }
                ]
            },
        }
    ]
}

_UNEXPECTED_TOKEN_RESPONSE = {
    "choices": [
        {
            "message": {"content": "Yes"},
            "logprobs": {
                "content": [
                    {
                        "token": "Yes",
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": "Yes", "logprob": -0.1},
                            {"token": "No", "logprob": -2.0},
                        ],
                    }
                ]
            },
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


def test_build_judge_prompt_contains_criterion():
    for criterion in CRITERIA:
        prompt = build_judge_prompt(
            criterion=criterion,
            question="test question",
            context="test context",
            expected_answer="test answer",
            response="test response",
        )
        assert criterion in prompt


def test_build_judge_prompt_contains_pass_fail_instructions():
    prompt = build_judge_prompt(
        criterion="Faithfulness",
        question="What is K3s?",
        context="K3s is a lightweight Kubernetes distribution.",
        expected_answer="K3s is lightweight Kubernetes.",
        response="K3s is Kubernetes.",
    )
    assert "Pass" in prompt
    assert "Fail" in prompt
    assert "What is K3s?" in prompt


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
        return _PASS_RESPONSE

    semaphore = asyncio.Semaphore(1)
    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        result = await evaluate_sample(
            sample, 0, llm_endpoint="http://localhost:8080", semaphore=semaphore
        )

    assert isinstance(result, SampleResult)
    assert result.sample_id == "sample_0"
    assert [c.criterion for c in result.criteria] == CRITERIA
    assert call_count == 1 + len(CRITERIA)


async def test_unexpected_token():
    sample = EvalSample(
        question="test",
        context="test ctx",
        expected_answer="test ans",
    )

    async def mock_call(prompt, *, endpoint, **kwargs):
        return _UNEXPECTED_TOKEN_RESPONSE

    semaphore = asyncio.Semaphore(1)
    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        result = await evaluate_sample(
            sample, 0, llm_endpoint="http://localhost:8080", semaphore=semaphore
        )

    assert isinstance(result, SampleResult)
    for criterion in CRITERIA:
        item = next(c for c in result.criteria if c.criterion == criterion)
        assert item.passed is False
        assert item.confidence == 0.0


async def test_rate_limiting():
    active_count = [0]
    max_concurrent = [0]

    async def mock_call(prompt, *, endpoint, **kwargs):
        active_count[0] += 1
        max_concurrent[0] = max(max_concurrent[0], active_count[0])
        await asyncio.sleep(0.02)
        active_count[0] -= 1
        return _PASS_RESPONSE

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
        return _PASS_RESPONSE

    semaphore = asyncio.Semaphore(1)
    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        result = await evaluate_sample(
            sample, 42, llm_endpoint="http://localhost:8080", semaphore=semaphore
        )

    assert result.sample_id == "sample_42"


async def test_evaluate_sample_all_criteria_scored():
    sample = EvalSample(question="q", context="c", expected_answer="e")

    async def mock_call(prompt, *, endpoint, **kwargs):
        return _PASS_RESPONSE

    semaphore = asyncio.Semaphore(1)
    with patch("eval.evaluator.call_llm", side_effect=mock_call):
        result = await evaluate_sample(
            sample, 0, llm_endpoint="http://localhost:8080", semaphore=semaphore
        )

    for criterion in CRITERIA:
        item = next(c for c in result.criteria if c.criterion == criterion)
        assert item.criterion == criterion
        assert item.passed is True
        assert 0.0 <= item.confidence <= 1.0
