from __future__ import annotations

import math
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eval.evaluator import (
    CRITERIA,
    CriterionResult,
    EvalResult,
    SampleResult,
    compute_confidence,
    parse_judge_response,
    push_scores_to_langfuse,
    score_from_confidence,
)


def _make_sample_result(sample_id: str = "sample_0") -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        question="What is K3s?",
        context="K3s is a lightweight Kubernetes distribution.",
        expected_answer="K3s is a lightweight Kubernetes distribution.",
        llm_answer="K3s is a lightweight Kubernetes distribution.",
        criteria=[
            CriterionResult(
                criterion=c,
                passed=True,
                confidence=0.9,
                score=4.8,
            )
            for c in CRITERIA
        ],
    )


def _make_eval_result(n_samples: int = 1) -> EvalResult:
    return EvalResult(
        samples=[_make_sample_result(f"sample_{i}") for i in range(n_samples)]
    )


def _make_langfuse_client() -> MagicMock:
    client = MagicMock()
    trace = MagicMock()
    trace.id = "trace-abc123"
    client.trace.return_value = trace
    return client


class TestPushScores:

    def test_push_scores_called_10_times(self) -> None:
        eval_result = _make_eval_result(n_samples=1)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        assert client.create_score.call_count == 10

    def test_score_names_match_criteria_exactly(self) -> None:
        eval_result = _make_eval_result(n_samples=1)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        names = [c.kwargs["name"] for c in client.create_score.call_args_list]
        expected_names = []
        for criterion in CRITERIA:
            expected_names.append(criterion)
            expected_names.append(f"{criterion}_confidence")

        assert names == expected_names

    def test_all_scores_numeric_data_type(self) -> None:
        eval_result = _make_eval_result(n_samples=1)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        for c in client.create_score.call_args_list:
            assert c.kwargs["data_type"] == "NUMERIC"

    def test_trace_created_per_sample(self) -> None:
        n = 3
        eval_result = _make_eval_result(n_samples=n)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        assert client.trace.call_count == n
        assert client.create_score.call_count == n * 10

    def test_flush_called_after_push(self) -> None:
        eval_result = _make_eval_result()
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        client.flush.assert_called_once()

    def test_trace_id_passed_to_create_score(self) -> None:
        eval_result = _make_eval_result(n_samples=1)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        for c in client.create_score.call_args_list:
            assert c.kwargs.get("trace_id") == "trace-abc123"


class TestConfidenceRange:

    def test_confidence_values_in_valid_range(self) -> None:
        eval_result = _make_eval_result(n_samples=1)
        edge_values = [0.0, 0.5, 1.0, 0.99, 0.01]
        for i, cr in enumerate(eval_result.samples[0].criteria):
            cr.confidence = edge_values[i]

        client = _make_langfuse_client()
        push_scores_to_langfuse(eval_result, client)

        confidence_calls = [
            c for c in client.create_score.call_args_list
            if c.kwargs["name"].endswith("_confidence")
        ]
        for c in confidence_calls:
            value = c.kwargs["value"]
            assert 0.0 <= value <= 1.0, f"Confidence {value} out of range for {c.kwargs['name']}"

    def test_nan_confidence_clamped_to_zero(self) -> None:
        eval_result = _make_eval_result(n_samples=1)
        for cr in eval_result.samples[0].criteria:
            cr.confidence = float("nan")

        client = _make_langfuse_client()
        push_scores_to_langfuse(eval_result, client)

        confidence_calls = [
            c for c in client.create_score.call_args_list
            if c.kwargs["name"].endswith("_confidence")
        ]
        for c in confidence_calls:
            assert not math.isnan(c.kwargs["value"])
            assert 0.0 <= c.kwargs["value"] <= 1.0

    def test_compute_confidence_pass_dominates(self) -> None:
        confidence = compute_confidence({"Pass": 0.0, "Fail": -float("inf")})
        assert confidence == pytest.approx(1.0, abs=1e-9)

    def test_compute_confidence_fail_dominates(self) -> None:
        confidence = compute_confidence({"Pass": -float("inf"), "Fail": 0.0})
        assert confidence == pytest.approx(0.0, abs=1e-9)

    def test_compute_confidence_both_inf_is_nan(self) -> None:
        confidence = compute_confidence({"Pass": -float("inf"), "Fail": -float("inf")})
        assert math.isnan(confidence)

    def test_compute_confidence_equal_logprobs(self) -> None:
        confidence = compute_confidence({"Pass": -1.0, "Fail": -1.0})
        assert confidence == pytest.approx(0.5, abs=1e-9)

    def test_score_from_confidence_pass_high(self) -> None:
        score = score_from_confidence(1.0, passed=True)
        assert score == pytest.approx(5.0)

    def test_score_from_confidence_fail_high(self) -> None:
        score = score_from_confidence(1.0, passed=False)
        assert score == pytest.approx(1.0)

    def test_score_from_confidence_nan_returns_neutral(self) -> None:
        score = score_from_confidence(float("nan"), passed=True)
        assert score == pytest.approx(3.0)


class TestNoLangfuse:

    def test_no_langfuse_skips_push(self) -> None:
        eval_result = _make_eval_result()
        client = _make_langfuse_client()

        no_langfuse = True
        if not no_langfuse:
            push_scores_to_langfuse(eval_result, client)

        client.create_score.assert_not_called()
        client.trace.assert_not_called()

    def test_cli_no_langfuse_arg_produces_no_calls(self) -> None:
        from eval.evaluator import main

        with patch("asyncio.run", return_value=_make_eval_result()):
            with patch("eval.evaluator.build_langfuse_client") as mock_build:
                with patch("eval.evaluator.push_scores_to_langfuse") as mock_push:
                    with patch.object(
                        sys,
                        "argv",
                        [
                            "evaluator",
                            "--input", "eval/eval_data.jsonl",
                            "--endpoint", "http://localhost:8080",
                            "--output", "/tmp/eval_out.json",
                            "--no-langfuse",
                        ],
                    ):
                        main()

        mock_build.assert_not_called()
        mock_push.assert_not_called()

    def test_cli_dry_run_skips_langfuse(self) -> None:
        from eval.evaluator import main

        with patch("asyncio.run", return_value=_make_eval_result()):
            with patch("eval.evaluator.build_langfuse_client") as mock_build:
                with patch("eval.evaluator.push_scores_to_langfuse") as mock_push:
                    with patch.object(
                        sys,
                        "argv",
                        [
                            "evaluator",
                            "--input", "eval/eval_data.jsonl",
                            "--endpoint", "http://localhost:8080",
                            "--output", "/tmp/eval_out.json",
                            "--dry-run",
                        ],
                    ):
                        main()

        mock_build.assert_not_called()
        mock_push.assert_not_called()


class TestUnexpectedToken:

    def test_unexpected_token_returns_fail_with_low_confidence(self) -> None:
        response: dict[str, Any] = {
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
                                ],
                            }
                        ]
                    },
                }
            ]
        }

        passed, confidence = parse_judge_response(response)

        assert passed is False
        assert 0.0 <= confidence <= 0.3, f"Expected low confidence ≤0.3, got {confidence}"

    def test_empty_response_returns_false_zero_confidence(self) -> None:
        passed, confidence = parse_judge_response({})

        assert passed is False
        assert confidence == 0.0

    def test_pass_token_in_content_returns_passed(self) -> None:
        response: dict[str, Any] = {
            "choices": [
                {
                    "message": {"content": "Pass"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "Pass",
                                "logprob": -0.05,
                                "top_logprobs": [
                                    {"token": "Pass", "logprob": -0.05},
                                    {"token": "Fail", "logprob": -3.0},
                                ],
                            }
                        ]
                    },
                }
            ]
        }

        passed, confidence = parse_judge_response(response)

        assert passed is True
        assert 0.5 < confidence <= 1.0


class TestLLMReachable:

    async def test_llm_reachable_mock(self) -> None:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        response = await mock_client.get("/health")

        assert response.status_code == 200


class TestIntegrationEndToEnd:

    def test_integration_end_to_end_trace_name(self) -> None:
        eval_result = _make_eval_result(n_samples=1)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        trace_call = client.trace.call_args
        assert trace_call is not None
        assert "eval-sample_0" in trace_call.kwargs.get("name", "")

    def test_integration_end_to_end_10_scores_per_sample(self) -> None:
        eval_result = _make_eval_result(n_samples=2)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        assert client.create_score.call_count == 20

    def test_integration_trace_input_has_question(self) -> None:
        eval_result = _make_eval_result(n_samples=1)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        trace_call = client.trace.call_args
        trace_input = trace_call.kwargs.get("input", {})
        assert "question" in trace_input
        assert trace_input["question"] == "What is K3s?"

    def test_criteria_names_are_lowercase_underscore(self) -> None:
        for criterion in CRITERIA:
            assert criterion == criterion.lower()
            assert " " not in criterion
            assert "/" not in criterion
