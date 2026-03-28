from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eval.evaluator import (
    CRITERIA,
    CriterionResult,
    EvalResult,
    SampleResult,
    push_scores_to_langfuse,
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
                confidence=1.0,
                score=1.0,
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
    client.start_observation.return_value = trace
    return client


class TestPushScores:

    def test_push_scores_called_10_times(self) -> None:
        eval_result = _make_eval_result(n_samples=1)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        assert client.create_score.call_count == 5

    def test_score_names_match_criteria_exactly(self) -> None:
        eval_result = _make_eval_result(n_samples=1)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        names = [c.kwargs["name"] for c in client.create_score.call_args_list]
        expected_names = list(CRITERIA)

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

        assert client.start_observation.call_count == n
        assert client.create_score.call_count == n * 5

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

    def test_langfuse_environments_are_tagged(self) -> None:
        eval_result = _make_eval_result(n_samples=1)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        assert client.start_observation.call_args.kwargs["as_type"] == "evaluator"
class TestNoLangfuse:

    def test_no_langfuse_skips_push(self) -> None:
        eval_result = _make_eval_result()
        client = _make_langfuse_client()

        no_langfuse = True
        if not no_langfuse:
            push_scores_to_langfuse(eval_result, client)

        client.create_score.assert_not_called()
        client.start_observation.assert_not_called()

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

        trace_call = client.start_observation.call_args
        assert trace_call is not None
        assert "eval-sample_0" in trace_call.kwargs.get("name", "")

    def test_integration_end_to_end_10_scores_per_sample(self) -> None:
        eval_result = _make_eval_result(n_samples=2)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        assert client.create_score.call_count == 10

    def test_integration_trace_input_has_question(self) -> None:
        eval_result = _make_eval_result(n_samples=1)
        client = _make_langfuse_client()

        push_scores_to_langfuse(eval_result, client)

        trace_call = client.start_observation.call_args
        trace_input = trace_call.kwargs.get("input", {})
        assert "question" in trace_input
        assert trace_input["question"] == "What is K3s?"

    def test_criteria_names_are_lowercase_underscore(self) -> None:
        for criterion in CRITERIA:
            assert criterion == criterion.lower()
            assert " " not in criterion
            assert "/" not in criterion
