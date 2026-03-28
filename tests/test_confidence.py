from __future__ import annotations

import math

from eval.evaluator import extract_confidence


def _make_response(top_logprobs: list[dict]) -> dict:
    first_token = top_logprobs[0]["token"] if top_logprobs else ""
    first_lp = top_logprobs[0]["logprob"] if top_logprobs else 0.0
    return {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": first_token,
                            "logprob": first_lp,
                            "top_logprobs": top_logprobs,
                        }
                    ]
                }
            }
        ]
    }


def test_confidence_pass_only():
    response = _make_response([{"token": "Pass", "logprob": 0.0}])
    assert abs(extract_confidence(response) - 1.0) < 1e-9


def test_confidence_math_two_tokens():
    pass_lp = -0.5
    fail_lp = -1.5
    expected = math.exp(pass_lp) / (math.exp(pass_lp) + math.exp(fail_lp))
    response = _make_response([
        {"token": "Pass", "logprob": pass_lp},
        {"token": "Fail", "logprob": fail_lp},
    ])
    assert abs(extract_confidence(response) - expected) < 1e-9


def test_confidence_equal_logprobs():
    response = _make_response([
        {"token": "Pass", "logprob": -1.0},
        {"token": "Fail", "logprob": -1.0},
    ])
    assert abs(extract_confidence(response) - 0.5) < 1e-9


def test_confidence_five_tokens():
    tokens = [
        {"token": "Pass", "logprob": -0.2},
        {"token": "Fail", "logprob": -1.0},
        {"token": "Yes", "logprob": -2.0},
        {"token": "No", "logprob": -3.0},
        {"token": "Maybe", "logprob": -4.0},
    ]
    all_lps = [t["logprob"] for t in tokens]
    max_lp = max(all_lps)
    expected = math.exp(-0.2 - max_lp) / sum(math.exp(lp - max_lp) for lp in all_lps)
    response = _make_response(tokens)
    assert abs(extract_confidence(response) - expected) < 1e-9


def test_confidence_no_pass_token():
    response = _make_response([
        {"token": "Yes", "logprob": -0.5},
        {"token": "No", "logprob": -1.5},
    ])
    assert extract_confidence(response) == 0.0


def test_confidence_empty_response():
    assert extract_confidence({}) == 0.0


def test_confidence_missing_choices():
    assert extract_confidence({"choices": []}) == 0.0


def test_confidence_missing_logprobs():
    response = {"choices": [{"message": {"content": "Pass"}}]}
    assert extract_confidence(response) == 0.0


def test_confidence_empty_top_logprobs():
    response = {
        "choices": [
            {
                "logprobs": {
                    "content": [{"token": "Pass", "logprob": -0.1, "top_logprobs": []}]
                }
            }
        ]
    }
    assert extract_confidence(response) == 0.0


def test_confidence_case_insensitive_pass():
    response = _make_response([
        {"token": " pass", "logprob": -0.1},
        {"token": "Fail", "logprob": -2.0},
    ])
    conf = extract_confidence(response)
    assert conf > 0.0


def test_confidence_numerically_stable_large_logprobs():
    pass_lp = -0.1
    fail_lp = -0.2
    expected = math.exp(pass_lp) / (math.exp(pass_lp) + math.exp(fail_lp))
    response = _make_response([
        {"token": "Pass", "logprob": pass_lp},
        {"token": "Fail", "logprob": fail_lp},
    ])
    assert abs(extract_confidence(response) - expected) < 1e-9
