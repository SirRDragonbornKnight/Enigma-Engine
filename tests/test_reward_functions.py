from __future__ import annotations


def test_format_reward_requires_think_and_answer():
    from enigma_engine.core.reward_functions import format_reward

    assert format_reward("<think>2+2=4</think>\n4") == 1.0
    assert format_reward("<think>2+2=4</think>") == 0.0
    assert format_reward("4") == 0.0


def test_math_reward_with_explicit_ground_truth():
    from enigma_engine.core.reward_functions import math_reward

    assert math_reward("What is 2 + 2?", "<think>calc</think>\n4", ground_truth="4") == 1.0
    assert math_reward("What is 2 + 2?", "5", ground_truth="4") == 0.0


def test_math_reward_can_infer_simple_arithmetic_from_prompt():
    from enigma_engine.core.reward_functions import math_reward

    assert math_reward("Compute 7 * 6.", "42") == 1.0
    assert math_reward("Compute 7 * 6.", "41") == 0.0


def test_reasoning_reward_matches_grpo2_contract_example():
    from enigma_engine.core.reward_functions import reasoning_reward

    assert (
        reasoning_reward(
            "What is 2 + 2?",
            "<think>2+2=4</think>\n4",
            ground_truth="4",
        )
        == 1.0
    )
    assert (
        reasoning_reward(
            "What is 2 + 2?",
            "5",
            ground_truth="4",
        )
        == 0.0
    )
