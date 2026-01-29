from __future__ import annotations

from rl.reward import compute_reward


def test_reward_progress_and_collision() -> None:
    comps = compute_reward(
        prev_distance=5.0,
        current_distance=4.0,
        collided=True,
        reached_goal=False,
        action_norm=0.0,
        jerk_norm=0.0,
    )
    # Progress +1, collision -1 scaled by 5 => -5, total = 1 - 5 = -4
    assert comps.progress == 1.0
    assert comps.collision_penalty == -1.0
    assert comps.goal_bonus == 0.0
    assert comps.total() == -4.0

