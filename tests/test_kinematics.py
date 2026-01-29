from __future__ import annotations

import math

from rover_sim.rover import Rover


def test_rover_forward_motion() -> None:
    rng_seed = 0
    import random

    rng = random.Random(rng_seed)
    rover = Rover(
        radius=0.4,
        max_linear_speed=1.0,
        max_angular_speed=1.0,
        linear_accel_limit=10.0,
        angular_accel_limit=10.0,
        rng=rng,
    )
    rover.reset(x=0.0, y=0.0, yaw=0.0)

    dt = 0.1
    v_cmd = 1.0
    w_cmd = 0.0
    rover.step(v_cmd, w_cmd, dt)
    state = rover.get_state()

    assert math.isclose(state.x, v_cmd * dt, rel_tol=1e-4)
    assert math.isclose(state.y, 0.0, abs_tol=1e-6)

