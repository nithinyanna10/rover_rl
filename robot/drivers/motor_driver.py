from __future__ import annotations


class MotorDriver:
    """Stub motor driver for Raspberry Pi.

    TODO:
    - Implement hardware-specific initialization (e.g., GPIO, I2C, CAN).
    - Convert (v, w) commands into wheel velocities and send to motor controllers.
    """

    def __init__(self) -> None:
        # TODO: Initialize hardware interfaces here.
        pass

    def set_velocity(self, v: float, w: float) -> None:
        """Set linear and angular command.

        Parameters
        ----------
        v : float
            Linear velocity command [m/s].
        w : float
            Angular velocity command [rad/s].
        """
        # TODO: Map (v, w) into specific wheel commands.
        # For now, this is a no-op stub.
        _ = v, w

    def stop(self) -> None:
        """Immediately stop all motors."""
        # TODO: Send zero commands to all motors.
        pass

