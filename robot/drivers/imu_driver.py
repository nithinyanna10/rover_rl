from __future__ import annotations

from typing import Dict, Any


class ImuDriver:
    """Stub IMU driver.

    TODO:
    - Connect to an actual IMU over I2C/SPI/UART.
    - Return orientation, angular velocity, and linear acceleration.
    """

    def __init__(self) -> None:
        # TODO: Initialize IMU hardware
        pass

    def read(self) -> Dict[str, Any]:
        """Return latest IMU data (stubbed)."""
        # Example structure; adapt to your IMU:
        return {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw_rate": 0.0,
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 0.0,
        }

