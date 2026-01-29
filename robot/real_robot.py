from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

from robot.api import RobotAPI
from robot.drivers.motor_driver import MotorDriver
from robot.drivers.imu_driver import ImuDriver
from robot.drivers.lidar_driver import LidarDriver
from rl.shield import LidarSafetyShield, ShieldConfig
from telemetry.logger import TelemetryLogger


class RealRobot(RobotAPI):
    """Placeholder real-robot implementation targeting Raspberry Pi.

    This class wires together higher-level control but leaves hardware-specific
    details to the driver stubs in `robot/drivers/`.
    """

    def __init__(
        self,
        motor: MotorDriver,
        imu: ImuDriver,
        lidar: LidarDriver,
        safety_cfg: ShieldConfig,
        telemetry_logger: TelemetryLogger | None = None,
    ) -> None:
        self.motor = motor
        self.imu = imu
        self.lidar = lidar
        self.shield = LidarSafetyShield(safety_cfg)
        self.telemetry_logger = telemetry_logger

        self._goal: Tuple[float, float] = (0.0, 0.0)
        self._state: Dict[str, float] = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "v": 0.0,
            "w": 0.0,
        }

    def reset(self, goal: Tuple[float, float]) -> None:
        self._goal = goal
        # TODO: Reset odometry to (0,0,0) or current location depending on setup.
        self._state.update({"x": 0.0, "y": 0.0, "yaw": 0.0, "v": 0.0, "w": 0.0})
        self.stop()

    def read_sensors(self) -> Dict[str, Any]:
        # LiDAR scan
        lidar_ranges = self.lidar.get_scan()
        # IMU / odometry (stubbed)
        imu_data = self.imu.read()
        # TODO: Integrate wheel or visual odometry to update pose.

        return {
            "pose": dict(self._state),
            "lidar_ranges": lidar_ranges,
            "imu": imu_data,
            "goal": {"x": self._goal[0], "y": self._goal[1]},
        }

    def get_state(self) -> Dict[str, float]:
        return dict(self._state)

    def send_command(self, v: float, w: float) -> None:
        # TODO: Convert (v, w) into wheel commands using your robot's kinematics.
        self.motor.set_velocity(v, w)
        self._state["v"] = float(v)
        self._state["w"] = float(w)

    def stop(self) -> None:
        self.motor.stop()
        self._state["v"] = 0.0
        self._state["w"] = 0.0


def run_control_loop() -> None:
    """Entry point: real-robot control loop with policy and safety shield."""
    parser = argparse.ArgumentParser(description="Real robot control loop (Raspberry Pi).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/real_robot.yaml",
        help="Path to real_robot YAML config.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to TorchScript policy artifact.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    robot_cfg = cfg["robot"]
    policy_cfg = cfg["policy"]
    lidar_cfg = cfg["lidar"]
    logging_cfg = cfg["logging"]

    control_rate_hz = float(robot_cfg.get("control_rate_hz", 15.0))
    dt = 1.0 / control_rate_hz

    telemetry_logger = TelemetryLogger(logging_cfg["telemetry_path"])

    # Instantiate drivers (stubs to be implemented for your hardware)
    motor = MotorDriver()
    imu = ImuDriver()
    lidar = LidarDriver(
        num_rays=int(lidar_cfg["num_rays"]),
        fov_deg=float(lidar_cfg["fov_deg"]),
        max_range=float(lidar_cfg["max_range"]),
    )

    safety_cfg = ShieldConfig(
        min_distance=float(robot_cfg["safety_min_distance"]),
        turn_angular_speed=float(robot_cfg["max_angular_speed"]),
    )

    robot = RealRobot(
        motor=motor,
        imu=imu,
        lidar=lidar,
        safety_cfg=safety_cfg,
        telemetry_logger=telemetry_logger,
    )

    # Load TorchScript policy
    policy = torch.jit.load(policy_cfg["artifact_path"], map_location="cpu")
    policy.eval()

    # TODO: Choose a suitable goal for your environment here.
    robot.reset(goal=(0.0, 0.0))

    print("Starting control loop. Press Ctrl+C to stop.")

    step_idx = 0
    try:
        while True:
            t_start = time.time()

            sensors = robot.read_sensors()
            lidar_ranges = sensors["lidar_ranges"]

            # Build observation vector compatible with training pipeline.
            # You will likely need to adapt this based on your deployed obs format.
            obs = np.zeros((1, len(lidar_ranges) + 4), dtype=np.float32)
            obs[0, : len(lidar_ranges)] = np.array(lidar_ranges, dtype=np.float32)
            # TODO: fill in goal vector and velocities from sensors/odometry.

            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs)
                action_tensor = policy(obs_tensor)
            action = action_tensor.numpy()[0]

            # Safety shield
            shield = robot.shield
            shielded_action, overridden = shield.filter_action(
                action=action,
                lidar_ranges=lidar_ranges,
                lidar_fov_deg=lidar_cfg["fov_deg"],
            )

            v_cmd = float(shielded_action[0])
            w_cmd = float(shielded_action[1])

            robot.send_command(v_cmd, w_cmd)

            # Telemetry
            telemetry_logger.log_step(
                {
                    "step": step_idx,
                    "pose": robot.get_state(),
                    "action": [v_cmd, w_cmd],
                    "action_overridden": overridden,
                    "lidar": {
                        "ranges": lidar_ranges,
                        "min_range": float(np.min(lidar_ranges))
                        if lidar_ranges
                        else None,
                    },
                }
            )

            step_idx += 1

            elapsed = time.time() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Stopping control loop (KeyboardInterrupt).")
    except Exception as exc:  # noqa: BLE001
        print(f"Exception in control loop: {exc}", file=sys.stderr)
    finally:
        robot.stop()
        telemetry_logger.close()


if __name__ == "__main__":
    run_control_loop()

