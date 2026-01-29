RL Rover: Waypoint Navigation + Obstacle Avoidance
==================================================

This repository implements a **full RL rover stack**:

- Lightweight 2D simulator with LiDAR, obstacles, and goal.
- PPO training with Stable-Baselines3.
- Standardized evaluation and metrics.
- Real-time telemetry dashboard (Streamlit).
- Sim2Real interface with a `RobotAPI` abstraction and safety shield.

All code targets **Python 3.11**.

---

Coordinate System & Kinematics
------------------------------

- **World frame**:
  - \(x\): right (meters)
  - \(y\): up (meters)
  - Yaw \(\theta\): radians, counter-clockwise from +x.
- **Rover**:
  - Differential drive, controlled by **linear velocity** \(v\) (m/s) and **angular velocity** \(\omega\) (rad/s).
  - State update (Euler integration):
    - \(x_{t+1} = x_t + v \cos(\theta_t)\, \Delta t\)
    - \(y_{t+1} = y_t + v \sin(\theta_t)\, \Delta t\)
    - \(\theta_{t+1} = \theta_t + \omega\, \Delta t\)

LiDAR & World
-------------

- **LiDAR**:
  - 2D planar LiDAR, N rays, configurable FOV and max range.
  - Returns per-ray **distance** with **Gaussian noise** (clipped to \([0, \text{max\_range}]\)).
  - Optional "front ultrasonic" is the minimum distance in a front angular sector.
- **World**:
  - Axis-aligned rectangular world.
  - Obstacles are axis-aligned rectangles.
  - Rover is a circle; collision = circle-rectangle overlap.

RL Environment
--------------

- **Interface**: `gymnasium.Env`
- **Observation space**:
  - Normalized LiDAR distances \(\in [0, 1]\) (downsampled / configurable).
  - Goal relative vector in **rover frame** \((dx, dy)\).
  - Current rover linear and angular velocity.
- **Action space**:
  - Continuous `Box(low=[v_min, w_min], high=[v_max, w_max])`
  - Actions are linear and angular velocities.
- **Reward shaping** (see `rl/reward.py`):
  - **Progress reward**: positive when distance to goal decreases.
  - **Collision penalty**: large negative when collision occurs.
  - **Smoothness penalty**: small penalty on action magnitude and jerk (change in action).
  - **Goal bonus**: positive when reaching the goal within tolerance.
- **Termination**:
  - Goal reached (within radius).
  - Collision with any obstacle or boundary.
  - Timeout (max steps per episode).

Curriculum & Domain Randomization
---------------------------------

- **Curriculum**:
  - Early training: sparse obstacles.
  - Later training: more cluttered maps and random obstacle fields.
  - Controlled via `configs/train_ppo.yaml`.
- **Domain randomization** (per episode):
  - Friction / effective max speed scaling.
  - Sensor noise scaling.
  - Wheel slip (perturb linear/angular velocities).
  - Simple actuation latency (actions buffered by 0–k steps).

Safety Shield
-------------

- Implemented in `rl/shield.py` as a **Simple LiDAR Shield**:
  - Checks minimum LiDAR range.
  - If min distance \< threshold:
    - Stops forward motion.
    - Turns away from closer side (based on left/right LiDAR sectors).
  - Used both in **simulation** and **real robot control loops**.

Telemetry & Dashboard
---------------------

- `telemetry/logger.py`:
  - Writes structured JSONL telemetry:
    - Rover pose (x, y, yaw)
    - Velocities and actions
    - LiDAR summaries (min/max/selected beams)
    - Reward components
    - Episode statistics and timing (FPS / step time)
- `telemetry/streamlit_app.py`:
  - Reads one or more telemetry JSONL files.
  - Real-time updates via auto-refresh (1–5 Hz).
  - Live plots:
    - Rover path, obstacles, and goal.
    - Latest LiDAR scan (polar or line plot).
    - Reward component history.
    - Episode metrics and step timing.

Sim2Real: RobotAPI Abstraction
------------------------------

- `robot/api.py` defines `RobotAPI`:
  - `reset(goal)`, `read_sensors()`, `send_command(v, w)`, `get_state()`, `stop()`.
- `robot/sim_robot.py`:
  - Wraps the simulator environment and rover as a `RobotAPI`.
- `robot/real_robot.py`:
  - Raspberry Pi–oriented placeholder implementation:
    - Uses `drivers/motor_driver.py`, `imu_driver.py`, `lidar_driver.py` (stubs).
    - Clear TODOs and safe defaults (stop on error, stop on too-close obstacle).
  - Provides a **control loop** runner:
    - Fixed-rate (10–20 Hz) loop.
    - Reads sensors, runs the policy, applies safety shield, sends motor commands.
    - Optional teleop override hook.

Architecture Overview
---------------------

High-level component diagram:

  +------------------------+        +------------------------+
  |      Configs (.yaml)  |        |   Telemetry Dashboard  |
  +-----------+------------+        |  (Streamlit, JSONL)   |
              |                     +-----------+-----------+
              v                                 ^
  +-----------+------------+        telemetry   |
  |  RL (PPO, reward,      |--------------------+
  |  shield, train/eval)   |
  +-----------+------------+
              |
              v
  +-----------+------------+
  |  Gymnasium Env         |
  |  (rover_sim.env)       |
  +-----------+------------+
              |
    obs,rew   v    actions
  +-----------+------------+       +-----------------------+
  |  Rover + Sensors       |<----->|   RobotAPI           |
  |  (world, rover, lidar) |       |  (sim/real backends) |
  +-----------+------------+       +-----------------------+
              |
              v
       +------+------+
       |  Renderer   |
       |  (pygame)   |
       +-------------+

Repo Layout
-----------

- `configs/`: YAML configs for sim, training, eval, and real robot.
- `rover_sim/`: 2D simulator, world, rover, sensors, rendering, Gymnasium env, maps.
- `rl/`: PPO training, evaluation, reward shaping, safety shield, policy export, inference.
- `telemetry/`: JSONL logger and Streamlit dashboard.
- `robot/`: `RobotAPI`, simulated robot, and Raspberry Pi stubs and drivers.
- `scripts/`: convenience scripts to run sim, training, eval, and dashboard.
- `tests/`: unit and smoke tests.

Installation
------------

Requirements:

- Python 3.11
- A working C compiler (for some Python packages, e.g., `pygame`).

From your Downloads folder:

```bash
cd ~/Downloads
python3.11 -m venv .venv-rover
source .venv-rover/bin/activate
cd rover_rl
pip install -r requirements.txt
```

Quickstart: Run the Simulator
-----------------------------

From inside the `rover_rl` directory:

```bash
python scripts/run_sim.py --config configs/sim.yaml
```

Features:

- Top-down 2D world.
- Obstacles, rover, goal, LiDAR rays, and trajectory visualization.
- Keyboard teleop:
  - `W/S`: increase/decrease linear velocity.
  - `A/D`: increase/decrease angular velocity.
  - `SPACE`: emergency stop.
  - `ESC` or window close: exit.

Train PPO (Smoke Run)
---------------------

To train a PPO policy (short smoke run by default):

```bash
python -m rl.train --config configs/train_ppo.yaml
```

Outputs:

- Checkpoints in `runs/<timestamp>/checkpoints/`.
- TensorBoard logs in `runs/<timestamp>/tb/`.
- Monitor CSV logs in `runs/<timestamp>/monitor/`.

You can watch training with:

```bash
tensorboard --logdir runs
```

Evaluate a Checkpoint
---------------------

Given a trained checkpoint (e.g. `runs/2026-01-28_12-00-00/checkpoints/best_model.zip`):

```bash
python -m rl.evaluate \
  --config configs/eval.yaml \
  --model-path runs/2026-01-28_12-00-00/checkpoints/best_model.zip
```

Outputs:

- Metrics JSON (example) in `runs/<timestamp>/eval/metrics.json`:

```json
{
  "success_rate": 0.82,
  "collision_rate": 0.10,
  "avg_time_to_goal": 34.7,
  "avg_path_length": 12.3,
  "avg_episode_reward": 56.1
}
```

- Plots (PNG):
  - `learning_curve.png` – episode reward vs timesteps.
  - `success_by_map.png` – success rate per fixed map.

Run Inference with Visualization
--------------------------------

To run a trained policy in the simulator with visualization and telemetry logging:

```bash
python -m rl.inference \
  --config configs/eval.yaml \
  --model-path runs/2026-01-28_12-00-00/checkpoints/best_model.zip \
  --telemetry-path telemetry_logs/inference.jsonl
```

This:

- Loads the PPO policy.
- Wraps actions with the safety shield.
- Runs in the simulator with pygame visualization.
- Logs structured telemetry to the specified JSONL file.

Telemetry Dashboard
-------------------

To run the real-time telemetry dashboard (against telemetry logs):

```bash
streamlit run telemetry/streamlit_app.py -- \
  --log-path telemetry_logs/inference.jsonl
```

Dashboard panels:

- **Rover state**: pose, velocity, action outputs.
- **LiDAR scan**: latest scan as a polar/line plot.
- **Map view**: obstacles, rover trail, and goal.
- **Rewards**: components and total reward over time.
- **Performance**: FPS and episode stats.

Sim2Real: Raspberry Pi Deployment
---------------------------------

High-level steps to deploy on a Raspberry Pi rover:

1. **Train a policy in simulation** (as above) and export it:

   ```bash
   python -m rl.policy_export \
     --model-path runs/2026-01-28_12-00-00/checkpoints/best_model.zip \
     --output-path exported_policies/ppo_rover.ts
   ```

2. **Copy artifacts to the Pi**:
   - `exported_policies/ppo_rover.ts`
   - `configs/real_robot.yaml`
   - The `robot/` and `rl/` packages (or install from a packaged wheel).

3. **Implement hardware drivers** in `robot/drivers/`:
   - `motor_driver.py`: connect to motor controllers (e.g., PWM, CAN).
   - `imu_driver.py`: connect to IMU (e.g., via I2C/SPI).
   - `lidar_driver.py`: connect to your LiDAR sensor (e.g., RPLIDAR).

4. **Configure RobotAPI** in `robot/real_robot.py`:
   - Wire up the drivers.
   - Confirm `read_sensors()` returns LiDAR, IMU, and odometry as expected.

5. **Run the control loop** (example):

   ```bash
   python -m robot.real_robot \
     --config configs/real_robot.yaml \
     --policy exported_policies/ppo_rover.ts
   ```

   This starts:

   - A fixed-rate (10–20 Hz) control loop.
   - Sensor reads -> policy inference -> safety shield -> motor commands.
   - Safe shutdown on exceptions or close-range obstacles.

Safety Notes
------------

- **Simulator shield**:
  - Prevents the rover from driving directly into obstacles when LiDAR detects very close objects.
- **Real robot shield**:
  - If minimum LiDAR distance \< threshold:
    - Immediately commands zero linear velocity and a turning motion.
  - On any unhandled exception in the control loop:
    - Motor commands are set to zero and the loop exits.
- **Logged data**:
  - Telemetry logs may include:
    - Pose, velocities, actions.
    - LiDAR distances (optionally downsampled).
    - Reward components and episode outcomes.
  - Do **not** log any sensitive data from your environment (e.g., camera feeds, GPS locations) unless explicitly required and handled securely.

Verification Checklist
----------------------

Once everything is installed and configured, you can run:

1. **Run sim (manual control)**:

   ```bash
   python scripts/run_sim.py --config configs/sim.yaml
   ```

2. **Train PPO (short smoke run, e.g., 50k steps)**:

   ```bash
   python -m rl.train --config configs/train_ppo.yaml --total-timesteps 50000
   ```

3. **Evaluate a saved checkpoint**:

   ```bash
   python -m rl.evaluate \
     --config configs/eval.yaml \
     --model-path runs/2026-01-28_12-00-00/checkpoints/best_model.zip
   ```

4. **Run inference visualization with telemetry logging**:

   ```bash
   python -m rl.inference \
     --config configs/eval.yaml \
     --model-path runs/2026-01-28_12-00-00/checkpoints/best_model.zip \
     --telemetry-path telemetry_logs/inference.jsonl
   ```

5. **Run telemetry dashboard**:

   ```bash
   streamlit run telemetry/streamlit_app.py -- \
     --log-path telemetry_logs/inference.jsonl
   ```

