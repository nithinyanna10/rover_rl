from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--log-path",
        type=str,
        default="telemetry_logs/inference.jsonl",
        help="Path to telemetry JSONL log file.",
    )
    return parser.parse_args()


def load_telemetry(path: str, max_rows: int = 2000) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records)
    return df.tail(max_rows)


def main() -> None:
    args = parse_args()

    st.set_page_config(page_title="RL Rover Telemetry", layout="wide")
    st.title("RL Rover Telemetry Dashboard")

    status_placeholder = st.empty()

    # Layout
    col1, col2 = st.columns(2)

    map_fig = col1.empty()
    lidar_fig = col2.empty()

    reward_fig = st.empty()
    stats_placeholder = st.empty()

    refresh_interval = st.sidebar.slider("Refresh interval (s)", 0.5, 5.0, 1.0, 0.5)

    while True:
        df = load_telemetry(args.log_path)
        if df.empty:
            status_placeholder.info(f"Waiting for telemetry at '{args.log_path}'...")
            time.sleep(refresh_interval)
            continue

        status_placeholder.success(f"Streaming from '{args.log_path}' ({len(df)} records)")

        latest = df.iloc[-1]

        # Rover pose and action
        st.sidebar.subheader("Rover State")
        st.sidebar.write(
            f"x={latest.get('pose.x', 0.0):.2f}, "
            f"y={latest.get('pose.y', 0.0):.2f}, "
            f"yaw={latest.get('pose.yaw', 0.0):.2f}"
        )
        st.sidebar.write(
            f"v={latest.get('pose.v', 0.0):.2f}, "
            f"w={latest.get('pose.w', 0.0):.2f}"
        )
        st.sidebar.write(
            f"action_v={latest.get('action.0', 0.0):.2f}, "
            f"action_w={latest.get('action.1', 0.0):.2f}"
        )

        # Map view (2D path)
        with map_fig.container():
            fig, ax = plt.subplots()
            if "pose.x" in df.columns and "pose.y" in df.columns:
                ax.plot(df["pose.x"], df["pose.y"], "-y", label="Path")
                ax.scatter([latest.get("pose.x", 0.0)], [latest.get("pose.y", 0.0)], c="b", label="Rover")
            if "goal.x" in df.columns and "goal.y" in df.columns:
                ax.scatter(df["goal.x"].iloc[-1], df["goal.y"].iloc[-1], c="g", marker="*", label="Goal")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title("Rover Path")
            ax.legend(loc="upper right")
            map_fig.pyplot(fig)
            plt.close(fig)

        # LiDAR scan
        if "lidar.ranges" in df.columns:
            ranges_series = df["lidar.ranges"].dropna()
            if not ranges_series.empty:
                latest_ranges = ranges_series.iloc[-1]
                if isinstance(latest_ranges, list):
                    lidar_vals = np.array(latest_ranges, dtype=float)
                    with lidar_fig.container():
                        fig2, ax2 = plt.subplots()
                        ax2.plot(lidar_vals)
                        ax2.set_ylim(0, np.max(lidar_vals) * 1.1 if np.max(lidar_vals) > 0 else 1.0)
                        ax2.set_xlabel("Beam index")
                        ax2.set_ylabel("Distance [m]")
                        ax2.set_title("Latest LiDAR Scan")
                        lidar_fig.pyplot(fig2)
                        plt.close(fig2)

        # Reward components history
        reward_cols = [
            "reward_components.progress",
            "reward_components.collision_penalty",
            "reward_components.goal_bonus",
            "reward_components.action_mag_penalty",
            "reward_components.jerk_penalty",
            "reward",
        ]
        existing = [c for c in reward_cols if c in df.columns]
        if existing:
            with reward_fig.container():
                fig3, ax3 = plt.subplots()
                for col in existing:
                    ax3.plot(df[col].values, label=col)
                ax3.set_title("Reward Components")
                ax3.set_xlabel("Step")
                ax3.legend(loc="upper right")
                reward_fig.pyplot(fig3)
                plt.close(fig3)

        # Stats
        fps = latest.get("fps", None)
        step_time = latest.get("step_time", None)
        stats_text = "Episode stats:\n"
        if fps is not None:
            stats_text += f"- FPS: {fps:.1f}\n"
        if step_time is not None:
            stats_text += f"- Step time: {step_time:.3f} s\n"
        stats_placeholder.text(stats_text)

        time.sleep(refresh_interval)


if __name__ == "__main__":
    main()

