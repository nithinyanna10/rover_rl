from __future__ import annotations

import argparse
import os

import torch
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PPO policy to TorchScript.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to SB3 PPO model .zip",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output TorchScript file path (.ts or .pt)",
    )
    args = parser.parse_args()

    model = PPO.load(args.model_path)
    policy = model.policy

    # Dummy input: observation vector length is inferred from policy
    obs_dim = policy.observation_space.shape[0]  # type: ignore[arg-type]
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)

    scripted = torch.jit.trace(policy, dummy_obs)

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    scripted.save(args.output_path)
    print(f"Exported TorchScript policy to {args.output_path}")


if __name__ == "__main__":
    main()

