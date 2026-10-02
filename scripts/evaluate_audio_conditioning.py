"""Run the controlled provider-versus-Qwen-prior evaluation."""

from __future__ import annotations

import argparse

from src.evaluation import run_audio_conditioning_evaluation
from src.utils.config_loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate audio conditioning")
    parser.add_argument("config", help="Path to the evaluation YAML")
    arguments = parser.parse_args()
    paths = run_audio_conditioning_evaluation(load_config(arguments.config))
    print(f"summary: {paths[0]}")
    print(f"samples: {paths[1]}")


if __name__ == "__main__":
    main()

