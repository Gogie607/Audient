"""Official command-line entry point for a complete training YAML."""

from __future__ import annotations

import argparse

from src.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a V2 training session")
    parser.add_argument("config", help="Path to the complete training YAML")
    arguments = parser.parse_args()
    run_training(arguments.config)


if __name__ == "__main__":
    main()
