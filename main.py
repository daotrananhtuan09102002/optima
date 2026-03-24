import argparse
from pathlib import Path

from instaoptima.config import ExperimentConfig
from instaoptima.runner import InstaOptimaExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--runs-per-launch",
        "-r",
        type=int,
        default=None,
        help=(
            "Number of runs to execute in this launch. Defaults to the remaining "
            "runs up to config.num_runs."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        "-a",
        type=Path,
        default=None,
        help=(
            "Existing artifact directory to append runs to. If omitted, a new "
            "artifact directory is created."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    experiment = InstaOptimaExperiment(config, artifact_root=args.artifact_root)
    experiment.run(num_runs=args.runs_per_launch)


if __name__ == "__main__":
    main()
