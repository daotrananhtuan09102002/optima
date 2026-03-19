import argparse

from instaoptima.config import ExperimentConfig
from instaoptima.runner import InstaOptimaExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    experiment = InstaOptimaExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
