import argparse
import logging

from nopin.config.settings import load_config
from nopin.core.nopin import NoPinocchio


def setup_logging(*, config):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level), format=config.logging.format
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="NoPinocchio Confidence Estimation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--question", type=str, required=True, help="Question to analyze"
    )

    args = parser.parse_args()

    config = load_config(config_path=args.config)
    setup_logging(config=config)

    logger = logging.getLogger(__name__)
    logger.info(f"Loaded configuration from {args.config}")
    np_system = NoPinocchio.from_config(config=config)
    logger.info(f"Analyzing question: {args.question}")

    result = np_system.analyze_question(question=args.question)

    print(f"Question: {args.question}")
    print(f"Answer: {result["answer"]}")
    print(f"Confidence Score: {result['confidence_score']:.3f}")


if __name__ == "__main__":
    main()
