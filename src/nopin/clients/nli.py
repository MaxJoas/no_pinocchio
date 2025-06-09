from typing import Any, Dict, List
from transformers import pipeline

from nopin.config.settings import Config


class NLIClient:
    """Natural Language Inference client."""

    def __init__(self, *, config: Config):
        """Initialize NLI client.

        Args:
            config: Configuration object.
        """
        self._config = config
        self._model = config.nli.model
        self._pipeline = self._get_nli_pipeline(
            model=self._model, device=config.nli.device
        )

    def _get_nli_pipeline(self, *, model: str, device: int):
        """Create NLI pipeline.

        Args:
            model: Model name.
            device: Device ID.
            top_k: Number of top predictions to return.

        Returns:
            Transformers pipeline object.
        """
        return pipeline(
            "text-classification",
            model=f"cross-encoder/{model}",
            device=device,
            top_k=3,
        )

    def __call__(self, text_pair: str) -> List[Dict[str, Any]]:
        """Perform natural language inference.

        Args:
            text_pair: Text pair separated by [SEP].

        Returns:
            List of dictionaries with labels and scores.
        """
        return self._pipeline(text_pair)
