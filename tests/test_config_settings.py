import pytest
from nopin.config.settings import Config, LLMConfig


def test_default_config_has_correct_client():
    """Test default config uses mistral client."""
    config = Config()
    assert config.llm.client == "mistral"


def test_default_config_has_correct_model():
    """Test default config uses correct model."""
    config = Config()
    assert config.llm.model == "mistral-medium-latest"


def test_invalid_client_raises_error():
    """Test invalid client raises ValueError."""
    with pytest.raises(ValueError):
        LLMConfig(client="invalid")


def test_invalid_mistral_model_raises_error():
    """Test invalid Mistral model raises ValueError."""
    with pytest.raises(ValueError):
        LLMConfig(client="mistral", model="invalid-model")
