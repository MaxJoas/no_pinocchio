import pytest
from nopin.clients.llm import LLMClient


@pytest.fixture
def mock_mistral_config(mocker):
    """Mock config for Mistral."""
    config = mocker.Mock()
    config.llm.client = "mistral"
    config.llm.model = "mistral-large"
    config.llm.temperature.get.return_value = mocker.Mock(min=0.1, max=0.9)
    config.api.mistral_api_key_env = "MISTRAL_API_KEY"
    return config


@pytest.fixture
def mock_mistral_config_invalid(mocker):
    """Mock config for Mistral."""
    config = mocker.Mock()
    config.llm.client = "mistral"
    config.llm.model = "mistral-large"
    config.llm.temperature.get.return_value = mocker.Mock(min=0.1, max=0.9)
    config.api.mistral_api_key_env = ""  # INVALID
    return config


@pytest.fixture
def mock_ollama_config(mocker):
    """Mock config for Ollama."""
    config = mocker.Mock()
    config.llm.client = "ollama"
    config.llm.model = "qwen2.5:0.5b"
    config.llm.temperature.get.return_value = mocker.Mock(min=0.1, max=0.9)
    return config


def test_mistral_chat_returns_response(mocker, mock_mistral_config):
    """Test Mistral client returns chat response."""
    mocker.patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    mock_mistral = mocker.patch("nopin.clients.llm.Mistral")

    mock_response = mocker.Mock()
    mock_response.choices = [mocker.Mock(message=mocker.Mock(content="Test response"))]
    mock_mistral.return_value.chat.complete.return_value = mock_response

    client = LLMClient(config=mock_mistral_config)
    result = client.chat(question="Test?", temperature="min")

    assert result == "Test response"


def test_invalid_temperature_raises_error(mocker, mock_mistral_config):
    """Test invalid temperature raises ValueError."""
    mocker.patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"})
    mocker.patch("nopin.clients.llm.Mistral")

    client = LLMClient(config=mock_mistral_config)

    with pytest.raises(ValueError):
        client.chat(question="Test?", temperature="medium")


def test_missing_api_key_raises_error(mocker, mock_mistral_config_invalid):
    """Test missing API key raises ValueError."""
    mocker.patch.dict("os.environ", {}, clear=True)

    with pytest.raises(ValueError):
        LLMClient(config=mock_mistral_config_invalid)
