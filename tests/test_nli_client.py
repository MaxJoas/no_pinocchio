import pytest
from nopin.clients.nli import NLIClient


@pytest.fixture
def mock_nli_config(mocker):
    """Mock NLI config."""
    config = mocker.Mock()
    config.nli.model = "nli-deberta-v3-small"
    config.nli.device = -1
    config.nli.top_k = None
    return config


def test_nli_client_initializes_pipeline(mocker, mock_nli_config):
    """Test NLI client creates pipeline."""
    mock_pipeline = mocker.patch("nopin.clients.nli.pipeline")

    NLIClient(config=mock_nli_config)

    assert mock_pipeline.called


def test_nli_client_returns_predictions(mocker, mock_nli_config):
    """Test NLI client returns prediction results."""
    mock_predictions = [{"label": "entailment", "score": 0.8}]
    mock_pipeline = mocker.patch("nopin.clients.nli.pipeline")
    mock_pipeline.return_value.return_value = mock_predictions

    client = NLIClient(config=mock_nli_config)
    result = client("text1 [SEP] text2")

    assert result == mock_predictions


def test_nli_pipeline_uses_correct_model(mocker, mock_nli_config):
    """Test NLI pipeline uses correct model path."""
    mock_pipeline = mocker.patch("nopin.clients.nli.pipeline")

    NLIClient(config=mock_nli_config)

    mock_pipeline.assert_called_with(
        "text-classification",
        model="cross-encoder/nli-deberta-v3-small",
        device=-1,
        top_k=None,
    )
