import pytest
from nopin.core.nopin import NoPinocchio


pytest_plugins = ["pytest_mock"]


@pytest.fixture
def dummy_question():
    """Dummy question for testing."""
    return "What is 2+2?"


@pytest.fixture
def dummy_answer():
    """Dummy answer for testing."""
    return "4"


@pytest.fixture
def mock_nopin_components(mocker):
    """Mock all NoPinocchio components."""
    mock_prompts = mocker.Mock()
    mock_llm = mocker.Mock()
    mock_nli = mocker.Mock()

    return {"prompts": mock_prompts, "llm_client": mock_llm, "nli_client": mock_nli}


def test_analyze_question_returns_dict(mock_nopin_components):
    """Test analyze_question returns dictionary."""
    # Setup mocks for a simple flow
    mock_nopin_components["llm_client"].chat.side_effect = ["Answer", "Sample", "A"]
    mock_nopin_components["nli_client"].return_value = [
        [{"label": "contradiction", "score": 0.1}]
    ]

    nopin = NoPinocchio(k=1, alpha=0.5, beta=0.5, **mock_nopin_components)
    result = nopin.analyze_question(question="Test?")

    assert isinstance(result, dict)


def test_analyze_question_contains_question(mock_nopin_components):
    """Test result contains the original question."""
    mock_nopin_components["llm_client"].chat.side_effect = ["Answer", "Sample", "A"]
    mock_nopin_components["nli_client"].return_value = [
        [{"label": "contradiction", "score": 0.1}]
    ]

    nopin = NoPinocchio(k=1, alpha=0.5, beta=0.5, **mock_nopin_components)
    result = nopin.analyze_question(question="Test question")

    assert result["question"] == "Test question"


def test_confidence_score_in_valid_range(mock_nopin_components):
    """Test confidence score is between 0 and 1."""
    mock_nopin_components["llm_client"].chat.side_effect = ["Answer", "Sample", "A"]
    mock_nopin_components["nli_client"].return_value = [
        [{"label": "contradiction", "score": 0.1}]
    ]

    nopin = NoPinocchio(k=1, alpha=0.5, beta=0.5, **mock_nopin_components)

    result = nopin.analyze_question(question="Test question")
    confidence = result["confidence_score"]

    assert 0.0 <= confidence <= 1.0


def test_from_config_creates_instance(mocker):
    """Test from_config creates NoPinocchio instance."""
    mock_config = mocker.Mock()
    mock_config.nopinocchio.k = 3
    mock_config.nopinocchio.alpha = 0.6
    mock_config.nopinocchio.beta = 0.7

    mocker.patch("nopin.core.nopin.Prompts")
    mocker.patch("nopin.core.nopin.LLMClient")
    mocker.patch("nopin.core.nopin.NLIClient")

    instance = NoPinocchio.from_config(config=mock_config)

    assert instance._k == 3
