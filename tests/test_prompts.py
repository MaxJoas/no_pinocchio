from nopin.core.prompts import Prompts


def test_self_reflection_prompt_contains_question():
    """Test self-reflection prompt includes the question."""
    prompts = Prompts()
    result = prompts.self_reflection.render(question="What is 2+2?", answer="4")

    assert "What is 2+2?" in result


def test_self_reflection_prompt_contains_answer():
    """Test self-reflection prompt includes the answer."""
    prompts = Prompts()
    result = prompts.self_reflection.render(question="Test?", answer="Test answer")

    assert "Test answer" in result


def test_consistency_prompt_contains_question():
    """Test consistency prompt includes the question."""
    prompts = Prompts()
    result = prompts.consistency.render(question="What is the capital?")

    assert "What is the capital?" in result


def test_prompts_render_as_strings():
    """Test prompts render to string format."""
    prompts = Prompts()
    result = prompts.self_reflection.render(question="test", answer="test")

    assert isinstance(result, str)
