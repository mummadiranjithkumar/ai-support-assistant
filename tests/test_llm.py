from unittest.mock import Mock, patch

import pytest

from app.llm import ask_llm


def test_ask_llm_success():
    mock_response = Mock()

    mock_response.json.return_value = {
        "response": "A support ticket is a customer request for help."
    }

    mock_response.raise_for_status.return_value = None

    with patch(
        "app.llm.requests.post",
        return_value=mock_response
    ) as mock_post:

        result = ask_llm(
            "What is a support ticket?"
        )

    assert result == (
        "A support ticket is a customer request for help."
    )

    mock_post.assert_called_once()


def test_ask_llm_sends_correct_payload():
    mock_response = Mock()

    mock_response.json.return_value = {
        "response": "Test response"
    }

    mock_response.raise_for_status.return_value = None

    with patch(
        "app.llm.requests.post",
        return_value=mock_response
    ) as mock_post:

        ask_llm("Hello Ollama")

    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": "Hello Ollama",
            "stream": False
        },
        timeout=120
    )


def test_ask_llm_rejects_non_string_prompt():
    with pytest.raises(TypeError):
        ask_llm(123)


def test_ask_llm_rejects_empty_prompt():
    with pytest.raises(ValueError):
        ask_llm("")


def test_ask_llm_missing_response_field():
    mock_response = Mock()

    mock_response.json.return_value = {}

    mock_response.raise_for_status.return_value = None

    with patch(
        "app.llm.requests.post",
        return_value=mock_response
    ):

        with pytest.raises(ValueError):
            ask_llm("Hello")