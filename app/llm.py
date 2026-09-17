import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def ask_llm(prompt: str) -> str:
    """
    Send a prompt to the locally running Ollama model
    and return the generated response.
    """

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    if not prompt.strip():
        raise ValueError("prompt cannot be empty")

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    if "response" not in data:
        raise ValueError(
            "Ollama response does not contain 'response'"
        )

    return data["response"]