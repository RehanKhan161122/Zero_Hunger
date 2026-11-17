"""Unit tests for local Ollama client and error handling in src/ollama_client.py."""

from unittest.mock import MagicMock, patch
import requests

from src.ollama_client import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, OllamaClient


def test_ollama_client_init_defaults():
    client = OllamaClient()
    assert client.base_url == DEFAULT_OLLAMA_BASE_URL
    assert client.model == DEFAULT_OLLAMA_MODEL


def test_ollama_client_init_custom():
    client = OllamaClient(base_url="http://127.0.0.1:11434/", model="custom:model", timeout=60)
    assert client.base_url == "http://127.0.0.1:11434"
    assert client.model == "custom:model"
    assert client.timeout == 60


@patch("requests.get")
def test_check_health_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": "qwen2.5:7b"}, {"name": "llama3:latest"}]
    }
    mock_get.return_value = mock_resp

    client = OllamaClient(model="qwen2.5:7b")
    connected, msg, models = client.check_health()

    assert connected is True
    assert "available" in msg
    assert "qwen2.5:7b" in models


@patch("requests.get")
def test_check_health_connection_error(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

    client = OllamaClient()
    connected, msg, models = client.check_health()

    assert connected is False
    assert "Cannot connect to Ollama" in msg
    assert models == []


@patch("requests.post")
def test_generate_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "## 1. Waste Analysis & Severity\nWaste is low.",
        "eval_count": 85,
        "total_duration": 1500000000,
    }
    mock_post.return_value = mock_resp

    client = OllamaClient(model="qwen2.5:7b")
    result = client.generate("Analyze canteen waste")

    assert result["success"] is True
    assert "Waste is low" in result["response"]
    assert result["eval_count"] == 85
    assert result["error"] is None


@patch("requests.post")
def test_generate_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("Server offline")

    client = OllamaClient()
    result = client.generate("Analyze canteen waste")

    assert result["success"] is False
    assert "Could not connect to local Ollama" in result["error"]


@patch("requests.post")
def test_generate_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout("Timed out")

    client = OllamaClient()
    result = client.generate("Analyze canteen waste")

    assert result["success"] is False
    assert "timed out" in result["error"]


@patch("requests.post")
def test_generate_404_model_missing(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_post.return_value = mock_resp

    client = OllamaClient(model="nonexistent-model")
    result = client.generate("Analyze canteen waste")

    assert result["success"] is False
    assert "not found on local Ollama" in result["error"]
