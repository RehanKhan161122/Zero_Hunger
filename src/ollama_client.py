"""Local Ollama client for Qwen2.5 7B with health checking, timing, and error handling.

Strictly communicates with the local Ollama daemon only (no external or cloud APIs).
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "360"))
DEFAULT_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "250"))
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))


class OllamaClient:
    """Client for querying a local Ollama instance."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        num_predict: Optional[int] = None,
    ):
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
        self.num_predict = num_predict if num_predict is not None else DEFAULT_NUM_PREDICT

    def check_health(self) -> Tuple[bool, str, List[str]]:
        """Check if local Ollama server is running and inspect loaded models.
        
        Returns:
            Tuple of (is_connected: bool, status_message: str, model_names: List[str])
        """
        tags_url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(tags_url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                if self.model in models or any(self.model in m for m in models):
                    return (
                        True,
                        f"Connected to Ollama. Model '{self.model}' is available.",
                        models,
                    )
                else:
                    return (
                        True,
                        f"Connected to Ollama, but model '{self.model}' was not found in installed models. Run: `ollama run {self.model}`",
                        models,
                    )
            else:
                return (
                    False,
                    f"Ollama server returned HTTP {resp.status_code}: {resp.text}",
                    [],
                )
        except requests.exceptions.ConnectionError:
            return (
                False,
                f"Cannot connect to Ollama at {self.base_url}. Make sure the Ollama service is running. Start it with `ollama serve` or open the Ollama app.",
                [],
            )
        except requests.exceptions.Timeout:
            return (
                False,
                f"Connection to Ollama at {self.base_url} timed out.",
                [],
            )
        except Exception as e:
            return (
                False,
                f"Ollama health check error: {str(e)}",
                [],
            )

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        num_predict: Optional[int] = None,
        timeout: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send prompt to local Ollama /api/generate with stream=False.
        
        Returns structured dictionary with success status, response text, timing, and error details.
        """
        target_model = model or self.model
        generate_url = f"{self.base_url}/api/generate"
        effective_timeout = timeout if timeout is not None else self.timeout
        effective_num_predict = num_predict if num_predict is not None else self.num_predict

        gen_options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": effective_num_predict,
            "num_ctx": DEFAULT_NUM_CTX,
        }
        if options:
            gen_options.update(options)

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": gen_options,
        }

        if system_instruction:
            payload["system"] = system_instruction

        start_time = time.time()
        try:
            response = requests.post(
                generate_url,
                json=payload,
                timeout=effective_timeout,
            )
            duration_s = round(time.time() - start_time, 2)

            if response.status_code == 200:
                result_json = response.json()
                generated_text = result_json.get("response", "").strip()
                eval_count = result_json.get("eval_count")
                total_duration_ms = (
                    round(result_json.get("total_duration", 0) / 1e6, 2)
                    if "total_duration" in result_json
                    else None
                )

                return {
                    "success": True,
                    "response": generated_text,
                    "model": target_model,
                    "duration_seconds": duration_s,
                    "total_duration_ms": total_duration_ms,
                    "eval_count": eval_count,
                    "error": None,
                }
            elif response.status_code == 404:
                return {
                    "success": False,
                    "response": "",
                    "model": target_model,
                    "duration_seconds": duration_s,
                    "total_duration_ms": None,
                    "eval_count": None,
                    "error": (
                        f"Model '{target_model}' not found on local Ollama. "
                        f"Please pull it first by running: `ollama run {target_model}`."
                    ),
                }
            else:
                return {
                    "success": False,
                    "response": "",
                    "model": target_model,
                    "duration_seconds": duration_s,
                    "total_duration_ms": None,
                    "eval_count": None,
                    "error": f"Ollama HTTP {response.status_code} error: {response.text}",
                }

        except requests.exceptions.ConnectionError:
            duration_s = round(time.time() - start_time, 2)
            return {
                "success": False,
                "response": "",
                "model": target_model,
                "duration_seconds": duration_s,
                "total_duration_ms": None,
                "eval_count": None,
                "error": (
                    f"Could not connect to local Ollama at {self.base_url}. "
                    f"Please verify Ollama is running (`ollama serve`)."
                ),
            }
        except requests.exceptions.Timeout:
            duration_s = round(time.time() - start_time, 2)
            return {
                "success": False,
                "response": "",
                "model": target_model,
                "duration_seconds": duration_s,
                "total_duration_ms": None,
                "eval_count": None,
                "error": (
                    f"Request to Ollama timed out after {self.timeout} seconds. "
                    "The model might still be loading or generating a complex output."
                ),
            }
        except Exception as e:
            duration_s = round(time.time() - start_time, 2)
            return {
                "success": False,
                "response": "",
                "model": target_model,
                "duration_seconds": duration_s,
                "total_duration_ms": None,
                "eval_count": None,
                "error": f"Unexpected error during Ollama generation: {str(e)}",
            }
