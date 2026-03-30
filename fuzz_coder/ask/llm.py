from __future__ import annotations

from typing import Any, Dict, Optional

import requests


def call_llm(
    prompt: str,
    model: str,
    ollama_url: str,
    temperature: float = 0.1,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Call LLM and return structured status.

    Returns:
        {
          "ok": bool,
          "response": str,
          "error": Optional[str]
        }
    """
    try:
        r = requests.post(
            ollama_url,
            json=dict(
                model=model,
                prompt=prompt,
                stream=False,
                options=dict(temperature=temperature),
            ),
            timeout=timeout,
        )
        payload = r.json()
        if "response" not in payload:
            return {
                "ok": False,
                "response": "",
                "error": "LLM payload missing 'response' field",
            }
        return {
            "ok": True,
            "response": payload["response"],
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "response": "",
            "error": str(e),
        }
