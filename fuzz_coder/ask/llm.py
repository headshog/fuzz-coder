from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

import requests


def call_llm(
    prompt: str,
    model: str,
    ollama_url: str,
    temperature: float = 0.1,
    timeout: int = 600,
    stream: bool = False,
    token_callback: Optional[Callable[[str], None]] = None,
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
        payload_data = dict(
            model=model,
            prompt=prompt,
            stream=bool(stream),
            options=dict(temperature=temperature),
        )

        if stream:
            r = requests.post(
                ollama_url,
                json=payload_data,
                timeout=timeout,
                stream=True,
            )
            if hasattr(r, "raise_for_status"):
                r.raise_for_status()
            if not hasattr(r, "iter_lines"):
                payload = r.json()
                text = str(payload.get("response", "") or "")
                if token_callback is not None and text:
                    token_callback(text)
                if "response" not in payload:
                    return {
                        "ok": False,
                        "response": "",
                        "error": "LLM payload missing 'response' field",
                    }
                return {
                    "ok": True,
                    "response": text,
                    "error": None,
                }
            response_parts = []
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except Exception:
                    continue
                if chunk.get("error"):
                    return {
                        "ok": False,
                        "response": "".join(response_parts),
                        "error": str(chunk.get("error")),
                    }
                token = chunk.get("response", "")
                if token:
                    token = str(token)
                    response_parts.append(token)
                    if token_callback is not None:
                        token_callback(token)
                if chunk.get("done"):
                    break
            return {
                "ok": True,
                "response": "".join(response_parts),
                "error": None,
            }

        r = requests.post(
            ollama_url,
            json=payload_data,
            timeout=timeout,
        )
        if hasattr(r, "raise_for_status"):
            r.raise_for_status()
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
