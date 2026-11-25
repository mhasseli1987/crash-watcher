"""Optional local Ollama provider. Call from a background worker, never the UI thread."""
from __future__ import annotations

import json
import math
from urllib import error, request

from .config import DesktopSettings
from .smart_review import SmartVerdict


class OllamaError(RuntimeError):
    """A local service, model, or response failure."""


def _get_json(url: str, timeout: float) -> dict:
    try:
        with request.urlopen(url, timeout=timeout) as response:
            value = json.load(response)
    except (OSError, ValueError, error.URLError) as exc:
        raise OllamaError("Ollama is unavailable or returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise OllamaError("Ollama returned an invalid response")
    return value


def model_available(settings: DesktopSettings) -> bool:
    """Return whether the configured model is installed; never download it."""
    return settings.ollama_model in installed_models(settings)


def installed_models(settings: DesktopSettings) -> list[str]:
    """Discover installed models from the local service only."""
    settings.validate()
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    data = _get_json(url, min(settings.ollama_timeout_ms / 1000, 3.0))
    models = data.get("models")
    if not isinstance(models, list):
        raise OllamaError("Ollama model list is invalid")
    return [item["name"] for item in models
            if isinstance(item, dict) and isinstance(item.get("name"), str)]


def _parse_verdict(content: str, allowed_bets=("RED", "GREEN", "YELLOW", "SKIP")) -> SmartVerdict:
    try:
        data = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise OllamaError("Ollama prediction is not JSON") from exc
    if not isinstance(data, dict) or data.get("bet") not in allowed_bets:
        raise OllamaError("Ollama prediction has an invalid bet")
    confidence = data.get("confidence")
    if (type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1):
        raise OllamaError("Ollama prediction has invalid confidence")
    second = data.get("second")
    if second is not None and (second not in ("RED", "GREEN", "YELLOW")
                               or second == data["bet"] or data["bet"] == "SKIP"):
        raise OllamaError("Ollama prediction has invalid second bet")
    cashout = data.get("cashout")
    if cashout is not None and (type(cashout) not in (int, float)
                               or not math.isfinite(cashout) or not 1.4 <= cashout <= 1.9):
        raise OllamaError("Ollama prediction has invalid cashout")
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        raise OllamaError("Ollama prediction has invalid reason")
    return SmartVerdict(data["bet"], float(confidence), reason, second,
                        float(cashout) if cashout is not None else None)


def predict(settings: DesktopSettings, prompt: str) -> SmartVerdict:
    """Request one compact JSON verdict from an installed local model."""
    settings.validate()
    if not settings.ollama_enabled and settings.ai_provider != "local":
        raise OllamaError("Ollama is disabled")
    if not model_available(settings):
        raise OllamaError(f"Ollama model {settings.ollama_model} is not installed")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty text")
    allowed_bets = ("PLAY", "SKIP") if settings.bot_mode == "classic" else ("RED", "GREEN", "YELLOW", "SKIP")
    payload = {"model": settings.ollama_model, "stream": False, "think": False,
               "keep_alive": settings.ollama_keep_alive,
               "format": {"type": "object", "properties": {
                   "bet": {"type": "string", "enum": list(allowed_bets)},
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                   "reason": {"type": "string"}},
                   "required": ["bet", "confidence", "reason"], "additionalProperties": False},
               # The reason field can run long even under the JSON schema; a
               # budget of 80 tokens cut the JSON mid-string and every such
               # round ended with "prediction is not JSON" instead of a verdict.
               "options": {"temperature": 0.2, "num_predict": 260},
               "messages": [{"role": "system", "content":
                             "Return only the required JSON object. Reason at most 8 words. "
                             "If evidence is insufficient choose SKIP. Do not invent certainty."},
                            {"role": "user", "content": prompt}]}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    call = request.Request(settings.ollama_base_url.rstrip("/") + "/api/chat",
                           data=body, headers={"Content-Type": "application/json"},
                           method="POST")
    try:
        with request.urlopen(call, timeout=settings.ollama_timeout_ms / 1000) as response:
            data = json.load(response)
    except (OSError, ValueError, error.URLError) as exc:
        raise OllamaError("Ollama request failed or timed out") from exc
    try:
        content = data["message"]["content"]
    except (TypeError, KeyError) as exc:
        raise OllamaError("Ollama response has no message") from exc
    return _parse_verdict(content, allowed_bets)
