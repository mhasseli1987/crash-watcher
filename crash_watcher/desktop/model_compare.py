"""مقایسهٔ موازی چند مدل روی یک اسنپ‌شات و انتخاب خودکار بهترین.

مدل «auto» یک مدل ناشناخته است؛ وقتی پرووایدر چند مدل دارد بهتر است همان
وضعیت به چند مدل داده شود و بر اساس خطای واقعیِ گذشته انتخاب شود. برای هر
مدل Brier score نگه می‌داریم: میانگین (احتمال اعلامی − نتیجه)^2 روی دست‌های
سنجیده‌شده. هرچه کمتر بهتر. تا وقتی هیچ مدلی حداقلِ نمونه را ندارد، انتخاب
بر اساس اطمینان پاسخ است (cold start).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json

MIN_SAMPLES = 5


@dataclass(frozen=True)
class ModelAttempt:
    """پاسخ یک مدل به همان اسنپ‌شات."""

    model: str
    verdict: object | None = None  # SmartVerdict
    raw: str = ""
    prompt: str = ""
    elapsed: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is not None


@dataclass
class ModelScore:
    tries: int = 0
    brier_sum: float = 0.0
    hits: int = 0

    @property
    def brier(self) -> float | None:
        return self.brier_sum / self.tries if self.tries else None

    def as_dict(self) -> dict:
        return {"tries": self.tries, "brier": self.brier, "hits": self.hits,
                "accuracy": self.hits / self.tries if self.tries else None}


class ModelLeaderboard:
    """Brier هر مدل، ذخیره‌شده بین جلسه‌ها."""

    def __init__(self, path: Path | None = None, min_samples: int = MIN_SAMPLES) -> None:
        self.path = path
        self.min_samples = max(1, int(min_samples))
        self.scores: dict[str, ModelScore] = {}
        if path is not None:
            self.load()

    # -- persistence ---------------------------------------------------
    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        for model, item in data.items():
            if not isinstance(item, dict):
                continue
            try:
                tries = int(item.get("tries", 0))
                brier_sum = float(item.get("brier_sum", 0.0))
                hits = int(item.get("hits", 0))
            except (TypeError, ValueError):
                continue
            if tries > 0:
                self.scores[str(model)] = ModelScore(tries, brier_sum, hits)

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {model: {"tries": score.tries, "brier_sum": score.brier_sum,
                            "hits": score.hits} for model, score in self.scores.items()}
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temporary.replace(self.path)
        except Exception:
            pass

    # -- scoring -------------------------------------------------------
    def record(self, model: str, probability: float, hit: bool) -> None:
        """یک پیش‌بینی سنجیده‌شده را به سابقهٔ همان مدل اضافه کن."""
        score = self.scores.setdefault(model, ModelScore())
        probability = min(1.0, max(0.0, float(probability)))
        score.tries += 1
        score.brier_sum += (probability - (1.0 if hit else 0.0)) ** 2
        score.hits += 1 if hit else 0

    def best(self, models) -> str | None:
        """کمترین Brier بین مدل‌هایی که نمونهٔ کافی دارند."""
        chosen = None
        for model in models:
            score = self.scores.get(str(model))
            if score is None or score.tries < self.min_samples:
                continue
            if chosen is None or score.brier < self.scores[chosen].brier:
                chosen = str(model)
        return chosen

    def summary(self) -> dict:
        return {model: score.as_dict() for model, score in sorted(self.scores.items())}

    # -- selection -----------------------------------------------------
    def choose(self, attempts) -> tuple[ModelAttempt | None, str]:
        """از میان پاسخ‌های موازی، بهترین مدل را انتخاب کن."""
        valid = [attempt for attempt in attempts if attempt.ok]
        if not valid:
            return None, "هیچ مدلی پاسخ قابل خواندن نداد"
        best = self.best([attempt.model for attempt in valid])
        if best is not None:
            chosen = next(attempt for attempt in valid if attempt.model == best)
            score = self.scores[best].brier
            return chosen, f"کمترین Brier: {best} ({score:.3f})"
        chosen = max(valid, key=lambda attempt: (attempt.verdict.confidence, -attempt.elapsed))
        return chosen, f"شروع سرد: بیشترین اطمینان ({chosen.model})"


def parse_models(text: str) -> list[str]:
    """فهرست مدل‌ها را از یک رشتهٔ جداشده با کاما/ویرگول بساز."""
    seen: list[str] = []
    for chunk in (text or "").replace("،", ",").split(","):
        name = chunk.strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def compare(models, ask, max_workers: int | None = None) -> list[ModelAttempt]:
    """همان اسنپ‌شات را موازی به همهٔ مدل‌ها بده.

    `ask(model)` باید (verdict, raw, prompt) برگرداند. خطای هر مدل جدا نگه
    داشته می‌شود تا یک مدل خراب بقیه را از کار نیندازد.
    """
    models = list(models)
    if not models:
        return []

    def run(model: str) -> ModelAttempt:
        started = perf_counter()
        try:
            verdict, raw, prompt = ask(model)
        except Exception as exc:  # noqa: BLE001 - one model must not sink the rest
            return ModelAttempt(model, None, "", "", perf_counter() - started, str(exc))
        return ModelAttempt(model, verdict, raw, prompt, perf_counter() - started)

    with ThreadPoolExecutor(max_workers=max_workers or min(len(models), 4)) as pool:
        return list(pool.map(run, models))
