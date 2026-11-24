"""Thread-safe next-round prediction arbitration."""
from dataclasses import dataclass
import math
from threading import Lock
from time import monotonic

from .smart_review import SmartVerdict


@dataclass(frozen=True)
class Prediction:
    target_round_id: str
    context_version: int
    source: str
    verdict: SmartVerdict
    created_at: float


class PredictionManager:
    """Keep one target active and reject callbacks for older contexts."""

    def __init__(self, clock=monotonic):
        self._clock = clock
        self._lock = Lock()
        self._target = None
        self._version = 0
        self._deadline = 0.0
        self._started = 0.0
        self._frozen = False
        self._answers = {}
        self._requests = set()

    def begin(self, target_round_id: str, deadline_s: float) -> int:
        if not target_round_id or deadline_s <= 0:
            raise ValueError("target and deadline are required")
        with self._lock:
            self._target = target_round_id
            self._version += 1
            self._deadline = self._clock() + deadline_s
            self._started = self._clock()
            self._frozen = False
            self._answers = {}
            self._requests = set()
            return self._version

    def reserve(self, target_round_id: str, version: int, source: str) -> bool:
        with self._lock:
            key = (version, source)
            if (self._frozen or target_round_id != self._target or version != self._version
                    or key in self._requests or self._clock() >= self._deadline):
                return False
            self._requests.add(key)
            return True

    def submit(self, target_round_id: str, version: int, source: str,
               verdict: SmartVerdict | None) -> bool:
        if source not in ("ROUTER", "OLLAMA", "LOCAL_ENGINE"):
            return False
        if (verdict is None or verdict.bet not in ("RED", "GREEN", "YELLOW", "SKIP")
                or not isinstance(verdict.confidence, (int, float))
                or not math.isfinite(verdict.confidence)
                or not 0 <= verdict.confidence <= 1):
            return False
        with self._lock:
            if (self._frozen or target_round_id != self._target or version != self._version
                    or self._clock() >= self._deadline
                    or (source in self._answers and source != "LOCAL_ENGINE")):
                return False
            self._answers[source] = Prediction(target_round_id, version, source,
                                               verdict, self._clock())
            return True

    def freeze(self, target_round_id: str) -> Prediction | None:
        with self._lock:
            if self._frozen or target_round_id != self._target:
                return None
            self._frozen = True
            for source in ("ROUTER", "OLLAMA", "LOCAL_ENGINE"):
                answer = self._answers.get(source)
                if answer is not None:
                    return answer
            return None

    def latencies_ms(self) -> dict[str, float | None]:
        with self._lock:
            return {source.lower() + "_latency_ms":
                    round((answer.created_at - self._started) * 1000, 2) if answer else None
                    for source in ("ROUTER", "OLLAMA", "LOCAL_ENGINE")
                    for answer in (self._answers.get(source),)}

    def clear_local(self, target_round_id: str, version: int) -> None:
        with self._lock:
            if not self._frozen and target_round_id == self._target and version == self._version:
                self._answers.pop("LOCAL_ENGINE", None)

    def answers(self) -> dict[str, Prediction]:
        with self._lock:
            return dict(self._answers)

    def elapsed_ms(self) -> float:
        with self._lock:
            return round((self._clock() - self._started) * 1000, 2)
