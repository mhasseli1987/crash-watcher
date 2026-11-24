from crash_watcher.desktop.prediction_manager import PredictionManager
from crash_watcher.desktop.smart_review import SmartVerdict
from concurrent.futures import ThreadPoolExecutor


def test_priority_freeze_and_late_answer():
    now = [0.0]
    manager = PredictionManager(clock=lambda: now[0])
    version = manager.begin("next-1", 4)
    assert manager.reserve("next-1", version, "ROUTER")
    assert not manager.reserve("next-1", version, "ROUTER")
    assert manager.submit("next-1", version, "LOCAL_ENGINE", SmartVerdict("RED", .5, ""))
    assert manager.submit("next-1", version, "OLLAMA", SmartVerdict("GREEN", .6, ""))
    assert manager.freeze("next-1").source == "OLLAMA"
    assert not manager.submit("next-1", version, "ROUTER", SmartVerdict("RED", .9, ""))
    assert manager.freeze("next-1") is None


def test_stale_context_and_deadline():
    now = [0.0]
    manager = PredictionManager(clock=lambda: now[0])
    old = manager.begin("next-1", 4)
    new = manager.begin("next-2", 4)
    verdict = SmartVerdict("RED", .5, "")
    assert not manager.submit("next-1", old, "ROUTER", verdict)
    assert not manager.submit("next-2", old, "ROUTER", verdict)
    now[0] = 5
    assert not manager.submit("next-2", new, "ROUTER", verdict)
    assert manager.freeze("next-2") is None


def test_confirmed_context_refreshes_local_fallback():
    manager = PredictionManager()
    version = manager.begin("next", 10)
    assert manager.submit("next", version, "LOCAL_ENGINE", SmartVerdict("RED", .5, ""))
    assert manager.submit("next", version, "LOCAL_ENGINE", SmartVerdict("GREEN", .6, ""))
    assert manager.freeze("next").verdict.bet == "GREEN"


def test_many_simultaneous_callbacks_cannot_change_frozen_round():
    manager = PredictionManager()
    version = manager.begin("round-7", 10)
    verdict = SmartVerdict("RED", .7, "")
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda _: manager.submit("round-7", version, "ROUTER", verdict),
                                range(50)))
    assert sum(results) == 1
    assert manager.freeze("round-7").verdict == verdict
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda _: manager.submit("round-7", version, "OLLAMA", verdict),
                                range(50)))
    assert not any(results)


def test_rapid_transitions_reject_all_previous_requests():
    manager = PredictionManager()
    versions = [manager.begin(f"round-{index}", 10) for index in range(20)]
    for index, version in enumerate(versions[:-1]):
        assert not manager.reserve(f"round-{index}", version, "ROUTER")
        assert not manager.submit(f"round-{index}", version, "ROUTER",
                                  SmartVerdict("RED", .6, ""))
    assert manager.freeze("round-19") is None
