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


def test_local_answer_is_ready_while_router_is_still_waiting(tmp_path, monkeypatch):
    from threading import Event, Thread
    from crash_watcher import StrategyEngine
    from crash_watcher.desktop.config import DesktopSettings, Region
    from crash_watcher.desktop import worker as module
    from crash_watcher.desktop.smart_review import SmartSnapshot
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    settings = DesktopSettings(ollama_enabled=True, bot_mode="turnbal", result_region=Region(1, 0, 0, 300, 120))
    worker = module.MonitorWorker(settings, StrategyEngine(settings.strategy()))
    started, release, local_ready = Event(), Event(), Event()
    def remote(*args):
        started.set()
        assert release.wait(3)
        return SmartVerdict("RED", .6, "remote"), "", ""
    def local(*args):
        local_ready.set()
        return SmartVerdict("GREEN", .6, "local")
    monkeypatch.setattr(worker, "_ask", remote)
    monkeypatch.setattr(module, "predict_ollama", local)
    version = worker._prediction.begin("next", 10)
    thread = Thread(target=worker._hybrid_ai, args=(
        "next", version, SmartSnapshot(recent=("1.5",), current="", rounds_seen=1), settings))
    thread.start()
    try:
        assert started.wait(2) and local_ready.wait(2)
        # The provider signal is set just before submission; wait for its lock to release.
        with worker._provider_locks["OLLAMA"]:
            assert worker._prediction.freeze("next").source == "OLLAMA"
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()


def test_classic_router_uses_color_context_but_rejects_color_actions(tmp_path, monkeypatch):
    from crash_watcher import StrategyEngine
    from crash_watcher.desktop.config import DesktopSettings, Region
    from crash_watcher.desktop import worker as module
    from crash_watcher.desktop.smart_review import SmartSnapshot
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    settings = DesktopSettings(bot_mode="classic", result_region=Region(1, 0, 0, 300, 120),
                               smart_review_rules="bet on YELLOW")
    worker = module.MonitorWorker(settings, StrategyEngine(settings.strategy()))
    prompts = []
    monkeypatch.setattr(module, "ask", lambda *args, **kwargs:
                        prompts.append(args[3]) or '{"bet":"RED","confidence":0.9}')
    verdict, _, _ = worker._ask("test", SmartSnapshot(recent=("1.20",)), settings, False, None, 1)
    assert verdict is None
    assert "1.20=RED" in prompts[0]
    assert "bet on YELLOW" not in prompts[0]
