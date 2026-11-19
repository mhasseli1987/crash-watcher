from datetime import datetime, timezone
import pytest
from crash_watcher import StrategyEngine, StrategyConfig
from crash_watcher.core.enums import StrategyState, EventType

@pytest.mark.parametrize("value", [None,"10",True,False,-1,0,float("nan"),float("inf"),float("-inf"),[],complex(1,2)])
def test_invalid_multiplier_atomic(value):
    e = StrategyEngine(StrategyConfig())
    e.process_round("t",10)
    with pytest.raises(ValueError):
        e.process_round("x",value)
    assert e.attempt_number == 0 and e.remaining_attempts == 10
    assert e.process_round("x",1).bet_registered

@pytest.mark.parametrize("rid", [None,"","  ",1,False,[]])
def test_invalid_ids(rid):
    e = StrategyEngine(StrategyConfig())
    with pytest.raises(ValueError):
        e.process_round(rid,10)
    assert e.state == StrategyState.WAITING_FOR_YELLOW

def test_duplicate_and_reset():
    e = StrategyEngine(StrategyConfig())
    e.process_round(" t ",10)
    e.process_round("loss",1)
    for rid,value in [("t",15),("loss",10)]:
        r = e.process_round(rid,value)
        assert r.duplicate and not r.bet_registered and not r.cycle_triggered
        assert r.events[0].event_type == EventType.DUPLICATE_ROUND_IGNORED
        assert e.attempt_number == 1 and e.remaining_attempts == 9
    assert e.reset().event_type == EventType.CYCLE_RESET
    assert e.state == StrategyState.WAITING_FOR_YELLOW
    assert e.attempt_number == e.remaining_attempts == 0
    assert e.process_round("t",10).duplicate
    assert e.process_round("new",10).cycle_triggered

def test_duplicate_winner_not_trigger():
    e = StrategyEngine(StrategyConfig())
    e.process_round("t",10)
    e.process_round("w",10)
    assert e.process_round("w",10).duplicate
    assert e.state == StrategyState.WAITING_FOR_YELLOW

def test_deterministic_clock():
    now = datetime(2026,1,1,tzinfo=timezone.utc)
    a = StrategyEngine(StrategyConfig(),clock=lambda: now)
    b = StrategyEngine(StrategyConfig(),clock=lambda: now)
    for i,m in enumerate([1,10,2,10,10]+[2]*10):
        assert a.process_round(str(i),m) == b.process_round(str(i),m)
