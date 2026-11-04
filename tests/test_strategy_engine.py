from decimal import Decimal
import pytest
from crash_watcher import StrategyConfig, StrategyEngine
from crash_watcher.core.enums import BetOutcome, CycleOutcome, StrategyState, EventType

@pytest.fixture
def engine():
    return StrategyEngine(StrategyConfig(fixed_bet_amount=7))

def test_initial(engine):
    assert engine.state == StrategyState.WAITING_FOR_YELLOW
    assert engine.attempt_number == engine.remaining_attempts == 0

@pytest.mark.parametrize("value,category", [(1.99,"RED"),(2,"GREEN"),(9.99,"GREEN"),(10,"YELLOW")])
def test_categories_and_trigger(engine,value,category):
    r = engine.process_round("a",value)
    assert r.category.value == category
    assert r.cycle_triggered == (value >= 10)
    assert not r.bet_registered
    assert r.bet_amount is None
    assert r.remaining_attempts == (10 if value >= 10 else 0)
    assert r.attempt_number == 0
    assert r.previous_state == StrategyState.WAITING_FOR_YELLOW

@pytest.mark.parametrize("attempt", range(1,11))
def test_win_on_each_attempt(engine,attempt):
    engine.process_round("trigger",10)
    for i in range(1,attempt):
        r = engine.process_round(str(i),1.5)
        assert r.bet_outcome == BetOutcome.LOST
        assert r.attempt_number == i
        assert r.remaining_attempts == 10-i
        assert r.bet_amount == Decimal(7)
    r = engine.process_round("winner",10)
    assert r.bet_registered and r.bet_outcome == BetOutcome.WON
    assert r.cycle_outcome == CycleOutcome.WON
    assert r.attempt_number == attempt
    assert r.remaining_attempts == engine.attempt_number == engine.remaining_attempts == 0
    assert engine.state == StrategyState.WAITING_FOR_YELLOW
    assert not r.cycle_triggered
    assert not engine.process_round("after",2).bet_registered
    assert engine.process_round("new",11).cycle_triggered

@pytest.mark.parametrize("value", [1.1,2,9.99])
def test_first_loss(engine,value):
    engine.process_round("trigger",12)
    r = engine.process_round("loss",value)
    assert r.bet_outcome == BetOutcome.LOST
    assert r.attempt_number == 1 and r.remaining_attempts == 9

def test_expiry_and_new_cycle(engine):
    engine.process_round("trigger",10)
    for i in range(10):
        r = engine.process_round(str(i),2)
        assert r.bet_amount == Decimal(7)
    assert r.cycle_outcome == CycleOutcome.EXPIRED
    assert r.attempt_number == 10 and r.remaining_attempts == 0
    assert engine.state == StrategyState.WAITING_FOR_YELLOW
    assert not engine.process_round("idle",1).bet_registered
    assert engine.process_round("new",10).cycle_triggered

def test_multiple_cycles(engine):
    for i in range(5):
        trigger = engine.process_round(f"t{i}",10)
        assert trigger.cycle_id == i+1 and trigger.cycle_triggered
        win = engine.process_round(f"w{i}",11)
        assert win.cycle_outcome == CycleOutcome.WON and win.attempt_number == 1

def test_event_order(engine):
    r = engine.process_round("t",10)
    assert [e.event_type for e in r.events] == [EventType.YELLOW_TRIGGER_DETECTED,EventType.CYCLE_STARTED]
    r = engine.process_round("w",10)
    assert [e.event_type for e in r.events] == [EventType.SIMULATED_BET_REGISTERED,EventType.SIMULATED_BET_WON,EventType.CYCLE_WON,EventType.CYCLE_RESET]
    assert all(e.timestamp == r.timestamp and e.attempt_number == 1 for e in r.events)


def test_expiry_event_order(engine):
    engine.process_round("t",10)
    for i in range(10):
        r = engine.process_round(str(i),1)
    assert [e.event_type for e in r.events] == [EventType.SIMULATED_BET_REGISTERED,EventType.SIMULATED_BET_LOST,EventType.CYCLE_EXPIRED,EventType.CYCLE_RESET]


def test_mixed_cycles_and_reset(engine):
    engine.process_round("t",10)
    engine.process_round("loss",2)
    engine.reset()
    assert not engine.process_round("idle",1).bet_registered
    r = engine.process_round("new",10)
    assert r.attempt_number == 0 and r.remaining_attempts == 10
    assert engine.process_round("winner",10).attempt_number == 1
