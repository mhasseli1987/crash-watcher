from dataclasses import FrozenInstanceError
from decimal import Decimal
import pytest
from crash_watcher import StrategyConfig, StrategyEngine

@pytest.mark.parametrize("field", ["fixed_bet_amount","red_threshold","yellow_threshold"])
@pytest.mark.parametrize("value", [0,-1,True,"10",None,float("nan"),float("inf")])
def test_invalid_numbers(field,value):
    with pytest.raises(ValueError):
        StrategyConfig(**{field:value})

@pytest.mark.parametrize("value", [0,-1,True,1.5,"10",None])
def test_invalid_attempts(value):
    with pytest.raises(ValueError):
        StrategyConfig(max_attempts=value)

@pytest.mark.parametrize("red,yellow", [(10,10),(11,10)])
def test_threshold_order(red,yellow):
    with pytest.raises(ValueError):
        StrategyConfig(red_threshold=red,yellow_threshold=yellow)

def test_immutable_and_custom_config():
    cfg = StrategyConfig(fixed_bet_amount=Decimal("0.1"),max_attempts=2,red_threshold=3,yellow_threshold=5)
    with pytest.raises(FrozenInstanceError):
        cfg.fixed_bet_amount = 20
    e = StrategyEngine(cfg)
    with pytest.raises(AttributeError):
        e.config = StrategyConfig()
    assert e.process_round("t",5).cycle_triggered
    assert e.process_round("l1",2).bet_amount == Decimal("0.1")
    assert e.process_round("l2",4).cycle_outcome.value == "EXPIRED"

def test_invalid_config_object():
    with pytest.raises(ValueError):
        StrategyEngine(None)
