"""Run from the project root: python -m examples.strategy_simulation."""
from crash_watcher import StrategyConfig, StrategyEngine

def main() -> None:
    engine = StrategyEngine(StrategyConfig(fixed_bet_amount=10))
    values = [1.2, 3, 12, 1.5, 4, 11, 10] + [1.5] * 10 + [12, 10]
    print("round | multiplier | category | state | attempt | remaining | bet | outcome | cycle")
    for index, value in enumerate(values, 1):
        r = engine.process_round(f"round_{index:03}", value)
        print(f"{index:02} | {r.multiplier} | {r.category.value} | {r.current_state.value} | "
              f"{r.attempt_number} | {r.remaining_attempts} | {r.bet_amount} | "
              f"{r.bet_outcome.value} | {r.cycle_outcome.value}")

if __name__ == "__main__":
    main()
