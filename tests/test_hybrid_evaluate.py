from tools.evaluate import summarise_hybrid


def test_exact_round_pairing_and_chronological_baseline():
    rounds = [{"round": "r1", "color": "RED"},
              {"round": "r2", "color": "GREEN"},
              {"round": "r3", "color": "RED"}]
    predictions = [{"status": "SCORED", "actual_round_id": "r2",
                    "actual_result": "GREEN", "source": "OLLAMA",
                    "prediction": "GREEN", "providers": {"OLLAMA": {"prediction": "GREEN"}},
                    "ollama_latency_ms": 200},
                   {"status": "SCORED", "actual_round_id": "r3",
                    "actual_result": "GREEN", "source": "ROUTER", "prediction": "GREEN"}]
    report = summarise_hybrid(predictions, rounds)
    assert report["scored"] == 1
    assert report["sources"]["HYBRID"]["accuracy"] == 1
    assert report["sources"]["BASELINE"]["accuracy"] == 0
    assert report["fallback_rate"] == 1
