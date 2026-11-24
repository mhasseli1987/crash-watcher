"""سنجش آفلاین مدل از روی همان لاگ‌هایی که برنامه می‌نویسد.

منبع داده:
  * ``logs/rounds.jsonl``  — هر دور با شناسه و رنگ واقعی (خروجی کلیکر)
  * ``logs/turnbal-review.jsonl`` یا ``logs/smart-review.jsonl`` — پیشنهاد مدل

برای هر پیشنهاد، رنگِ دورِ *بعدی* نتیجهٔ واقعی است. خروجی:
  * دقت و بازهٔ اطمینان ویلسون ۹۵٪
  * Brier score (میانگین (احتمال − نتیجه)^2؛ کمتر بهتر)
  * دقت walk-forward (میانگین تجمعی به ترتیب زمان، بدون نگاه به آینده)
  * تفکیک بر اساس رنگ و مدل

اجرا:
  python tools/evaluate.py
  python tools/evaluate.py --reviews logs/smart-review.jsonl --json logs/evaluation.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    except FileNotFoundError:
        print(f"هشدار: {path} وجود ندارد", file=sys.stderr)
    return rows


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """بازهٔ اطمینان ویلسون برای یک نسبت؛ در نمونهٔ کوچک از نرمال امن‌تر است."""
    if total <= 0:
        return 0.0, 1.0
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def walk_forward(hits: list[int], block: int = 10) -> list[dict]:
    """دقت تجمعی به ترتیب زمان؛ هر گام فقط از داده‌های گذشته استفاده می‌کند."""
    steps: list[dict] = []
    correct = 0
    for index, hit in enumerate(hits, 1):
        correct += hit
        if index % block == 0 or index == len(hits):
            steps.append({"n": index, "accuracy": correct / index})
    return steps


def pair_predictions(reviews: list[dict], rounds: list[dict]) -> list[dict]:
    """هر پیشنهاد را به رنگ دورِ بعدی وصل کن؛ پیشنهادهای بی‌نتیجه حذف می‌شوند."""
    order = [(str(row.get("round")), str(row.get("color"))) for row in rounds
             if row.get("round") and row.get("color")]
    position: dict[str, int] = {}
    for index, (round_id, _) in enumerate(order):
        position.setdefault(round_id, index)
    paired: list[dict] = []
    for row in reviews:
        if not row.get("ok", True) or row.get("late"):
            continue
        bet = row.get("bet")
        if bet not in ("RED", "GREEN", "YELLOW"):
            continue
        round_id = str(row.get("round"))
        index = position.get(round_id)
        if index is None or index + 1 >= len(order):
            continue
        outcome = order[index + 1][1]
        second = row.get("second")
        confidence = row.get("confidence")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        paired.append({
            "round": round_id, "outcome": outcome, "bet": bet, "second": second,
            "confidence": min(1.0, max(0.0, confidence)),
            "hit": outcome == bet,
            "covered": outcome == bet or outcome == second,
            "model": str(row.get("model") or row.get("chosen_model") or "—"),
        })
    return paired


def summarise(rows: list[dict]) -> dict:
    total = len(rows)
    hits = sum(1 for row in rows if row["hit"])
    covered = sum(1 for row in rows if row["covered"])
    brier = (sum((row["confidence"] - (1.0 if row["hit"] else 0.0)) ** 2 for row in rows) / total
             if total else None)
    low, high = wilson(hits, total)
    by_color: dict[str, dict] = {}
    for row in rows:
        bucket = by_color.setdefault(row["bet"], {"n": 0, "hits": 0})
        bucket["n"] += 1
        bucket["hits"] += 1 if row["hit"] else 0
    by_model: dict[str, dict] = {}
    for row in rows:
        bucket = by_model.setdefault(row["model"], {"n": 0, "hits": 0, "brier_sum": 0.0})
        bucket["n"] += 1
        bucket["hits"] += 1 if row["hit"] else 0
        bucket["brier_sum"] += (row["confidence"] - (1.0 if row["hit"] else 0.0)) ** 2
    for bucket in by_model.values():
        bucket["brier"] = bucket["brier_sum"] / bucket["n"] if bucket["n"] else None
    return {
        "n": total, "hits": hits,
        "accuracy": hits / total if total else None,
        "wilson_low": low, "wilson_high": high,
        "covered": covered,
        "break_even_rate": (covered - hits) / total if total else None,
        "brier": brier,
        "walk_forward": walk_forward([1 if row["hit"] else 0 for row in rows]),
        "by_color": by_color, "by_model": by_model,
    }


def report(summary: dict) -> str:
    lines = ["=== سنجش مدل Crash Watcher ==="]
    if not summary["n"]:
        lines.append("هیچ پیش‌بینیِ قابل‌سنجشی پیدا نشد (rounds.jsonl و لاگ بررسی را چک کنید).")
        return "\n".join(lines)
    lines.append(f"پیش‌بینی سنجیده‌شده: {summary['n']} | درست: {summary['hits']} "
                 f"({summary['accuracy']:.1%})")
    lines.append(f"بازهٔ ویلسون ۹۵٪: {summary['wilson_low']:.1%} تا {summary['wilson_high']:.1%}")
    lines.append(f"سربه‌سر با پوشش دو رنگ: {summary['covered'] - summary['hits']} "
                 f"({summary['break_even_rate']:.1%})")
    lines.append(f"Brier: {summary['brier']:.3f}  (کمتر بهتر؛ ۰٫۲۵ = پیش‌بینی تصادفی ۵۰٪)")
    steps = summary["walk_forward"]
    if steps:
        lines.append("walk-forward (میانگین تجمعی): " +
                     "، ".join(f"{step['n']}:{step['accuracy']:.0%}" for step in steps[-5:]))
    lines.append("به تفکیک رنگ:")
    for color, bucket in sorted(summary["by_color"].items()):
        rate = bucket["hits"] / bucket["n"] if bucket["n"] else 0
        lines.append(f"  {color}: {bucket['hits']}/{bucket['n']} ({rate:.0%})")
    lines.append("به تفکیک مدل:")
    for model, bucket in sorted(summary["by_model"].items()):
        rate = bucket["hits"] / bucket["n"] if bucket["n"] else 0
        lines.append(f"  {model}: {bucket['hits']}/{bucket['n']} ({rate:.0%}) | "
                     f"Brier {bucket['brier']:.3f}")
    return "\n".join(lines)


def summarise_hybrid(predictions: list[dict], rounds: list[dict]) -> dict:
    """Score only frozen predictions against their recorded actual round."""
    actual = {str(row.get("round")): row.get("color") for row in rounds
              if row.get("round") and row.get("color")}
    seen = Counter()
    baseline = {}
    for row in rounds:
        rid, color = str(row.get("round")), row.get("color")
        if rid not in actual or color not in ("RED", "GREEN", "YELLOW"):
            continue
        baseline[rid] = max(seen, key=seen.get) if seen else None
        seen[color] += 1
    buckets = {name: {"n": 0, "hits": 0, "latency_ms": []}
               for name in ("ROUTER", "OLLAMA", "LOCAL_ENGINE", "HYBRID", "BASELINE")}
    fallback = 0
    matched = 0
    deadline_met = 0
    for row in predictions:
        rid = str(row.get("actual_round_id"))
        color = actual.get(rid)
        if color is None or color != row.get("actual_result") or row.get("status") != "SCORED":
            continue
        matched += 1
        deadline_met += bool(row.get("deadline_met"))
        source = row.get("source")
        if source not in ("ROUTER", "OLLAMA", "LOCAL_ENGINE"):
            continue
        fallback += source != "ROUTER"
        for name, bet in (("HYBRID", row.get("prediction")),
                          ("BASELINE", baseline.get(rid))):
            if bet in ("RED", "GREEN", "YELLOW"):
                buckets[name]["n"] += 1
                buckets[name]["hits"] += bet == color
        for name, value in (row.get("providers") or {}).items():
            if name not in buckets or not isinstance(value, dict):
                continue
            if value.get("prediction") not in ("RED", "GREEN", "YELLOW"):
                continue
            bucket = buckets[name]
            bucket["n"] += 1
            bucket["hits"] += value["prediction"] == color
            latency = row.get(name.lower() + "_latency_ms")
            if isinstance(latency, (int, float)) and math.isfinite(latency):
                bucket["latency_ms"].append(latency)
    for bucket in buckets.values():
        values = sorted(bucket.pop("latency_ms"))
        bucket["accuracy"] = bucket["hits"] / bucket["n"] if bucket["n"] else None
        bucket["mean_latency_ms"] = sum(values) / len(values) if values else None
        bucket["p95_latency_ms"] = values[math.ceil(.95 * len(values)) - 1] if values else None
    total = buckets["HYBRID"]["n"]
    return {"sources": buckets, "fallback_rate": fallback / total if total else None,
            "deadline_success_rate": deadline_met / matched if matched else None,
            "scored": total, "matched_rounds": matched}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="سنجش دقت و Brier مدل از لاگ‌ها")
    parser.add_argument("--reviews", type=Path, default=None,
                        help="لاگ پیشنهادهای مدل (پیش‌فرض: خودکار turnbal یا smart)")
    parser.add_argument("--rounds", type=Path, default=ROOT / "logs" / "rounds.jsonl",
                        help="لاگ دورها با رنگ واقعی")
    parser.add_argument("--json", type=Path, default=None,
                        help="نوشتن خلاصهٔ ماشین‌خوان در این مسیر")
    parser.add_argument("--hybrid", action="store_true", help="سنجش لاگ پیش‌بینی ترکیبی")
    args = parser.parse_args(argv)
    reviews_path = args.reviews
    if reviews_path is None:
        # هر دو حالت لاگ دارند؛ پیش‌فرض اولین فایل موجود را بخوان.
        candidates = (ROOT / "logs" / "turnbal-review.jsonl",
                      ROOT / "logs" / "smart-review.jsonl")
        reviews_path = next((path for path in candidates if path.exists()), candidates[0])
    rounds = read_jsonl(args.rounds)
    if args.hybrid:
        summary = summarise_hybrid(read_jsonl(ROOT / "logs" / "hybrid-predictions.jsonl"), rounds)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    reviews = read_jsonl(reviews_path)
    summary = summarise(pair_predictions(reviews, rounds))
    print(report(summary))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"خلاصه در {args.json} نوشته شد")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
