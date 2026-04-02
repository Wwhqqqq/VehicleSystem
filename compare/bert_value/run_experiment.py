from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.bert_value.io_util import default_multiturn_path, load_multiturn_sessions, load_reject_test_rows
from benchmarks.bert_value.metrics import StrategyRunSummary, latency_mean_p95
from benchmarks.bert_value.strategies import (
    BertOnlyRejectStrategy,
    BertPrescreenLlmDeepRejectStrategy,
    FullLlmRejectStrategy,
    RejectStrategy,
    default_llm_profile,
)
from benchmarks.common import result_path
from benchmarks.bert_value import visualize as visualize_module


async def _evaluate_rows(
    strategy: RejectStrategy,
    rows: list[tuple[str, int]],
    *,
    threshold: float,
    concurrency: int,
) -> StrategyRunSummary:
    summary = StrategyRunSummary(name=strategy.name)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(text: str, gold: int) -> None:
        async with semaphore:
            begin = time.perf_counter()
            pred, trace = await strategy.predict(text, threshold=threshold)
            elapsed_ms = (time.perf_counter() - begin) * 1000.0
            summary.latencies_ms.append(elapsed_ms)
            summary.reject.update(gold, pred)
            if trace.llm_calls:
                summary.cost.add_llm_call(trace.prompt_tokens, trace.completion_tokens)

    await asyncio.gather(*[_one(text, gold) for text, gold in rows])
    return summary


async def _evaluate_multiturn(
    strategy: RejectStrategy,
    sessions: list[dict],
    *,
    threshold: float,
    concurrency: int,
) -> StrategyRunSummary:
    summary = StrategyRunSummary(name=strategy.name)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _session(session: dict) -> None:
        async with semaphore:
            turns = [str(item).strip() for item in session.get("turns", []) if str(item).strip()]
            gold = [int(item) for item in session.get("gold", [])]
            if not turns or len(gold) != len(turns):
                return
            history_parts: list[str] = []
            all_ok = True
            for idx, query in enumerate(turns):
                begin = time.perf_counter()
                history = "\n".join(f"- {part}" for part in history_parts)
                pred, trace = await strategy.predict(query, threshold=threshold, history=history)
                elapsed_ms = (time.perf_counter() - begin) * 1000.0
                summary.latencies_ms.append(elapsed_ms)
                summary.reject.update(gold[idx], pred)
                if trace.llm_calls:
                    summary.cost.add_llm_call(trace.prompt_tokens, trace.completion_tokens)
                if pred != gold[idx]:
                    all_ok = False
                history_parts.append(query)
            summary.multiturn_sessions += 1
            if all_ok:
                summary.multiturn_sessions_all_correct += 1

    await asyncio.gather(*[_session(item) for item in sessions])
    return summary


def _serialize_summary(summary: StrategyRunSummary, *, include_multiturn_fields: bool) -> dict:
    mean_lat, p95_lat = latency_mean_p95(summary.latencies_ms)
    base = {
        "name": summary.name,
        "latency_ms_mean": mean_lat,
        "latency_ms_p95": p95_lat,
        "llm_calls": summary.cost.llm_calls,
        "estimated_prompt_tokens": summary.cost.estimated_prompt_tokens,
        "estimated_completion_tokens": summary.cost.estimated_completion_tokens,
        "estimated_total_tokens": summary.cost.estimated_prompt_tokens + summary.cost.estimated_completion_tokens,
        "reject_accuracy": summary.reject.accuracy,
        "false_accept_rate_on_should_reject": summary.reject.false_accept_rate,
        "false_reject_rate_on_should_accept": summary.reject.false_reject_rate,
        "reject_detection_recall": summary.reject.reject_detection_recall,
        "samples_scored": summary.reject.total,
        "gold_reject_labeled_samples": summary.reject.gold_reject,
        "gold_accept_labeled_samples": summary.reject.total - summary.reject.gold_reject,
    }
    if include_multiturn_fields:
        base["multiturn_session_success_rate"] = summary.multiturn_success_rate()
        base["multiturn_sessions"] = summary.multiturn_sessions
    return base


async def run_all(
    *,
    threshold: float,
    margin: float,
    max_samples: int | None,
    max_multiturn_sessions: int | None,
    concurrency: int,
    multiturn_path: Path,
    skip_llm: bool,
) -> dict:
    rows = load_reject_test_rows(max_samples)
    multiturn_sessions = load_multiturn_sessions(multiturn_path)
    if max_multiturn_sessions is not None and max_multiturn_sessions > 0:
        multiturn_sessions = multiturn_sessions[:max_multiturn_sessions]

    strategies: list[RejectStrategy] = [BertOnlyRejectStrategy()]
    if not skip_llm:
        profile = default_llm_profile()
        strategies.extend(
            [
                FullLlmRejectStrategy(profile),
                BertPrescreenLlmDeepRejectStrategy(profile, margin=margin),
            ]
        )

    singles: list[dict] = []
    for strategy in strategies:
        summary = await _evaluate_rows(strategy, rows, threshold=threshold, concurrency=concurrency)
        singles.append(_serialize_summary(summary, include_multiturn_fields=False))

    multiturn_block: list[dict] = []
    if multiturn_sessions:
        for strategy in strategies:
            summary = await _evaluate_multiturn(
                strategy, multiturn_sessions, threshold=threshold, concurrency=concurrency
            )
            multiturn_block.append(_serialize_summary(summary, include_multiturn_fields=True))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "margin": margin,
        "llm_profile": default_llm_profile(),
        "reject_test_samples": len(rows),
        "multiturn_sessions": len(multiturn_sessions),
        "multiturn_path": str(multiturn_path),
        "single_turn": singles,
        "multi_turn": multiturn_block,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


async def main() -> int:
    parser = argparse.ArgumentParser(description="BERT vs full-LLM reject routing experiment.")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("BERT_VALUE_THRESHOLD", "0.5")))
    parser.add_argument("--margin", type=float, default=float(os.getenv("BERT_VALUE_MARGIN", "0.12")))
    parser.add_argument("--max-samples", type=int, default=int(os.getenv("BERT_VALUE_MAX_SAMPLES", "0") or 0))
    parser.add_argument(
        "--max-multiturn-sessions",
        type=int,
        default=int(os.getenv("BERT_VALUE_MAX_MULTITURN", "0") or 0),
        help="Cap multiturn JSONL sessions (0 = all).",
    )
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("BERT_VALUE_CONCURRENCY", "6")))
    parser.add_argument(
        "--multiturn-path",
        type=Path,
        default=Path(os.getenv("BERT_VALUE_MULTITURN_PATH", str(default_multiturn_path()))),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only run BERT (no API calls). Useful for CI without credentials.",
    )
    args = parser.parse_args()

    max_samples = None if args.max_samples <= 0 else args.max_samples
    max_multiturn = None if args.max_multiturn_sessions <= 0 else args.max_multiturn_sessions
    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = result_path(f"bert_value_{stamp}")

    payload = await run_all(
        threshold=args.threshold,
        margin=args.margin,
        max_samples=max_samples,
        max_multiturn_sessions=max_multiturn,
        concurrency=args.concurrency,
        multiturn_path=args.multiturn_path,
        skip_llm=args.skip_llm,
    )
    results_path = out_dir / "results.json"
    _write_json(results_path, payload)
    print(f"Wrote {results_path}")

    if not args.skip_plots:
        plot_paths = visualize_module.build_figures(payload, out_dir)
        for item in plot_paths:
            print(f"Wrote {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
