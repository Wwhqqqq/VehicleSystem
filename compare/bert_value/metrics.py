from __future__ import annotations

import math
from dataclasses import dataclass, field


def percentile_sorted(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    return float(sorted_values[f] * (c - k) + sorted_values[c] * (k - f))


def latency_mean_p95(latencies_ms: list[float]) -> tuple[float, float]:
    if not latencies_ms:
        return 0.0, 0.0
    ordered = sorted(latencies_ms)
    return float(sum(ordered) / len(ordered)), percentile_sorted(ordered, 95.0)


@dataclass
class RejectMetrics:
    """Labels: 0 = should reject (拒识), 1 = accept (不拒识). Prediction same encoding."""

    total: int = 0
    correct: int = 0
    gold_reject: int = 0
    pred_accept_when_should_reject: int = 0
    pred_reject_when_should_accept: int = 0

    def update(self, gold: int, pred: int) -> None:
        self.total += 1
        if pred == gold:
            self.correct += 1
        if gold == 0:
            self.gold_reject += 1
            if pred == 1:
                self.pred_accept_when_should_reject += 1
        if gold == 1 and pred == 0:
            self.pred_reject_when_should_accept += 1

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def false_accept_rate(self) -> float:
        """Among utterances that should be rejected, fraction wrongly accepted (误召回)."""
        return self.pred_accept_when_should_reject / self.gold_reject if self.gold_reject else 0.0

    @property
    def false_reject_rate(self) -> float:
        """Among acceptable utterances, fraction wrongly rejected (误拒识)."""
        gold_accept = self.total - self.gold_reject
        return self.pred_reject_when_should_accept / gold_accept if gold_accept else 0.0

    @property
    def reject_detection_recall(self) -> float:
        """Among gold reject (0), fraction correctly rejected."""
        tp = self.gold_reject - self.pred_accept_when_should_reject
        return tp / self.gold_reject if self.gold_reject else 0.0


@dataclass
class CostAccount:
    llm_calls: int = 0
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0

    def add_llm_call(self, prompt_tokens: int, completion_tokens: int = 2) -> None:
        self.llm_calls += 1
        self.estimated_prompt_tokens += prompt_tokens
        self.estimated_completion_tokens += completion_tokens


@dataclass
class StrategyRunSummary:
    name: str
    latencies_ms: list[float] = field(default_factory=list)
    reject: RejectMetrics = field(default_factory=RejectMetrics)
    cost: CostAccount = field(default_factory=CostAccount)
    multiturn_sessions: int = 0
    multiturn_sessions_all_correct: int = 0

    def multiturn_success_rate(self) -> float:
        if not self.multiturn_sessions:
            return 0.0
        return self.multiturn_sessions_all_correct / self.multiturn_sessions
