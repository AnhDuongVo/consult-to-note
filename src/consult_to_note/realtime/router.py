"""Latency-aware model routing.

Candidates are ordered from best quality to fastest. For every update the router picks the best model
whose recent latency (EWMA plus a safety margin) fits the budget. If none fits, it takes the fastest one
and shrinks the output budget. Latencies are learned online, so the router adapts when an endpoint
slows down under load (for example when other clinics share the same GPU).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    model: str
    max_tokens: int = 384
    prior_latency_s: float = 0.6  # starting guess before we have measurements


@dataclass
class Decision:
    model: str
    max_tokens: int
    expected_s: float
    reason: str


@dataclass
class LatencyRouter:
    candidates: list[Candidate]
    budget_s: float = 1.0
    alpha: float = 0.3  # EWMA weight of the newest observation
    margin: float = 1.15  # expected latency is inflated by 15% before comparing with the budget
    explore_every: int = 15  # re-measure a skipped higher-quality model now and then
    ewma: dict[str, float] = field(default_factory=dict)
    history: list[tuple[str, float]] = field(default_factory=list)
    _decisions: int = 0

    def expected(self, model: str) -> float:
        prior = next(c.prior_latency_s for c in self.candidates if c.model == model)
        return self.ewma.get(model, prior)

    def choose(self, pending_utterances: int = 1) -> Decision:
        # More pending speech means a longer prompt and a longer patch: scale the estimate a little.
        load = 1.0 + 0.1 * max(0, pending_utterances - 1)
        self._decisions += 1
        if self.explore_every and self._decisions % self.explore_every == 0:
            best = self.candidates[0]
            est = self.expected(best.model) * load
            if est * self.margin > self.budget_s:
                return Decision(best.model, best.max_tokens, est, "exploration: re-measuring the best model")
        for c in self.candidates:
            est = self.expected(c.model) * load
            if est * self.margin <= self.budget_s:
                return Decision(c.model, c.max_tokens, est, f"fits budget ({est:.2f}s)")
        fastest = min(self.candidates, key=lambda c: self.expected(c.model))
        est = self.expected(fastest.model) * load
        # Keep max_tokens: truncating guided JSON would break the update entirely.
        return Decision(
            fastest.model, fastest.max_tokens, est, f"over budget ({est:.2f}s > {self.budget_s:.2f}s): fastest model"
        )

    def observe_failure(self, model: str) -> None:
        """A failed call (timeout, 429, wrong model id) must make a model look slow, not fast."""
        self.observe(model, max(self.expected(model), self.budget_s) * 2)

    def observe(self, model: str, latency_s: float) -> None:
        prev = self.ewma.get(model)
        self.ewma[model] = latency_s if prev is None else self.alpha * latency_s + (1 - self.alpha) * prev
        self.history.append((model, latency_s))
