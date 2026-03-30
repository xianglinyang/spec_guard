from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from speculative_smoothing.schemas import (
    BranchDraft,
    BranchScore,
    GuardrailDecision,
    GuardrailState,
    SafetyLens,
)


@runtime_checkable
class LensSampler(Protocol):
    """Selects lenses to evaluate for a specific guardrail state."""

    def sample_lenses(self, state: GuardrailState, k: int, rng_seed: int | None = None) -> list[SafetyLens]:
        """Return up to ``k`` selected lenses for this state."""


@runtime_checkable
class DraftBranchGenerator(Protocol):
    """Generates short safety-analysis branches under selected lenses."""

    def generate(
        self,
        state: GuardrailState,
        lenses: list[SafetyLens],
        branches_per_lens: int,
        seed: int | None = None,
    ) -> list[BranchDraft]:
        """Generate branch drafts for each selected lens."""


@runtime_checkable
class TargetScorer(Protocol):
    """Scores branch drafts with a stronger target guardrail model."""

    def score_branches(self, state: GuardrailState, branches: list[BranchDraft]) -> list[float | BranchScore]:
        """Return one score item per input branch in order."""


@runtime_checkable
class ScoreAggregator(Protocol):
    """Aggregates branch-level scores into one final guardrail decision."""

    def aggregate(
        self,
        state: GuardrailState,
        branches: list[BranchDraft],
        scores: list[float | BranchScore],
    ) -> GuardrailDecision:
        """Produce final decision, including disagreement/runtime metadata where available."""


@runtime_checkable
class AdaptiveAttacker(Protocol):
    """Interface for adaptive attack modules interacting with guardrail states/feedback."""

    def propose(
        self,
        state: GuardrailState,
        feedback: GuardrailDecision | dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any] | str:
        """Propose next attack artifact (trigger/payload/policy object)."""


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Normalizes benchmark/runtime internals into canonical GuardrailState objects."""

    def build_guardrail_state(
        self,
        *,
        user_goal: str,
        system_or_task_context: str,
        untrusted_context: str,
        recent_history: list[dict[str, Any]] | list[Any],
        proposed_tool_call: dict[str, Any] | str,
        metadata: dict[str, Any] | None = None,
    ) -> GuardrailState:
        """Construct canonical pre-tool-call state from backend-specific runtime payloads."""
