from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SafetyLens(str, Enum):
    """Safety analysis lenses used by speculative smoothing."""

    CONFLICT_INJECTION = "conflict_injection"
    PROVENANCE = "provenance"
    TOOL_RISK = "tool_risk"
    GOAL_CONSISTENCY = "goal_consistency"


class DecisionLabel(str, Enum):
    """Final guardrail decision labels."""

    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"


class VerdictLabel(str, Enum):
    """Optional parsed verdict label for one branch draft."""

    SAFE = "safe"
    UNSAFE = "unsafe"
    UNCERTAIN = "uncertain"


@dataclass(slots=True)
class EvidenceSpan:
    """Location and content hint for branch evidence extracted from context."""

    source: str
    start: int | None = None
    end: int | None = None
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HistoryTurn:
    """One recent interaction turn used in pre-tool-call guardrail context."""

    role: str
    content: str
    tool_name: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCallSpec:
    """Structured proposed action/tool call at the current decision boundary."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str | None = None
    raw_action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GuardrailState:
    """Structured state object for one pre-tool-call safety decision point."""

    user_goal: str
    untrusted_context: str
    recent_history: list[HistoryTurn]
    proposed_tool_call: ToolCallSpec
    task_context: str | None = None
    system_instructions: str | None = None
    state_id: str | None = None
    episode_id: str | None = None
    step_index: int | None = None
    timestamp_utc: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_context and not self.system_instructions:
            raise ValueError("Either task_context or system_instructions must be provided.")
        # Provide a backward-compatible alias behavior.
        if self.task_context is None and self.system_instructions is not None:
            self.task_context = self.system_instructions
        if self.system_instructions is None and self.task_context is not None:
            self.system_instructions = self.task_context

    @property
    def proposed_action(self) -> ToolCallSpec:
        """Alias for compatibility with callers using `proposed_action` naming."""

        return self.proposed_tool_call


@dataclass(slots=True)
class BranchDraft:
    """One short safety-analysis branch generated under a selected lens."""

    lens: SafetyLens
    text: str
    branch_id: str | None = None
    evidence_span: EvidenceSpan | None = None
    verdict_label: VerdictLabel | None = None
    logprob: float | None = None
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    randomness_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BranchScore:
    """Branch-level scoring output from target guardrail model."""

    branch_id: str | None
    score: float
    lens: SafetyLens | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DisagreementStats:
    """Statistics that capture disagreement across branch-level risk estimates."""

    mean: float
    std: float
    min_score: float
    max_score: float
    range_score: float
    pairwise_mean_abs_diff: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeStats:
    """Latency/runtime telemetry for one guardrail decision pass."""

    total_ms: float
    draft_ms: float | None = None
    scoring_ms: float | None = None
    aggregation_ms: float | None = None
    branch_count: int | None = None
    model_calls: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GuardrailDecision:
    """Final output object for a speculative smoothing guardrail decision."""

    final_score: float
    decision: DecisionLabel
    per_branch_scores: list[BranchScore]
    selected_lenses: list[SafetyLens]
    disagreement_stats: DisagreementStats | None = None
    runtime_metadata: RuntimeStats | None = None
    explanations: list[str] = field(default_factory=list)
    decision_thresholds: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
