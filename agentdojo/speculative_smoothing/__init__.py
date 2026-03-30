"""Core typed schemas and interfaces for speculative smoothing defense."""

from speculative_smoothing.interfaces import (
    AdaptiveAttacker,
    BenchmarkAdapter,
    DraftBranchGenerator,
    LensSampler,
    ScoreAggregator,
    TargetScorer,
)
from speculative_smoothing.aggregation import MeanDisagreementAggregator
from speculative_smoothing.config import SpeculativeSmoothingConfig
from speculative_smoothing.draft_generator import LLMDraftBranchGenerator
from speculative_smoothing.lens_sampler import StochasticLensSampler
from speculative_smoothing.schemas import (
    BranchDraft,
    BranchScore,
    DecisionLabel,
    DisagreementStats,
    EvidenceSpan,
    GuardrailDecision,
    GuardrailState,
    HistoryTurn,
    RuntimeStats,
    SafetyLens,
    ToolCallSpec,
    VerdictLabel,
)
from speculative_smoothing.state_builder import DefaultStateBuilder
from speculative_smoothing.target_scorer import HeuristicTargetScorer, LLMTargetScorer, TreeAttentionTargetScorer

__all__ = [
    "AdaptiveAttacker",
    "BenchmarkAdapter",
    "DraftBranchGenerator",
    "LensSampler",
    "ScoreAggregator",
    "TargetScorer",
    "SpeculativeSmoothingConfig",
    "MeanDisagreementAggregator",
    "HeuristicTargetScorer",
    "LLMTargetScorer",
    "TreeAttentionTargetScorer",
    "LLMDraftBranchGenerator",
    "StochasticLensSampler",
    "DefaultStateBuilder",
    "BranchDraft",
    "BranchScore",
    "DecisionLabel",
    "DisagreementStats",
    "EvidenceSpan",
    "GuardrailDecision",
    "GuardrailState",
    "HistoryTurn",
    "RuntimeStats",
    "SafetyLens",
    "ToolCallSpec",
    "VerdictLabel",
]
