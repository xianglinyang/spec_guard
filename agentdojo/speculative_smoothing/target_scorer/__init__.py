"""Target scorer implementations for speculative smoothing."""

from speculative_smoothing.target_scorer.heuristic import HeuristicTargetScorer
from speculative_smoothing.target_scorer.llm import LLMTargetScorer
from speculative_smoothing.target_scorer.tree_attention import TreeAttentionTargetScorer

__all__ = ["HeuristicTargetScorer", "LLMTargetScorer", "TreeAttentionTargetScorer"]
