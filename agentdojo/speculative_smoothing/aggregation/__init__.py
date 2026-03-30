"""Score aggregation implementations for speculative smoothing."""

from speculative_smoothing.aggregation.cvar import CVaRScoreAggregator
from speculative_smoothing.aggregation.mean_disagreement import MeanDisagreementAggregator

__all__ = ["CVaRScoreAggregator", "MeanDisagreementAggregator"]
