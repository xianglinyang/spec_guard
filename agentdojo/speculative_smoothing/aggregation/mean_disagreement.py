from __future__ import annotations

from dataclasses import dataclass

from speculative_smoothing.interfaces import ScoreAggregator
from speculative_smoothing.schemas import (
    BranchDraft,
    BranchScore,
    DecisionLabel,
    DisagreementStats,
    GuardrailDecision,
    GuardrailState,
    RuntimeStats,
    SafetyLens,
)


@dataclass(slots=True)
class MeanDisagreementAggregator(ScoreAggregator):
    """Aggregate branch risks via mean + disagreement penalty."""

    disagreement_weight: float = 0.25
    block_threshold: float = 0.70
    escalate_threshold: float = 0.45

    def aggregate(
        self,
        state: GuardrailState,
        branches: list[BranchDraft],
        scores: list[float | BranchScore],
    ) -> GuardrailDecision:
        branch_scores = [self._to_branch_score(i, b, s) for i, (b, s) in enumerate(zip(branches, scores))]
        values = [x.score for x in branch_scores] or [0.0]

        mean = sum(values) / len(values)
        min_v = min(values)
        max_v = max(values)
        range_v = max_v - min_v
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        pairwise = self._pairwise_mean_abs_diff(values)

        # S = mu + lambda * sigma
        final = mean + self.disagreement_weight * std

        if final >= self.block_threshold:
            decision = DecisionLabel.BLOCK
        elif final >= self.escalate_threshold:
            decision = DecisionLabel.ESCALATE
        else:
            decision = DecisionLabel.ALLOW

        selected_lenses: list[SafetyLens] = []
        seen: set[SafetyLens] = set()
        for branch in branches:
            if branch.lens not in seen:
                selected_lenses.append(branch.lens)
                seen.add(branch.lens)

        disagreement = DisagreementStats(
            mean=mean,
            std=std,
            min_score=min_v,
            max_score=max_v,
            range_score=range_v,
            pairwise_mean_abs_diff=pairwise,
        )

        return GuardrailDecision(
            final_score=final,
            decision=decision,
            per_branch_scores=branch_scores,
            selected_lenses=selected_lenses,
            disagreement_stats=disagreement,
            runtime_metadata=RuntimeStats(total_ms=0.0, branch_count=len(branches)),
            decision_thresholds={
                "block": self.block_threshold,
                "escalate": self.escalate_threshold,
                "disagreement_weight": self.disagreement_weight,
            },
            metadata={"aggregator": "MeanDisagreementAggregator", "state_id": state.state_id},
        )

    @staticmethod
    def _to_branch_score(index: int, branch: BranchDraft, item: float | BranchScore) -> BranchScore:
        if isinstance(item, BranchScore):
            return item
        return BranchScore(
            branch_id=branch.branch_id or f"branch_{index}",
            score=float(item),
            lens=branch.lens,
            rationale="raw_float_score",
        )

    @staticmethod
    def _pairwise_mean_abs_diff(values: list[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        diffs = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                diffs += abs(values[i] - values[j])
                count += 1
        return diffs / count if count else 0.0
