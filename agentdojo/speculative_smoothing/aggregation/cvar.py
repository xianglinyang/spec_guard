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
class CVaRScoreAggregator(ScoreAggregator):
    """Aggregate branch risks via Conditional Value-at-Risk (CVaR).

    CVaR_α is the expected score of the worst (1-α) fraction of branches,
    i.e. the mean of scores above the α-quantile.  Using α=0.5 means the
    decision is driven by the riskier half of branches.

    Why CVaR instead of mean + std:
    - Bounded in [0, 1] by construction.
    - Conservative: a single high-risk branch can elevate the final score.
    - Theoretically motivated: maximising CVaR query-variance per probe
      increases adaptive-attack query complexity (see SPBV analysis).
    """

    alpha: float = 0.5
    block_threshold: float = 0.70
    escalate_threshold: float = 0.45

    def aggregate(
        self,
        state: GuardrailState,
        branches: list[BranchDraft],
        scores: list[float | BranchScore],
    ) -> GuardrailDecision:
        branch_scores = [
            self._to_branch_score(i, b, s) for i, (b, s) in enumerate(zip(branches, scores))
        ]
        values = sorted(x.score for x in branch_scores)

        if not values:
            values = [0.0]

        # CVaR: mean of the worst (1-alpha) tail.
        tail_start = int(len(values) * self.alpha)
        tail = values[tail_start:] if tail_start < len(values) else values[-1:]
        final = sum(tail) / len(tail)

        # Disagreement stats (kept for analysis / logging parity with mean aggregator).
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        pairwise = self._pairwise_mean_abs_diff(values)
        disagreement = DisagreementStats(
            mean=mean,
            std=std,
            min_score=values[0],
            max_score=values[-1],
            range_score=values[-1] - values[0],
            pairwise_mean_abs_diff=pairwise,
        )

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
                "cvar_alpha": self.alpha,
            },
            metadata={"aggregator": "CVaRScoreAggregator", "state_id": state.state_id},
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
