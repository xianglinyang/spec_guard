from __future__ import annotations

from dataclasses import dataclass

from speculative_smoothing.interfaces import TargetScorer
from speculative_smoothing.schemas import BranchDraft, BranchScore, GuardrailState, SafetyLens, VerdictLabel


@dataclass(slots=True)
class HeuristicTargetScorer(TargetScorer):
    """Lightweight heuristic scorer for first integration.

    This scorer stands in for the expensive target guardrail model and provides
    branch-level risk estimates in [0, 1].
    """

    def score_branches(self, state: GuardrailState, branches: list[BranchDraft]) -> list[float | BranchScore]:
        out: list[BranchScore] = []
        for branch in branches:
            score = self._score_one(branch)
            out.append(
                BranchScore(
                    branch_id=branch.branch_id,
                    score=score,
                    lens=branch.lens,
                    rationale=f"heuristic_{branch.lens.value}",
                    metadata={"scorer": "HeuristicTargetScorer"},
                )
            )
        return out

    @staticmethod
    def _score_one(branch: BranchDraft) -> float:
        verdict = branch.verdict_label
        if verdict == VerdictLabel.UNSAFE:
            base = 0.90
        elif verdict == VerdictLabel.UNCERTAIN:
            base = 0.55
        else:
            base = 0.10

        text = (branch.text or "").lower()
        if branch.lens == SafetyLens.CONFLICT_INJECTION and any(x in text for x in ["ignore", "override", "jailbreak"]):
            base += 0.08
        if branch.lens == SafetyLens.PROVENANCE and any(x in text for x in ["unverified", "unknown source"]):
            base += 0.06
        if branch.lens == SafetyLens.TOOL_RISK and any(x in text for x in ["execute", "shell", "transfer", "delete"]):
            base += 0.08
        if branch.lens == SafetyLens.GOAL_CONSISTENCY and "uncertain" in text:
            base += 0.03

        return max(0.0, min(1.0, base))
