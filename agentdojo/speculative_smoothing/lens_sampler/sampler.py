from __future__ import annotations

import random
from dataclasses import dataclass, field

from speculative_smoothing.interfaces import LensSampler
from speculative_smoothing.schemas import GuardrailState, SafetyLens


@dataclass(slots=True)
class StochasticLensSampler(LensSampler):
    """Stochastic lens sampler for speculative smoothing.

    Strategy (when use_random_sampling=True, the default):
    - Randomly sample k lenses from the full pool using the provided seed.
    - The selection is fully determined by the seed, which is a fresh random
      value per invocation in production, making it unpredictable to attackers.

    Strategy (when use_random_sampling=False, legacy mode):
    - Compute heuristic lens scores from guardrail state text.
    - Add small stochastic jitter (controlled by seed).
    - Return top-k unique lenses.
    """

    use_random_sampling: bool = True
    base_weights: dict[SafetyLens, float] = field(
        default_factory=lambda: {
            SafetyLens.CONFLICT_INJECTION: 1.0,
            SafetyLens.PROVENANCE: 1.0,
            SafetyLens.TOOL_RISK: 1.0,
            SafetyLens.GOAL_CONSISTENCY: 1.0,
        }
    )
    jitter: float = 0.15

    def sample_lenses(self, state: GuardrailState, k: int, rng_seed: int | None = None) -> list[SafetyLens]:
        if k <= 0:
            return []

        all_lenses = list(SafetyLens)
        k = min(k, len(all_lenses))

        if self.use_random_sampling:
            rng = random.Random(rng_seed)
            return rng.sample(all_lenses, k)

        # Legacy heuristic mode.
        rng = random.Random(rng_seed)
        scored: list[tuple[float, SafetyLens]] = []
        for lens in all_lenses:
            base = self.base_weights.get(lens, 1.0)
            heuristic = self._heuristic_signal(lens, state)
            noise = rng.uniform(-self.jitter, self.jitter)
            scored.append((base + heuristic + noise, lens))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [lens for _, lens in scored[:k]]

    @staticmethod
    def _heuristic_signal(lens: SafetyLens, state: GuardrailState) -> float:
        ctx = (state.untrusted_context or "").lower()
        goal = (state.user_goal or "").lower()
        tool_name = (state.proposed_tool_call.tool_name or "").lower()

        if lens == SafetyLens.CONFLICT_INJECTION:
            tokens = ["ignore", "override", "developer", "system prompt", "must", "instead"]
            return 0.08 * sum(tok in ctx for tok in tokens)

        if lens == SafetyLens.PROVENANCE:
            tokens = ["source", "reference", "quoted", "from website", "claimed", "unverified"]
            return 0.08 * sum(tok in ctx for tok in tokens)

        if lens == SafetyLens.TOOL_RISK:
            risky_tools = ["execute", "shell", "send_email", "transfer", "delete", "purchase"]
            return 0.1 * sum(tok in tool_name for tok in risky_tools)

        if lens == SafetyLens.GOAL_CONSISTENCY:
            return 0.25 if goal and tool_name and goal.split()[0] not in tool_name else 0.0

        return 0.0
