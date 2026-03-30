from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from speculative_smoothing.interfaces import TargetScorer
from speculative_smoothing.schemas import BranchDraft, BranchScore, GuardrailState


@dataclass(slots=True)
class LLMTargetScorer(TargetScorer):
    """LLM-based target verifier scorer.

    Uses one call to score all branches for a state, with fallback scorer support.
    """

    client: Any
    model: str
    temperature: float = 0.0
    max_tokens: int = 256
    fallback: TargetScorer | None = None

    def score_branches(self, state: GuardrailState, branches: list[BranchDraft]) -> list[float | BranchScore]:
        if not branches:
            return []

        try:
            raw = self._chat_completion(self._build_messages(state, branches))
            parsed = self._parse_scores(raw)
            if not parsed:
                if self.fallback is not None:
                    return self.fallback.score_branches(state, branches)
                return [0.5 for _ in branches]
            out = self._align_scores(branches, parsed)
            if out:
                return out
            if self.fallback is not None:
                return self.fallback.score_branches(state, branches)
            return [0.5 for _ in branches]
        except Exception:
            if self.fallback is not None:
                return self.fallback.score_branches(state, branches)
            return [0.5 for _ in branches]

    def _chat_completion(self, messages: list[dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    @staticmethod
    def _build_messages(state: GuardrailState, branches: list[BranchDraft]) -> list[dict[str, str]]:
        payload = [
            {
                "branch_id": b.branch_id,
                "lens": b.lens.value,
                "text": b.text,
            }
            for b in branches
        ]
        prompt = (
            "Score branch-level risk for prompt-injection safety. "
            "Return strict JSON list, one object per branch, keys: branch_id, score, rationale. "
            "Score must be float in [0,1], where higher means more risky.\n"
            f"User goal: {state.user_goal}\n"
            f"Task context: {state.task_context or state.system_instructions}\n"
            f"Proposed tool: {state.proposed_tool_call.tool_name}\n"
            f"Branches: {json.dumps(payload, ensure_ascii=False)}"
        )
        return [
            {"role": "system", "content": "You are a strict verifier. Output JSON only."},
            {"role": "user", "content": prompt},
        ]

    @staticmethod
    def _parse_scores(raw: str) -> list[dict[str, Any]]:
        data: Any = None
        try:
            data = json.loads(raw)
        except Exception:
            pass

        if data is None:
            arr = re.search(r"\[.*\]", raw, re.DOTALL)
            if arr:
                try:
                    data = json.loads(arr.group(0))
                except Exception:
                    data = None

        if isinstance(data, dict) and isinstance(data.get("scores"), list):
            data = data["scores"]

        if not isinstance(data, list):
            return []

        out: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                out.append(item)
        return out

    @staticmethod
    def _align_scores(branches: list[BranchDraft], parsed: list[dict[str, Any]]) -> list[BranchScore]:
        by_id: dict[str, dict[str, Any]] = {}
        for item in parsed:
            bid = item.get("branch_id")
            if bid is not None:
                by_id[str(bid)] = item

        out: list[BranchScore] = []
        for idx, branch in enumerate(branches):
            source = None
            if branch.branch_id is not None:
                source = by_id.get(str(branch.branch_id))
            if source is None and idx < len(parsed):
                source = parsed[idx]

            if source is None:
                score = 0.5
                rationale = "missing_score"
            else:
                score = LLMTargetScorer._safe_score(source.get("score", 0.5))
                rationale = str(source.get("rationale", ""))

            out.append(
                BranchScore(
                    branch_id=branch.branch_id,
                    score=score,
                    lens=branch.lens,
                    rationale=rationale,
                    metadata={"scorer": "LLMTargetScorer"},
                )
            )

        return out

    @staticmethod
    def _safe_score(value: Any) -> float:
        try:
            v = float(value)
        except Exception:
            v = 0.5
        return max(0.0, min(1.0, v))
