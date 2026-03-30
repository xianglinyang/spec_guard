from __future__ import annotations

from typing import Any

from speculative_smoothing.interfaces import BenchmarkAdapter
from speculative_smoothing.schemas import GuardrailState, HistoryTurn, ToolCallSpec


class DefaultStateBuilder(BenchmarkAdapter):
    """Default benchmark adapter that normalizes runtime payloads into GuardrailState.

    This class is intentionally lightweight and backend-agnostic:
    callers pass the fields already available at a pre-tool-call boundary,
    and the adapter normalizes types for history/tool-call structures.
    """

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
        history = self._normalize_history(recent_history)
        tool_call = self._normalize_tool_call(proposed_tool_call)

        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("builder", "DefaultStateBuilder")

        return GuardrailState(
            user_goal=user_goal,
            task_context=system_or_task_context,
            untrusted_context=untrusted_context,
            recent_history=history,
            proposed_tool_call=tool_call,
            metadata=merged_metadata,
        )

    @staticmethod
    def _normalize_history(items: list[dict[str, Any]] | list[Any]) -> list[HistoryTurn]:
        out: list[HistoryTurn] = []
        for item in items:
            if isinstance(item, HistoryTurn):
                out.append(item)
                continue

            if isinstance(item, dict):
                out.append(
                    HistoryTurn(
                        role=str(item.get("role", "unknown")),
                        content=str(item.get("content", "")),
                        tool_name=(str(item["tool_name"]) if item.get("tool_name") is not None else None),
                        timestamp=(str(item["timestamp"]) if item.get("timestamp") is not None else None),
                        metadata=dict(item.get("metadata", {})),
                    )
                )
                continue

            out.append(HistoryTurn(role="unknown", content=str(item)))
        return out

    @staticmethod
    def _normalize_tool_call(item: dict[str, Any] | str) -> ToolCallSpec:
        if isinstance(item, ToolCallSpec):
            return item

        if isinstance(item, str):
            return ToolCallSpec(tool_name=item)

        return ToolCallSpec(
            tool_name=str(item.get("tool_name", "unknown_tool")),
            arguments=dict(item.get("arguments", {})),
            rationale=(str(item["rationale"]) if item.get("rationale") is not None else None),
            raw_action=(str(item["raw_action"]) if item.get("raw_action") is not None else None),
            metadata=dict(item.get("metadata", {})),
        )
