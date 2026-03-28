from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreBreakdown:
    success_bonus: float
    tool_task_deviation: float
    critic_score_1_to_10: int
    total_score: float


@dataclass
class EpisodeResult:
    success: bool
    utility: bool
    tool_calls: list[str]
    model_output: str
    textual_feedback: str
    tool_overlap: float
    task_deviation: float
    error: str | None = None
    raw_messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CandidateRecord:
    candidate_id: str
    text_trigger: str
    iteration_created: int
    parent_ids: list[str]
    score: ScoreBreakdown
    episode: EpisodeResult
    critic_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationState:
    iteration: int
    baseline: str
    selected_parent_ids: list[str]
    new_candidate_ids: list[str]
    best_candidate_id: str | None
    best_total_score: float | None
    depth: int | None = None
    stop_reason: str | None = None
