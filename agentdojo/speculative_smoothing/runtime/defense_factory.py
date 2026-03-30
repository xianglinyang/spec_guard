from __future__ import annotations

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor

from custom_registry import register_openrouter_defense
from speculative_smoothing.config import SpeculativeSmoothingConfig
from speculative_smoothing.runtime.guard_component import SpeculativeSmoothingGuardElement


@register_openrouter_defense(
    "speculative_smoothing",
    description=(
        "Prototype pre-tool-call defense using speculative smoothing: "
        "sample lenses, generate short draft branches, score and aggregate risk."
    ),
)
def build_speculative_smoothing_defense(
    *,
    llm,
    system_message_component,
    init_query_component,
    tool_output_formatter,
    base_pipeline_name,
    **_,
) -> AgentPipeline:
    cfg = SpeculativeSmoothingConfig.from_env()

    guard = SpeculativeSmoothingGuardElement(config=cfg, llm_client=getattr(llm, "client", None))
    tools_loop = ToolsExecutionLoop([guard, ToolsExecutor(tool_output_formatter), llm])

    pipeline = AgentPipeline([system_message_component, init_query_component, llm, tools_loop])
    pipeline.name = f"{base_pipeline_name}--speculative_smoothing"
    return pipeline
