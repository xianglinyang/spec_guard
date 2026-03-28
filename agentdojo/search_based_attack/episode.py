from __future__ import annotations

from dataclasses import dataclass

from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.task_suite import (
    TaskSuite,
    functions_stack_trace_from_messages,
    model_output_from_messages,
)
from agentdojo.types import get_text_content_as_str

from search_based_attack.schemas import EpisodeResult


class UniformTriggerLocator(BaseAttack):
    name = "uniform_trigger_locator"

    def attack(self, user_task: BaseUserTask, injection_task: BaseInjectionTask) -> dict[str, str]:
        return {}


@dataclass
class EpisodeRunner:
    suite: TaskSuite
    pipeline: object
    user_task: BaseUserTask
    injection_task: BaseInjectionTask

    def _build_injections(self, trigger: str) -> dict[str, str]:
        locator = UniformTriggerLocator(self.suite, self.pipeline)
        candidates = locator.get_injection_candidates(self.user_task)
        return {inj: trigger for inj in candidates}

    def run_one_episode(self, trigger: str) -> EpisodeResult:
        try:
            injections = self._build_injections(trigger)
            environment = self.suite.load_and_inject_default_environment(injections)
            task_environment = self.user_task.init_environment(environment)
            pre_environment = task_environment.model_copy(deep=True)
            prompt = self.user_task.PROMPT

            runtime = FunctionsRuntime(self.suite.tools)
            messages = []
            for _ in range(3):
                try:
                    _, _, task_environment, messages, _ = self.pipeline.query(prompt, runtime, task_environment)
                except AbortAgentError as e:
                    task_environment = e.task_environment
                    messages = e.messages
                model_output = model_output_from_messages(messages)
                if model_output is not None:
                    break

            model_output = model_output_from_messages(messages) if len(messages) > 0 else []
            functions_stack = functions_stack_trace_from_messages(messages)
            tool_calls = [call.function for call in functions_stack]

            utility = self.suite._check_task_result(
                self.user_task,
                model_output or [],
                pre_environment,
                task_environment,
                functions_stack,
            )
            success = self.suite._check_task_result(
                self.injection_task,
                model_output or [],
                pre_environment,
                task_environment,
                functions_stack,
            )

            gt_calls = self.injection_task.ground_truth(pre_environment.model_copy(deep=True))
            gt_tools = {c.function for c in gt_calls}
            called_tools = set(tool_calls)
            if len(gt_tools) == 0:
                tool_overlap = 0.0
            else:
                tool_overlap = len(gt_tools & called_tools) / len(gt_tools)

            task_deviation = 1.0 if not utility else 0.0
            assistant_text = ""
            if model_output:
                assistant_text = get_text_content_as_str(model_output)
            textual_feedback = (
                f"success={success}; utility={utility}; tool_calls={tool_calls}; output={assistant_text[:1500]}"
            )

            return EpisodeResult(
                success=bool(success),
                utility=bool(utility),
                tool_calls=tool_calls,
                model_output=assistant_text,
                textual_feedback=textual_feedback,
                tool_overlap=float(tool_overlap),
                task_deviation=float(task_deviation),
                raw_messages=[{"role": m["role"], "content": str(m.get("content"))} for m in messages],
            )
        except Exception as e:
            return EpisodeResult(
                success=False,
                utility=False,
                tool_calls=[],
                model_output="",
                textual_feedback=f"episode_error: {e}",
                tool_overlap=0.0,
                task_deviation=1.0,
                error=str(e),
            )
