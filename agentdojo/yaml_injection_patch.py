from __future__ import annotations

from typing import Any

import yaml
from yaml.parser import ParserError


def _escape_for_yaml_double_quoted(value: str) -> str:
    # AgentDojo suite templates mostly place injection placeholders inside YAML
    # double-quoted scalars. Escape content so attacker strings cannot break YAML.
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def apply_agentdojo_yaml_injection_patch() -> None:
    from agentdojo.task_suite.task_suite import TaskSuite, validate_injections, read_suite_file

    if getattr(TaskSuite, "_yaml_injection_patch_applied", False):
        return

    def _patched_load_and_inject_default_environment(self: Any, injections: dict[str, str]):
        environment_text = read_suite_file(self.name, "environment.yaml", self.data_path)
        injection_vector_defaults = self.get_injection_vector_defaults()
        validate_injections(injections, injection_vector_defaults)

        escaped_injections = {k: _escape_for_yaml_double_quoted(v) for k, v in injections.items()}
        injections_with_defaults = dict(injection_vector_defaults, **escaped_injections)

        injected_environment = environment_text.format(**injections_with_defaults)
        try:
            parsed = yaml.safe_load(injected_environment)
        except ParserError as exc:
            mark = exc.problem_mark
            hint = (
                f"YAML parse failed after injection formatting in suite={self.name}. "
                f"line={getattr(mark, 'line', -1) + 1}, column={getattr(mark, 'column', -1) + 1}. "
                "Likely cause: unescaped attack payload content in injection vectors."
            )
            raise ValueError(hint) from exc
        return self.environment_type.model_validate(parsed)

    TaskSuite.load_and_inject_default_environment = _patched_load_and_inject_default_environment
    TaskSuite._yaml_injection_patch_applied = True

