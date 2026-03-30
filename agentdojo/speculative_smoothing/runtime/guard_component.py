from __future__ import annotations

from dataclasses import asdict
import logging
import os
import time
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage

from speculative_smoothing.aggregation import CVaRScoreAggregator, MeanDisagreementAggregator
from speculative_smoothing.config import SpeculativeSmoothingConfig
from speculative_smoothing.draft_generator import LLMDraftBranchGenerator
from speculative_smoothing.lens_sampler import StochasticLensSampler
from speculative_smoothing.schemas import DecisionLabel
from speculative_smoothing.state_builder import DefaultStateBuilder
from speculative_smoothing.target_scorer import HeuristicTargetScorer, LLMTargetScorer, TreeAttentionTargetScorer

logger = logging.getLogger(__name__)


class SpeculativeSmoothingGuardElement(BasePipelineElement):
    """Pre-tool-call guard element using speculative smoothing prototype.

    This component inspects pending tool calls, performs draft branch analysis,
    aggregates branch risks, and can block the pending tool call when risk is high.
    """

    def __init__(self, config: SpeculativeSmoothingConfig | None = None, llm_client: Any | None = None) -> None:
        self.config = config or SpeculativeSmoothingConfig.from_env()
        self._llm_client = llm_client
        self._client_cache: dict[tuple[str, str], Any] = {}
        self.state_builder = DefaultStateBuilder()
        self.lens_sampler = StochasticLensSampler(jitter=self.config.lens_jitter)
        self.draft_generator = self._build_draft_generator()
        self.target_scorer = self._build_target_scorer()
        self.aggregator = self._build_aggregator()
        self._log_runtime_setup()

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: list[ChatMessage] | tuple[ChatMessage, ...] = [],
        extra_args: dict | None = None,
    ):
        extra_args = dict(extra_args or {})
        message_list = list(messages)

        tool_call = self._extract_latest_pending_tool_call(message_list)
        if tool_call is None:
            return query, runtime, env, message_list, extra_args

        # Fresh random seed per invocation — unpredictable to adaptive attackers.
        if self.config.use_fresh_seed:
            # Mask to signed int64 range (OpenAI API seed upper bound = 2^63-1).
            invocation_seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFFFFFFFFFF
        else:
            invocation_seed = self.config.seed

        t0 = time.perf_counter()
        state = self.state_builder.build_guardrail_state(
            user_goal=query,
            system_or_task_context=self._extract_system_or_task_context(message_list),
            untrusted_context=self._extract_untrusted_context(message_list),
            recent_history=self._extract_recent_history(message_list),
            proposed_tool_call=tool_call,
            metadata={"defense": "speculative_smoothing"},
        )

        t_lens = time.perf_counter()
        lenses = self.lens_sampler.sample_lenses(
            state,
            k=self.config.top_k_lenses,
            rng_seed=invocation_seed,
        )
        t_draft_start = time.perf_counter()
        branches = self.draft_generator.generate(
            state=state,
            lenses=lenses,
            branches_per_lens=self.config.branches_per_lens,
            seed=invocation_seed,
        )
        t_score_start = time.perf_counter()
        scores = self.target_scorer.score_branches(state, branches)
        t_agg_start = time.perf_counter()
        decision = self.aggregator.aggregate(state, branches, scores)
        t_end = time.perf_counter()

        if decision.runtime_metadata is not None:
            decision.runtime_metadata.total_ms = (t_end - t0) * 1000.0
            decision.runtime_metadata.draft_ms = (t_score_start - t_draft_start) * 1000.0
            decision.runtime_metadata.scoring_ms = (t_agg_start - t_score_start) * 1000.0
            decision.runtime_metadata.aggregation_ms = (t_end - t_agg_start) * 1000.0
            decision.runtime_metadata.branch_count = len(branches)
            draft_calls = (
                (len(lenses) * max(0, int(self.config.branches_per_lens)))
                if isinstance(self.draft_generator, LLMDraftBranchGenerator)
                else 0
            )
            verifier_llm_calls = 1 if isinstance(self.target_scorer, LLMTargetScorer) else 0
            verifier_tree_calls = 1 if self.config.verifier_backend.strip().lower() == "tree_attention" else 0
            decision.runtime_metadata.model_calls = draft_calls + verifier_llm_calls + verifier_tree_calls
            decision.runtime_metadata.metadata = {
                **decision.runtime_metadata.metadata,
                "invocation_seed": invocation_seed,
                "use_fresh_seed": self.config.use_fresh_seed,
                "lens_sampling_ms": (t_draft_start - t_lens) * 1000.0,
                "draft_concurrent_ms": (t_score_start - t_draft_start) * 1000.0,
                "draft_calls": draft_calls,
                "draft_backend": self.config.draft_backend,
                "draft_model": self.config.draft_model,
                "draft_base_url": (self.config.draft_base_url or self.config.base_url),
                "verifier_backend": self.config.verifier_backend,
                "verifier_llm_calls": verifier_llm_calls,
                "verifier_tree_calls": verifier_tree_calls,
                "verifier_model": (
                    self.config.verifier_tree_model_name_or_path
                    if self.config.verifier_backend.strip().lower() == "tree_attention"
                    else self.config.verifier_model
                ),
                "aggregator": self.config.aggregator,
            }

        extra_args["speculative_smoothing"] = asdict(decision)

        if self.config.log_decisions:
            logger.info(
                "[spec_smooth] decision=%s final=%.3f lenses=%s",
                decision.decision.value,
                decision.final_score,
                [x.value for x in decision.selected_lenses],
            )
            rt = decision.runtime_metadata
            if rt is not None:
                meta = rt.metadata or {}
                logger.info(
                    "[spec_smooth] usage draft_calls=%s draft_backend=%s draft_model=%s draft_base_url=%s verifier_backend=%s verifier_llm_calls=%s verifier_tree_calls=%s verifier_model=%s total_ms=%.1f",
                    meta.get("draft_calls", "n/a"),
                    meta.get("draft_backend", "n/a"),
                    meta.get("draft_model", "n/a"),
                    meta.get("draft_base_url", "n/a"),
                    meta.get("verifier_backend", "n/a"),
                    meta.get("verifier_llm_calls", "n/a"),
                    meta.get("verifier_tree_calls", "n/a"),
                    meta.get("verifier_model", "n/a"),
                    rt.total_ms,
                )

        if decision.decision == DecisionLabel.BLOCK:
            self._clear_pending_tool_calls(message_list)
            if self.config.log_decisions:
                logger.info("[spec_smooth] pending tool call blocked before execution")

        return query, runtime, env, message_list, extra_args

    def _build_aggregator(self):
        name = self.config.aggregator.strip().lower()
        if name == "cvar":
            return CVaRScoreAggregator(
                alpha=self.config.cvar_alpha,
                block_threshold=self.config.block_threshold,
                escalate_threshold=self.config.escalate_threshold,
            )
        return MeanDisagreementAggregator(
            disagreement_weight=self.config.disagreement_weight,
            block_threshold=self.config.block_threshold,
            escalate_threshold=self.config.escalate_threshold,
        )

    def _build_draft_generator(self):
        if self.config.draft_backend.strip().lower() != "llm":
            raise ValueError(
                f"Unsupported SPEC_SMOOTHING_DRAFT_BACKEND={self.config.draft_backend!r}. "
                "Only 'llm' is supported."
            )

        draft_base_url = self.config.draft_base_url.strip() or self.config.base_url
        draft_api_key_env = self.config.draft_api_key_env.strip() or self.config.api_key_env
        client = self._resolve_openai_client(
            prefer_main=self.config.draft_use_main_client,
            api_key_env=draft_api_key_env,
            base_url=draft_base_url,
        )
        if client is None:
            raise ValueError(
                "draft_backend=llm but no OpenAI client is available. "
                "Set SPEC_SMOOTHING_API_KEY_ENV / OPENROUTER_API_KEY correctly."
            )

        return LLMDraftBranchGenerator(
            client=client,
            model=self.config.draft_model,
            temperature=self.config.draft_temperature,
            max_tokens=self.config.draft_max_tokens,
            min_branch_tokens=self.config.draft_branch_min_tokens,
            max_branch_tokens=self.config.draft_branch_max_tokens,
        )

    def _build_target_scorer(self):
        fallback = HeuristicTargetScorer()
        verifier_backend = self.config.verifier_backend.strip().lower()
        if verifier_backend == "tree_attention":
            return TreeAttentionTargetScorer(
                model_name_or_path=self.config.verifier_tree_model_name_or_path,
                device=self.config.verifier_tree_device,
                dtype=self.config.verifier_tree_dtype,
                strict=self.config.verifier_tree_strict,
                fallback=fallback,
            )

        if verifier_backend != "llm":
            return fallback

        client = self._resolve_openai_client(
            prefer_main=True,
            api_key_env=self.config.api_key_env,
            base_url=self.config.base_url,
        )
        if client is None:
            logger.warning("[spec_smooth] verifier_backend=llm but no OpenAI client available; using heuristic scorer.")
            return fallback

        return LLMTargetScorer(
            client=client,
            model=self.config.verifier_model,
            temperature=self.config.verifier_temperature,
            max_tokens=self.config.verifier_max_tokens,
            fallback=fallback,
        )

    def _resolve_openai_client(self, *, prefer_main: bool, api_key_env: str, base_url: str) -> Any | None:
        if prefer_main and self._llm_client is not None:
            return self._llm_client

        api_key = os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        cache_key = (api_key_env, base_url)
        if cache_key in self._client_cache:
            return self._client_cache[cache_key]

        try:
            import openai
        except Exception:
            return None

        headers = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_X_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers or None,
        )
        self._client_cache[cache_key] = client
        return client

    def _log_runtime_setup(self) -> None:
        logger.info(
            "[spec_smooth] setup draft_backend=%s draft_model=%s draft_use_main_client=%s draft_base_url=%s verifier_backend=%s verifier_model=%s",
            self.config.draft_backend,
            self.config.draft_model,
            self.config.draft_use_main_client,
            (self.config.draft_base_url or "<main_client>"),
            self.config.verifier_backend,
            (
                self.config.verifier_tree_model_name_or_path
                if self.config.verifier_backend.strip().lower() == "tree_attention"
                else self.config.verifier_model
            ),
        )
        if self.config.verifier_backend.strip().lower() == "tree_attention":
            logger.info(
                "[spec_smooth] tree verifier device=%s dtype=%s",
                self.config.verifier_tree_device,
                self.config.verifier_tree_dtype,
            )

    @staticmethod
    def _extract_latest_pending_tool_call(messages: list[Any]) -> dict[str, Any] | None:
        for msg in reversed(messages):
            tool_calls = SpeculativeSmoothingGuardElement._msg_get(msg, "tool_calls", None)
            if tool_calls:
                first = tool_calls[0]
                fn = SpeculativeSmoothingGuardElement._msg_get(first, "function", None)
                args = SpeculativeSmoothingGuardElement._msg_get(first, "args", {}) or {}
                name = fn if isinstance(fn, str) else SpeculativeSmoothingGuardElement._msg_get(first, "function", "unknown")
                if isinstance(first, dict):
                    if isinstance(first.get("function"), dict):
                        name = first["function"].get("name", name)
                        args = first["function"].get("arguments", args) or args
                return {
                    "tool_name": str(name),
                    "arguments": dict(args) if isinstance(args, dict) else {"raw_args": args},
                }
        return None

    @staticmethod
    def _extract_system_or_task_context(messages: list[Any]) -> str:
        for msg in messages:
            role = str(SpeculativeSmoothingGuardElement._msg_get(msg, "role", ""))
            if role == "system":
                return SpeculativeSmoothingGuardElement._content_to_text(
                    SpeculativeSmoothingGuardElement._msg_get(msg, "content", "")
                )
        return ""

    @staticmethod
    def _extract_untrusted_context(messages: list[Any]) -> str:
        # Use latest tool observation as untrusted external context.
        for msg in reversed(messages):
            role = str(SpeculativeSmoothingGuardElement._msg_get(msg, "role", ""))
            if role == "tool":
                return SpeculativeSmoothingGuardElement._content_to_text(
                    SpeculativeSmoothingGuardElement._msg_get(msg, "content", "")
                )
        return ""

    @staticmethod
    def _extract_recent_history(messages: list[Any], max_turns: int = 8) -> list[dict[str, Any]]:
        recent = messages[-max_turns:]
        out: list[dict[str, Any]] = []
        for msg in recent:
            out.append(
                {
                    "role": str(SpeculativeSmoothingGuardElement._msg_get(msg, "role", "unknown")),
                    "content": SpeculativeSmoothingGuardElement._content_to_text(
                        SpeculativeSmoothingGuardElement._msg_get(msg, "content", "")
                    ),
                }
            )
        return out

    @staticmethod
    def _clear_pending_tool_calls(messages: list[Any]) -> None:
        for msg in reversed(messages):
            tool_calls = SpeculativeSmoothingGuardElement._msg_get(msg, "tool_calls", None)
            if tool_calls:
                SpeculativeSmoothingGuardElement._msg_set(msg, "tool_calls", [])
                content = SpeculativeSmoothingGuardElement._msg_get(msg, "content", None)
                normalized = SpeculativeSmoothingGuardElement._normalize_content_blocks(content)
                if normalized is None:
                    normalized = [{"type": "text", "content": "[speculative_smoothing] blocked high-risk tool call."}]
                SpeculativeSmoothingGuardElement._msg_set(msg, "content", normalized)
                return

    @staticmethod
    def _normalize_content_blocks(content: Any) -> list[dict[str, Any]] | None:
        if content is None:
            return None
        if isinstance(content, list):
            out: list[dict[str, Any]] = []
            for item in content:
                if isinstance(item, dict) and ("content" in item or "text" in item):
                    txt = item.get("content", item.get("text", ""))
                    out.append({"type": "text", "content": str(txt) if txt is not None else ""})
                else:
                    out.append({"type": "text", "content": str(item)})
            return out
        if isinstance(content, dict):
            txt = content.get("content", content.get("text", ""))
            return [{"type": "text", "content": str(txt) if txt is not None else ""}]
        return [{"type": "text", "content": str(content)}]

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("content") is not None:
                        parts.append(str(item.get("content")))
                    elif item.get("text") is not None:
                        parts.append(str(item.get("text")))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        if isinstance(content, dict):
            if content.get("content") is not None:
                return str(content.get("content"))
            if content.get("text") is not None:
                return str(content.get("text"))
        return str(content)

    @staticmethod
    def _msg_get(msg: Any, key: str, default: Any) -> Any:
        if isinstance(msg, dict):
            return msg.get(key, default)
        return getattr(msg, key, default)

    @staticmethod
    def _msg_set(msg: Any, key: str, value: Any) -> None:
        if isinstance(msg, dict):
            msg[key] = value
            return
        try:
            setattr(msg, key, value)
        except Exception:
            # If immutable message object, silently skip.
            pass
