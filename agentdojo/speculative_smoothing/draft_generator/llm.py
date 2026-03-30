from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import random
import re
from typing import Any

from speculative_smoothing.interfaces import DraftBranchGenerator
from speculative_smoothing.schemas import BranchDraft, EvidenceSpan, GuardrailState, SafetyLens, VerdictLabel


@dataclass(slots=True)
class LLMDraftBranchGenerator(DraftBranchGenerator):
    """LLM draft branch generator with input-side context randomization.

    Each branch receives a differently randomized view of the untrusted context
    (chunk reordering, random start offset) conditioned on its unique seed.
    All K branch requests are dispatched concurrently via a ThreadPoolExecutor,
    allowing a vLLM server's continuous batching to process them in one batch.
    """

    client: Any
    model: str
    temperature: float = 0.7
    max_tokens: int = 24
    min_branch_tokens: int = 8
    max_branch_tokens: int = 12
    fallback: DraftBranchGenerator | None = None

    def generate(
        self,
        state: GuardrailState,
        lenses: list[SafetyLens],
        branches_per_lens: int,
        seed: int | None = None,
    ) -> list[BranchDraft]:
        if branches_per_lens <= 0 or not lenses:
            return []

        # Build per-branch task descriptors up front.
        tasks: list[tuple[SafetyLens, int, int, int | None]] = []
        for lens_idx, lens in enumerate(lenses):
            for branch_idx in range(branches_per_lens):
                branch_seed = self._branch_seed(seed, lens_idx, branch_idx)
                tasks.append((lens, lens_idx, branch_idx, branch_seed))

        results: list[BranchDraft | None] = [None] * len(tasks)

        def _run(item: tuple[int, tuple[SafetyLens, int, int, int | None]]) -> tuple[int, BranchDraft]:
            pos, (lens, lens_idx, branch_idx, branch_seed) = item
            draft = self._generate_one_branch(state, lens, branch_idx, branch_seed)
            return pos, draft

        try:
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                futures = {executor.submit(_run, (i, task)): i for i, task in enumerate(tasks)}
                for future in as_completed(futures):
                    pos, draft = future.result()
                    results[pos] = draft
        except Exception:
            if self.fallback is not None:
                return self.fallback.generate(state, lenses, branches_per_lens, seed=seed)
            raise

        return [r for r in results if r is not None]

    def _generate_one_branch(
        self,
        state: GuardrailState,
        lens: SafetyLens,
        branch_idx: int,
        branch_seed: int | None,
    ) -> BranchDraft:
        rng = random.Random(branch_seed)
        ctx = self._randomize_context_view(state.untrusted_context or "", rng)
        messages = self._build_messages(state, lens, ctx, self.min_branch_tokens, self.max_branch_tokens)
        choice = self._chat_completion_choice(messages, seed=branch_seed)
        raw_text = self._extract_choice_text(choice)
        branch_text = self._normalize_branch_text(raw_text, lens=lens, state=state)
        evidence = self._extract_evidence(branch_text) or self._fallback_evidence(state)
        verdict = self._extract_verdict(branch_text)

        return BranchDraft(
            lens=lens,
            text=branch_text,
            branch_id=f"{lens.value}_{branch_idx}",
            evidence_span=EvidenceSpan(source="untrusted_context", text=evidence),
            verdict_label=verdict,
            generation_metadata={
                "generator": "LLMDraftBranchGenerator",
                "raw": raw_text[:400],
                "context_randomized": ctx != (state.untrusted_context or ""),
            },
            seed=branch_seed,
            randomness_metadata={"context_seed": branch_seed},
        )

    def _chat_completion_choice(self, messages: list[dict[str, str]], seed: int | None = None) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs.pop("seed", None)
            resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0]

    @staticmethod
    def _randomize_context_view(text: str, rng: random.Random) -> str:
        """Return a randomly reordered chunk view of the untrusted context.

        Splits the text into sentence-level chunks, shuffles them, and optionally
        drops a leading chunk to vary the start offset. This creates semantically
        distinct views of the same input across branches without altering content.
        """
        if not text or len(text) < 80:
            return text

        chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", text.strip()) if c.strip()]
        if len(chunks) <= 1:
            return text

        rng.shuffle(chunks)

        # With 30% probability, drop the first chunk to vary the start.
        if len(chunks) > 2 and rng.random() < 0.3:
            chunks = chunks[1:]

        return " ".join(chunks)

    @staticmethod
    def _build_messages(
        state: GuardrailState,
        lens: SafetyLens,
        ctx: str,
        min_branch_tokens: int,
        max_branch_tokens: int,
    ) -> list[dict[str, str]]:
        prompt = (
            "You are drafting ONE short safety-analysis branch before a tool call.\n"
            f"Lens: {lens.value}\n"
            f"Length target: {min_branch_tokens}-{max_branch_tokens} tokens.\n"
            "Output exactly one line and follow this template strictly:\n"
            f"[LENS:{lens.value}] [EVIDENCE:<very short>] [VERDICT:<safe|unsafe|uncertain>]\n\n"
            f"User goal: {state.user_goal}\n"
            f"Task/system context: {state.task_context or state.system_instructions}\n"
            f"Untrusted context: {ctx}\n"
            f"Proposed tool: {state.proposed_tool_call.tool_name}\n"
            f"Proposed args: {state.proposed_tool_call.arguments}\n"
            "Do not output anything except that single line."
        )
        return [
            {"role": "system", "content": "You are a concise safety analyst."},
            {"role": "user", "content": prompt},
        ]

    @staticmethod
    def _extract_choice_text(choice: Any) -> str:
        message = LLMDraftBranchGenerator._obj_get(choice, "message", None)
        content = LLMDraftBranchGenerator._obj_get(message, "content", "")
        if content is None:
            return ""
        return str(content).strip()

    def _normalize_branch_text(self, text: str, lens: SafetyLens, state: GuardrailState) -> str:
        compact = " ".join((text or "").split()).strip()
        verdict = self._extract_verdict(compact)
        evidence = self._extract_evidence(compact) or self._fallback_evidence(state)

        if "[LENS:" not in compact or "[EVIDENCE:" not in compact or "[VERDICT:" not in compact:
            verdict_str = verdict.value if verdict is not None else VerdictLabel.UNCERTAIN.value
            compact = f"[LENS:{lens.value}] [EVIDENCE:{evidence}] [VERDICT:{verdict_str}]"

        words = compact.split()
        if len(words) > self.max_branch_tokens:
            compact = " ".join(words[: self.max_branch_tokens]).strip()

        return compact

    @staticmethod
    def _extract_evidence(text: str) -> str | None:
        m = re.search(r"\[EVIDENCE:(.*?)\]", text, re.IGNORECASE)
        if not m:
            return None
        return m.group(1).strip() or None

    @staticmethod
    def _extract_verdict(text: str) -> VerdictLabel | None:
        m = re.search(r"\[VERDICT:(.*?)\]", text, re.IGNORECASE)
        if not m:
            return None
        value = m.group(1).strip().lower()
        if value == VerdictLabel.SAFE.value:
            return VerdictLabel.SAFE
        if value == VerdictLabel.UNSAFE.value:
            return VerdictLabel.UNSAFE
        if value == VerdictLabel.UNCERTAIN.value:
            return VerdictLabel.UNCERTAIN
        return None

    @staticmethod
    def _fallback_evidence(state: GuardrailState) -> str:
        text = " ".join((state.untrusted_context or "").split()).strip()
        if len(text) <= 64:
            return text
        return f"{text[:64].rstrip()}..."

    @staticmethod
    def _obj_get(obj: Any, key: str, default: Any) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _branch_seed(seed: int | None, lens_idx: int, branch_idx: int) -> int | None:
        if seed is None:
            return None
        return int(seed) + lens_idx * 10_000 + branch_idx
