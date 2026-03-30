from __future__ import annotations

from dataclasses import dataclass
import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


@dataclass(slots=True)
class SpeculativeSmoothingConfig:
    """Runtime configuration for the first-version speculative smoothing defense."""

    top_k_lenses: int = 2
    branches_per_lens: int = 2
    seed: int = 42
    use_fresh_seed: bool = True
    disagreement_weight: float = 0.25
    block_threshold: float = 0.70
    escalate_threshold: float = 0.45
    cvar_alpha: float = 0.5
    aggregator: str = "cvar"
    lens_jitter: float = 0.15
    log_decisions: bool = True
    draft_backend: str = "llm"
    verifier_backend: str = "tree_attention"
    draft_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    draft_use_main_client: bool = False
    draft_base_url: str = "http://127.0.0.1:8000/v1"
    draft_api_key_env: str = "LOCAL_OPENAI_API_KEY"
    verifier_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    draft_temperature: float = 0.7
    verifier_temperature: float = 0.0
    draft_max_tokens: int = 24
    verifier_max_tokens: int = 256
    draft_top_logprobs_k: int = 8
    draft_noise_std: float = 0.35
    draft_branch_min_tokens: int = 8
    draft_branch_max_tokens: int = 12
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    verifier_tree_model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    verifier_tree_device: str = "auto"
    verifier_tree_dtype: str = "auto"
    verifier_tree_strict: bool = True

    @classmethod
    def from_env(cls) -> "SpeculativeSmoothingConfig":
        return cls(
            top_k_lenses=_env_int("SPEC_SMOOTHING_TOP_K_LENSES", 2),
            branches_per_lens=_env_int("SPEC_SMOOTHING_BRANCHES_PER_LENS", 2),
            seed=_env_int("SPEC_SMOOTHING_SEED", 42),
            use_fresh_seed=_env_bool("SPEC_SMOOTHING_USE_FRESH_SEED", True),
            disagreement_weight=_env_float("SPEC_SMOOTHING_DISAGREEMENT_WEIGHT", 0.25),
            block_threshold=_env_float("SPEC_SMOOTHING_BLOCK_THRESHOLD", 0.70),
            escalate_threshold=_env_float("SPEC_SMOOTHING_ESCALATE_THRESHOLD", 0.45),
            cvar_alpha=_env_float("SPEC_SMOOTHING_CVAR_ALPHA", 0.5),
            aggregator=_env_str("SPEC_SMOOTHING_AGGREGATOR", "cvar"),
            lens_jitter=_env_float("SPEC_SMOOTHING_LENS_JITTER", 0.15),
            log_decisions=_env_bool("SPEC_SMOOTHING_LOG_DECISIONS", True),
            draft_backend=_env_str("SPEC_SMOOTHING_DRAFT_BACKEND", "llm"),
            verifier_backend=_env_str("SPEC_SMOOTHING_VERIFIER_BACKEND", "tree_attention"),
            draft_model=_env_str("SPEC_SMOOTHING_DRAFT_MODEL", "meta-llama/Llama-3.2-3B-Instruct"),
            draft_use_main_client=_env_bool("SPEC_SMOOTHING_DRAFT_USE_MAIN_CLIENT", False),
            draft_base_url=_env_str("SPEC_SMOOTHING_DRAFT_BASE_URL", "http://127.0.0.1:8000/v1"),
            draft_api_key_env=_env_str("SPEC_SMOOTHING_DRAFT_API_KEY_ENV", "LOCAL_OPENAI_API_KEY"),
            verifier_model=_env_str("SPEC_SMOOTHING_VERIFIER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            draft_temperature=_env_float("SPEC_SMOOTHING_DRAFT_TEMPERATURE", 0.7),
            verifier_temperature=_env_float("SPEC_SMOOTHING_VERIFIER_TEMPERATURE", 0.0),
            draft_max_tokens=_env_int("SPEC_SMOOTHING_DRAFT_MAX_TOKENS", 24),
            verifier_max_tokens=_env_int("SPEC_SMOOTHING_VERIFIER_MAX_TOKENS", 256),
            draft_top_logprobs_k=_env_int("SPEC_SMOOTHING_DRAFT_TOP_LOGPROBS_K", 8),
            draft_noise_std=_env_float("SPEC_SMOOTHING_DRAFT_NOISE_STD", 0.35),
            draft_branch_min_tokens=_env_int("SPEC_SMOOTHING_DRAFT_BRANCH_MIN_TOKENS", 8),
            draft_branch_max_tokens=_env_int("SPEC_SMOOTHING_DRAFT_BRANCH_MAX_TOKENS", 12),
            api_key_env=_env_str("SPEC_SMOOTHING_API_KEY_ENV", _env_str("OPENROUTER_API_KEY_ENV", "OPENROUTER_API_KEY")),
            base_url=_env_str("SPEC_SMOOTHING_BASE_URL", _env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")),
            verifier_tree_model_name_or_path=_env_str(
                "SPEC_SMOOTHING_VERIFIER_TREE_MODEL_NAME_OR_PATH",
                "meta-llama/Llama-3.1-8B-Instruct",
            ),
            verifier_tree_device=_env_str("SPEC_SMOOTHING_VERIFIER_TREE_DEVICE", "auto"),
            verifier_tree_dtype=_env_str("SPEC_SMOOTHING_VERIFIER_TREE_DTYPE", "auto"),
            verifier_tree_strict=_env_bool("SPEC_SMOOTHING_VERIFIER_TREE_STRICT", True),
        )
