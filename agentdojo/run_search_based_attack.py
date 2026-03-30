#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import random

import openai
from dotenv import load_dotenv

from agentdojo.task_suite.load_suites import get_suite
from search_based_attack.config import load_config
from search_based_attack.critic import Critic
from search_based_attack.episode import EpisodeRunner
from search_based_attack.logging_io import RunLogger
from search_based_attack.mutator import Mutator
from search_based_attack.pipeline_factory import build_pipeline
from search_based_attack.pool import CandidatePool
from search_based_attack.scorer import compute_score
from search_based_attack.schemas import CandidateRecord, IterationState
from yaml_injection_patch import apply_agentdojo_yaml_injection_patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run search-based optimization attack on AgentDojo.")
    parser.add_argument("--config", default="search_based_attack_config.yaml")
    parser.add_argument("--iterations", type=int, default=None, help="Override max iterations from config")
    parser.add_argument(
        "--single-trigger",
        type=str,
        default=None,
        help="Run one episode with this trigger, log it, and optionally continue search.",
    )
    return parser.parse_args()


def _evaluate_and_add_candidate(
    *,
    pool: CandidatePool,
    logger: RunLogger,
    episode_runner: EpisodeRunner,
    critic: Critic,
    trigger: str,
    iteration: int,
    parent_ids: list[str],
    scoring_cfg,
) -> str | None:
    if pool.has_text(trigger):
        return None

    episode = episode_runner.run_one_episode(trigger)
    critic_score, critic_reason = critic.score(trigger, episode)
    score = compute_score(episode, critic_score, scoring_cfg)
    rec = CandidateRecord(
        candidate_id=pool.new_id(),
        text_trigger=trigger,
        iteration_created=iteration,
        parent_ids=parent_ids,
        score=score,
        episode=episode,
        critic_reason=critic_reason,
    )
    added = pool.add(rec)
    if added:
        logger.log_candidate(rec)
        return rec.candidate_id
    return None


def main() -> int:
    args = parse_args()
    apply_agentdojo_yaml_injection_patch()
    cfg = load_config(args.config)

    load_dotenv(".env")
    for module in cfg.runtime.module_to_load:
        importlib.import_module(module)

    random.seed(cfg.experiment.seed)

    api_key = os.getenv(cfg.openrouter.api_key_env) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"Missing API key. Set {cfg.openrouter.api_key_env} (preferred) or OPENAI_API_KEY in environment/.env."
        )

    pipeline = build_pipeline(
        openrouter_model=cfg.openrouter.model,
        model_alias=cfg.openrouter.model_alias,
        base_url=cfg.openrouter.base_url,
        api_key=api_key,
        defense=cfg.benchmark.defense,
        system_message_name=cfg.benchmark.system_message_name,
        system_message=cfg.benchmark.system_message,
        tool_output_format=cfg.benchmark.tool_output_format,
    )

    suite = get_suite(cfg.benchmark.benchmark_version, cfg.benchmark.suite)
    user_task = suite.get_user_task_by_id(cfg.benchmark.user_task)
    injection_task = suite.get_injection_task_by_id(cfg.benchmark.injection_task)

    episode_runner = EpisodeRunner(
        suite=suite,
        pipeline=pipeline,
        user_task=user_task,
        injection_task=injection_task,
    )

    client = openai.OpenAI(api_key=api_key, base_url=cfg.openrouter.base_url)
    mutator = Mutator(client, cfg.mutation)
    critic = Critic(client, cfg.critic)

    logger = RunLogger(cfg.logging.out_dir, cfg.experiment.name)
    pool = CandidatePool(seed=cfg.experiment.seed)

    # seed pool with configured candidates or explicit single trigger
    seed_candidates = [args.single_trigger] if args.single_trigger else list(cfg.pool.init_candidates)
    for trig in seed_candidates:
        _evaluate_and_add_candidate(
            pool=pool,
            logger=logger,
            episode_runner=episode_runner,
            critic=critic,
            trigger=trig,
            iteration=0,
            parent_ids=[],
            scoring_cfg=cfg.scoring,
        )

    max_iters = args.iterations if args.iterations is not None else cfg.experiment.max_iterations

    for i in range(1, max_iters + 1):
        parents = pool.select_parents(
            top_k=cfg.pool.top_k,
            random_k=cfg.pool.exploration_random_k,
            parent_sample_k=cfg.pool.parent_sample_k,
        )
        if not parents:
            break

        parent_ids = [p.candidate_id for p in parents]
        children = mutator.mutate(parents)

        new_ids: list[str] = []
        for child in children:
            cid = _evaluate_and_add_candidate(
                pool=pool,
                logger=logger,
                episode_runner=episode_runner,
                critic=critic,
                trigger=child,
                iteration=i,
                parent_ids=parent_ids,
                scoring_cfg=cfg.scoring,
            )
            if cid is not None:
                new_ids.append(cid)

        pool.prune(cfg.pool.max_pool_size)
        best = pool.best()
        logger.log_iteration(
            IterationState(
                iteration=i,
                baseline="sboa",
                selected_parent_ids=parent_ids,
                new_candidate_ids=new_ids,
                best_candidate_id=best.candidate_id if best else None,
                best_total_score=best.score.total_score if best else None,
            )
        )

    best = pool.best()
    logger.write_summary(
        {
            "run_dir": str(logger.run_dir),
            "total_candidates": len(pool),
            "best_candidate": (
                {
                    "candidate_id": best.candidate_id,
                    "text_trigger": best.text_trigger,
                    "total_score": best.score.total_score,
                    "success": best.episode.success,
                    "utility": best.episode.utility,
                }
                if best
                else None
            ),
        }
    )

    print(f"[done] search-based optimization attack run saved to: {logger.run_dir}")
    if best:
        print(f"[best] {best.candidate_id} score={best.score.total_score:.3f} success={best.episode.success}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
