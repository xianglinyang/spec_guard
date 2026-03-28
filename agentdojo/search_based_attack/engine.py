from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from search_based_attack.logging_io import RunLogger
from search_based_attack.pool import CandidatePool
from search_based_attack.schemas import CandidateRecord, IterationState
from search_based_attack.scorer import compute_score


@dataclass
class PairConfig:
    max_iterations: int
    children_per_iteration: int
    max_pool_size: int


@dataclass
class SboaConfig:
    max_iterations: int
    top_k: int
    random_k: int
    parent_sample_k: int
    max_pool_size: int


@dataclass
class TapConfig:
    max_iterations: int
    max_depth: int
    branching_width: int
    max_pool_size: int


@dataclass
class AutoDanConfig:
    max_iterations: int
    population_size: int
    num_elites: int
    mutation_rate: float
    crossover_rate: float
    children_per_generation: int
    crossover_prompt_template: str | None
    max_pool_size: int


class SearchEngine:
    def __init__(
        self,
        *,
        pool: CandidatePool,
        logger: RunLogger,
        episode_runner,
        critic,
        mutator,
        scoring_cfg,
        rng: random.Random,
        baseline: str,
    ) -> None:
        self.pool = pool
        self.logger = logger
        self.episode_runner = episode_runner
        self.critic = critic
        self.mutator = mutator
        self.scoring_cfg = scoring_cfg
        self.rng = rng
        self.baseline = baseline

    def evaluate_and_add(
        self,
        *,
        trigger: str,
        iteration: int,
        parent_ids: list[str],
        metadata: dict | None = None,
    ) -> CandidateRecord | None:
        if self.pool.has_text(trigger):
            return None

        episode = self.episode_runner.run_one_episode(trigger)
        critic_score, critic_reason = self.critic.score(trigger, episode)
        score = compute_score(episode, critic_score, self.scoring_cfg)
        rec = CandidateRecord(
            candidate_id=self.pool.new_id(),
            text_trigger=trigger,
            iteration_created=iteration,
            parent_ids=parent_ids,
            score=score,
            episode=episode,
            critic_reason=critic_reason,
            metadata=metadata or {},
        )
        if self.pool.add(rec):
            self.logger.log_candidate(rec)
            return rec
        return None

    def _seed(self, init_trigger: str) -> CandidateRecord:
        rec = self.evaluate_and_add(
            trigger=init_trigger,
            iteration=0,
            parent_ids=[],
            metadata={"baseline": self.baseline, "phase": "seed"},
        )
        if rec is None:
            best = self.pool.best()
            if best is None:
                raise RuntimeError("failed to add seed candidate")
            return best
        return rec

    def run_pair(self, init_trigger: str, cfg: PairConfig) -> CandidateRecord:
        seed = self._seed(init_trigger)
        if seed.episode.success:
            self.logger.log_iteration(
                IterationState(
                    iteration=0,
                    baseline=self.baseline,
                    selected_parent_ids=[],
                    new_candidate_ids=[seed.candidate_id],
                    best_candidate_id=seed.candidate_id,
                    best_total_score=seed.score.total_score,
                    stop_reason="success",
                )
            )
            return seed

        for i in range(1, cfg.max_iterations + 1):
            parent = self.pool.best()
            if parent is None:
                break
            parent_ids = [parent.candidate_id]
            new_ids: list[str] = []
            for _ in range(cfg.children_per_iteration):
                child = self.mutator.mutate_single(parent)
                rec = self.evaluate_and_add(
                    trigger=child,
                    iteration=i,
                    parent_ids=parent_ids,
                    metadata={"baseline": self.baseline, "depth": i},
                )
                if rec is not None:
                    new_ids.append(rec.candidate_id)
                    if rec.episode.success:
                        best = self.pool.best()
                        self.logger.log_iteration(
                            IterationState(
                                iteration=i,
                                baseline=self.baseline,
                                selected_parent_ids=parent_ids,
                                new_candidate_ids=new_ids,
                                best_candidate_id=best.candidate_id if best else None,
                                best_total_score=best.score.total_score if best else None,
                                stop_reason="success",
                            )
                        )
                        return rec

            self.pool.prune(cfg.max_pool_size)
            best = self.pool.best()
            self.logger.log_iteration(
                IterationState(
                    iteration=i,
                    baseline=self.baseline,
                    selected_parent_ids=parent_ids,
                    new_candidate_ids=new_ids,
                    best_candidate_id=best.candidate_id if best else None,
                    best_total_score=best.score.total_score if best else None,
                )
            )

        best = self.pool.best()
        if best is None:
            return seed
        return best

    def run_sboa(self, init_trigger: str, cfg: SboaConfig) -> CandidateRecord:
        seed = self._seed(init_trigger)
        if seed.episode.success:
            self.logger.log_iteration(
                IterationState(
                    iteration=0,
                    baseline=self.baseline,
                    selected_parent_ids=[],
                    new_candidate_ids=[seed.candidate_id],
                    best_candidate_id=seed.candidate_id,
                    best_total_score=seed.score.total_score,
                    stop_reason="success",
                )
            )
            return seed

        for i in range(1, cfg.max_iterations + 1):
            parents = self.pool.select_parents(
                top_k=cfg.top_k,
                random_k=cfg.random_k,
                parent_sample_k=cfg.parent_sample_k,
            )
            if not parents:
                break

            parent_ids = [p.candidate_id for p in parents]
            children = self.mutator.mutate(parents)
            new_ids: list[str] = []
            for child in children:
                rec = self.evaluate_and_add(
                    trigger=child,
                    iteration=i,
                    parent_ids=parent_ids,
                    metadata={"baseline": self.baseline},
                )
                if rec is not None:
                    new_ids.append(rec.candidate_id)
                    if rec.episode.success:
                        best = self.pool.best()
                        self.logger.log_iteration(
                            IterationState(
                                iteration=i,
                                baseline=self.baseline,
                                selected_parent_ids=parent_ids,
                                new_candidate_ids=new_ids,
                                best_candidate_id=best.candidate_id if best else None,
                                best_total_score=best.score.total_score if best else None,
                                stop_reason="success",
                            )
                        )
                        return rec

            self.pool.prune(cfg.max_pool_size)
            best = self.pool.best()
            self.logger.log_iteration(
                IterationState(
                    iteration=i,
                    baseline=self.baseline,
                    selected_parent_ids=parent_ids,
                    new_candidate_ids=new_ids,
                    best_candidate_id=best.candidate_id if best else None,
                    best_total_score=best.score.total_score if best else None,
                )
            )

        best = self.pool.best()
        if best is None:
            return seed
        return best

    def run_tap(self, init_trigger: str, cfg: TapConfig) -> CandidateRecord:
        seed = self._seed(init_trigger)
        if seed.episode.success:
            self.logger.log_iteration(
                IterationState(
                    iteration=0,
                    baseline=self.baseline,
                    selected_parent_ids=[],
                    new_candidate_ids=[seed.candidate_id],
                    best_candidate_id=seed.candidate_id,
                    best_total_score=seed.score.total_score,
                    stop_reason="success",
                )
            )
            return seed

        frontier: deque[tuple[CandidateRecord, int]] = deque([(seed, 0)])
        iter_count = 0
        while frontier and iter_count < cfg.max_iterations:
            parent, depth = frontier.popleft()
            if depth >= cfg.max_depth:
                continue

            iter_count += 1
            parent_ids = [parent.candidate_id]
            new_ids: list[str] = []

            for _ in range(cfg.branching_width):
                child = self.mutator.mutate_single(parent)
                rec = self.evaluate_and_add(
                    trigger=child,
                    iteration=iter_count,
                    parent_ids=parent_ids,
                    metadata={"baseline": self.baseline, "depth": depth + 1},
                )
                if rec is None:
                    continue

                new_ids.append(rec.candidate_id)
                frontier.append((rec, depth + 1))
                if rec.episode.success:
                    best = self.pool.best()
                    self.logger.log_iteration(
                        IterationState(
                            iteration=iter_count,
                            baseline=self.baseline,
                            selected_parent_ids=parent_ids,
                            new_candidate_ids=new_ids,
                            best_candidate_id=best.candidate_id if best else None,
                            best_total_score=best.score.total_score if best else None,
                            depth=depth + 1,
                            stop_reason="success",
                        )
                    )
                    return rec

            self.pool.prune(cfg.max_pool_size)
            best = self.pool.best()
            self.logger.log_iteration(
                IterationState(
                    iteration=iter_count,
                    baseline=self.baseline,
                    selected_parent_ids=parent_ids,
                    new_candidate_ids=new_ids,
                    best_candidate_id=best.candidate_id if best else None,
                    best_total_score=best.score.total_score if best else None,
                    depth=depth + 1,
                )
            )

        best = self.pool.best()
        if best is None:
            return seed
        return best

    def _roulette_select(self, candidates: list[CandidateRecord], k: int) -> list[CandidateRecord]:
        if not candidates or k <= 0:
            return []
        scores = [max(0.0, c.score.total_score) for c in candidates]
        total = sum(scores)
        if total <= 0:
            return [self.rng.choice(candidates) for _ in range(k)]

        out: list[CandidateRecord] = []
        for _ in range(k):
            r = self.rng.random() * total
            acc = 0.0
            chosen = candidates[-1]
            for cand, s in zip(candidates, scores):
                acc += s
                if acc >= r:
                    chosen = cand
                    break
            out.append(chosen)
        return out

    def _build_population(self, seed: CandidateRecord, population_size: int) -> list[CandidateRecord]:
        population: list[CandidateRecord] = [seed]
        retries = 0
        max_retries = max(10, population_size * 4)
        while len(population) < population_size and retries < max_retries:
            parent = self.rng.choice(population)
            child = self.mutator.mutate_single(parent)
            rec = self.evaluate_and_add(
                trigger=child,
                iteration=0,
                parent_ids=[parent.candidate_id],
                metadata={"baseline": self.baseline, "phase": "population_init"},
            )
            if rec is not None:
                population.append(rec)
                if rec.episode.success:
                    break
            else:
                retries += 1
        return population

    def run_autodan(self, init_trigger: str, cfg: AutoDanConfig) -> CandidateRecord:
        seed = self._seed(init_trigger)
        if seed.episode.success:
            self.logger.log_iteration(
                IterationState(
                    iteration=0,
                    baseline=self.baseline,
                    selected_parent_ids=[],
                    new_candidate_ids=[seed.candidate_id],
                    best_candidate_id=seed.candidate_id,
                    best_total_score=seed.score.total_score,
                    stop_reason="success",
                )
            )
            return seed

        population = self._build_population(seed, cfg.population_size)
        if any(c.episode.success for c in population):
            best = max(population, key=lambda x: x.score.total_score)
            return best

        for gen in range(1, cfg.max_iterations + 1):
            ranked = sorted(population, key=lambda x: x.score.total_score, reverse=True)
            elites = ranked[: max(1, min(cfg.num_elites, len(ranked)))]

            needed = max(0, cfg.children_per_generation)
            parent_count = max(2, needed)
            parents = self._roulette_select(population, parent_count)
            parent_ids = [p.candidate_id for p in parents]

            offspring_triggers: list[str] = []
            for i in range(0, len(parents) - 1, 2):
                p1 = parents[i]
                p2 = parents[i + 1]
                if self.rng.random() < cfg.crossover_rate:
                    c1, c2 = self.mutator.llm_crossover(
                        p1,
                        p2,
                        prompt_template=cfg.crossover_prompt_template,
                    )
                else:
                    c1 = p1.text_trigger
                    c2 = p2.text_trigger

                if self.rng.random() < cfg.mutation_rate:
                    c1 = self.mutator.mutate_single_text(c1, [p1])
                if self.rng.random() < cfg.mutation_rate:
                    c2 = self.mutator.mutate_single_text(c2, [p2])

                offspring_triggers.extend([c1, c2])
                if len(offspring_triggers) >= needed:
                    break

            offspring_triggers = offspring_triggers[:needed]
            new_ids: list[str] = []
            evaluated_children: list[CandidateRecord] = []

            for trig in offspring_triggers:
                rec = self.evaluate_and_add(
                    trigger=trig,
                    iteration=gen,
                    parent_ids=parent_ids,
                    metadata={"baseline": self.baseline, "generation": gen},
                )
                if rec is not None:
                    new_ids.append(rec.candidate_id)
                    evaluated_children.append(rec)
                    if rec.episode.success:
                        best = self.pool.best()
                        self.logger.log_iteration(
                            IterationState(
                                iteration=gen,
                                baseline=self.baseline,
                                selected_parent_ids=parent_ids,
                                new_candidate_ids=new_ids,
                                best_candidate_id=best.candidate_id if best else None,
                                best_total_score=best.score.total_score if best else None,
                                stop_reason="success",
                            )
                        )
                        return rec

            population = (elites + evaluated_children)[: cfg.population_size]
            if not population:
                population = elites
            self.pool.prune(cfg.max_pool_size)
            best = self.pool.best()
            self.logger.log_iteration(
                IterationState(
                    iteration=gen,
                    baseline=self.baseline,
                    selected_parent_ids=parent_ids,
                    new_candidate_ids=new_ids,
                    best_candidate_id=best.candidate_id if best else None,
                    best_total_score=best.score.total_score if best else None,
                )
            )

        best = self.pool.best()
        if best is None:
            return seed
        return best
