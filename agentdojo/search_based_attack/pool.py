from __future__ import annotations

import random
from typing import Iterable

from search_based_attack.schemas import CandidateRecord


class CandidatePool:
    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._records: list[CandidateRecord] = []
        self._norm_to_id: dict[str, str] = {}
        self._next_id = 0

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(text.split()).strip().lower()

    def has_text(self, text: str) -> bool:
        return self._norm(text) in self._norm_to_id

    def new_id(self) -> str:
        cid = f"cand_{self._next_id:06d}"
        self._next_id += 1
        return cid

    def add(self, record: CandidateRecord) -> bool:
        key = self._norm(record.text_trigger)
        if key in self._norm_to_id:
            return False
        self._records.append(record)
        self._norm_to_id[key] = record.candidate_id
        return True

    def __len__(self) -> int:
        return len(self._records)

    def all(self) -> list[CandidateRecord]:
        return list(self._records)

    def best(self) -> CandidateRecord | None:
        if not self._records:
            return None
        return max(self._records, key=lambda x: x.score.total_score)

    def top_k(self, k: int) -> list[CandidateRecord]:
        if k <= 0:
            return []
        return sorted(self._records, key=lambda x: x.score.total_score, reverse=True)[:k]

    def random_k(self, k: int) -> list[CandidateRecord]:
        if k <= 0 or not self._records:
            return []
        k = min(k, len(self._records))
        return self._rng.sample(self._records, k)

    def select_parents(self, top_k: int, random_k: int, parent_sample_k: int) -> list[CandidateRecord]:
        candidates = self.top_k(top_k)
        seen = {c.candidate_id for c in candidates}
        for item in self.random_k(random_k):
            if item.candidate_id not in seen:
                candidates.append(item)
                seen.add(item.candidate_id)

        if not candidates:
            return []
        if len(candidates) <= parent_sample_k:
            return candidates
        return self._rng.sample(candidates, parent_sample_k)

    def prune(self, max_pool_size: int) -> None:
        if len(self._records) <= max_pool_size:
            return
        kept = sorted(self._records, key=lambda x: x.score.total_score, reverse=True)[:max_pool_size]
        self._records = kept
        self._norm_to_id = {self._norm(rec.text_trigger): rec.candidate_id for rec in self._records}

    def extend(self, records: Iterable[CandidateRecord]) -> None:
        for record in records:
            self.add(record)
