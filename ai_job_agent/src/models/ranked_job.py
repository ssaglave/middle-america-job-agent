"""RankedJob dataclass — wraps a Job with scoring breakdown and explanation."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ai_job_agent.src.models.job import Job


@dataclass
class RankedJob:
    """A Job augmented with ranking scores and a human-readable explanation.

    The ``explanation`` field is specifically required for the demo:
    e.g. "72% skill match | preferred state (IA) | posted 3 days ago"
    """

    job: Job
    total_score: float       # 0–100 weighted composite
    skill_score: float       # 0–100 raw skill sub-score
    location_score: float    # 0–100 raw location sub-score
    recency_score: float     # 0–100 raw recency sub-score
    explanation: str         # human-readable summary for demo narration

    def to_dict(self) -> dict:
        return {
            **self.job.to_dict(),
            "total_score": round(self.total_score, 2),
            "skill_score": round(self.skill_score, 2),
            "location_score": round(self.location_score, 2),
            "recency_score": round(self.recency_score, 2),
            "explanation": self.explanation,
        }
