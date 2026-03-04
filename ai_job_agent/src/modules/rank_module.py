"""Rank module: scores filtered jobs on skill match, location, and recency.

Scoring formula (weights from rank_config.yaml):
    total = w_skill * skill_score
          + w_loc   * location_score
          + w_rec   * recency_score

Each sub-score is in [0, 100].

Output:
    - Top-N RankedJob objects sorted by total_score descending.
    - data/processed/ranked_jobs_<date>.json   — top-N results
    - data/processed/rank_trace_<date>.json    — full per-job score breakdown
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from ai_job_agent.src.models.job import Job
from ai_job_agent.src.models.ranked_job import RankedJob
from ai_job_agent.src.utils.logger import setup_logger
from ai_job_agent.src.utils.storage import save_json

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


class RankModule:
    """Ranks a list of filtered jobs and returns the top-N.

    Args:
        user_skills: List of skill strings from the user/sample resume,
                     e.g. ["python", "tensorflow", "mlflow", "aws"].
    """

    def __init__(self, user_skills: Optional[List[str]] = None) -> None:
        self._logger: logging.Logger = setup_logger("rank_module")
        self._rank_cfg = self._load_yaml("rank_config.yaml")
        self._loc_cfg = self._load_yaml("locations.yaml")
        self._user_skills: List[str] = [s.lower() for s in (user_skills or [])]

    # ── config ────────────────────────────────────────────────────────────────

    def _load_yaml(self, filename: str) -> dict:
        path = _CONFIG_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ── sub-scorers ───────────────────────────────────────────────────────────

    def _skill_score(self, job: Job) -> float:
        """Jaccard similarity between job skills and user skills × 100.

        Combines required_skills and preferred_skills from the job.
        Returns 0 if neither side has any skills.
        """
        job_skills = {s.lower() for s in job.required_skills + job.preferred_skills}
        user_skills = set(self._user_skills)

        if not job_skills or not user_skills:
            return 0.0

        intersection = job_skills & user_skills
        union = job_skills | user_skills
        return (len(intersection) / len(union)) * 100

    def _location_score(self, job: Job) -> float:
        """Score based on location tier defined in rank_config.yaml."""
        loc_scores = self._rank_cfg.get("location_scores", {})
        loc = job.location.lower()

        if "remote" in loc:
            return float(loc_scores.get("remote", 100))
        if "hybrid" in loc:
            return float(loc_scores.get("hybrid", 90))

        preferred = self._loc_cfg.get("preferred_states", [])
        for state in preferred:
            if f", {state}" in job.location or f" {state}" in job.location:
                return float(loc_scores.get("preferred_state", 80))

        middle_america = self._loc_cfg.get("middle_america_states", [])
        for state in middle_america:
            if f", {state}" in job.location or f" {state}" in job.location:
                return float(loc_scores.get("middle_america", 60))

        return float(loc_scores.get("other", 20))

    def _recency_score(self, job: Job) -> float:
        """Exponential decay score based on posting age.

        Parses strings like "3 days ago", "2 weeks ago", "1 month ago".
        Unknown/missing dates score 50 (neutral).
        """
        decay_days = self._rank_cfg.get("recency_decay_days", 30)
        days_old = self._parse_days_old(job.posted_date)
        if days_old is None:
            return 50.0
        return 100.0 * math.exp(-days_old / decay_days)

    @staticmethod
    def _parse_days_old(posted_date: Optional[str]) -> Optional[int]:
        """Convert a relative date string to an integer number of days.

        Handles:
            "just now" / "today"   → 0
            "X hour(s) ago"        → 0
            "X day(s) ago"         → X
            "X week(s) ago"        → X * 7
            "X month(s) ago"       → X * 30
        Returns None if the string cannot be parsed.
        """
        if not posted_date:
            return None

        text = posted_date.lower().strip()

        if text in ("just now", "today"):
            return 0

        match = re.search(r"(\d+)\s*(hour|day|week|month)", text)
        if not match:
            return None

        value = int(match.group(1))
        unit = match.group(2)

        multipliers = {"hour": 0, "day": 1, "week": 7, "month": 30}
        return value * multipliers[unit]

    # ── explanation builder ───────────────────────────────────────────────────

    def _build_explanation(
        self,
        job: Job,
        skill_score: float,
        location_score: float,
        recency_score: float,
        days_old: Optional[int],
    ) -> str:
        """Build the human-readable explanation string required for the demo.

        Example: "72% skill match | preferred state (IA) | posted 3 days ago"
        """
        # Skill part
        skill_pct = f"{round(skill_score)}% skill match"

        # Location part
        loc = job.location.lower()
        if "remote" in loc:
            loc_label = "remote"
        elif "hybrid" in loc:
            loc_label = "hybrid"
        else:
            preferred = self._loc_cfg.get("preferred_states", [])
            middle = self._loc_cfg.get("middle_america_states", [])
            matched_state = None
            for state in preferred + middle:
                if f", {state}" in job.location or f" {state}" in job.location:
                    matched_state = state
                    break
            if matched_state and matched_state in preferred:
                loc_label = f"preferred state ({matched_state})"
            elif matched_state:
                loc_label = f"middle america ({matched_state})"
            else:
                loc_label = job.location

        # Recency part
        if days_old is None:
            recency_label = "posting date unknown"
        elif days_old == 0:
            recency_label = "posted today"
        elif days_old == 1:
            recency_label = "posted 1 day ago"
        else:
            recency_label = f"posted {days_old} days ago"

        return f"{skill_pct} | {loc_label} | {recency_label}"

    # ── public API ────────────────────────────────────────────────────────────

    def rank(self, jobs: List[Job]) -> List[RankedJob]:
        """Score and rank jobs; return top-N RankedJob objects.

        Writes:
            data/processed/ranked_jobs_<date>.json  — top-N results
            data/processed/rank_trace_<date>.json   — all jobs with scores

        Args:
            jobs: Filtered list of Job objects.

        Returns:
            Top-N RankedJob objects sorted by total_score descending.
        """
        self._logger.info(
            "Rank stage started | input_jobs=%d | user_skills=%s",
            len(jobs),
            self._user_skills,
        )

        weights = self._rank_cfg.get("weights", {})
        w_skill = weights.get("skill_match", 0.5)
        w_loc = weights.get("location", 0.3)
        w_rec = weights.get("recency", 0.2)
        top_n = self._rank_cfg.get("top_n", 10)

        ranked: List[RankedJob] = []
        trace: List[dict] = []

        for job in jobs:
            skill_s = self._skill_score(job)
            loc_s = self._location_score(job)
            rec_s = self._recency_score(job)
            total = w_skill * skill_s + w_loc * loc_s + w_rec * rec_s
            days_old = self._parse_days_old(job.posted_date)
            explanation = self._build_explanation(
                job, skill_s, loc_s, rec_s, days_old
            )

            rj = RankedJob(
                job=job,
                total_score=round(total, 2),
                skill_score=round(skill_s, 2),
                location_score=round(loc_s, 2),
                recency_score=round(rec_s, 2),
                explanation=explanation,
            )
            ranked.append(rj)

            trace.append(
                {
                    "job_id": job.job_id,
                    "title": job.title,
                    "company": job.company_name,
                    "location": job.location,
                    "total_score": round(total, 2),
                    "skill_score": round(skill_s, 2),
                    "location_score": round(loc_s, 2),
                    "recency_score": round(rec_s, 2),
                    "explanation": explanation,
                }
            )

            self._logger.debug(
                "Scored | %s @ %s | total=%.1f skill=%.1f loc=%.1f rec=%.1f",
                job.title,
                job.company_name,
                total,
                skill_s,
                loc_s,
                rec_s,
            )

        # Sort descending by total score
        ranked.sort(key=lambda r: r.total_score, reverse=True)
        top_ranked = ranked[:top_n]

        self._logger.info(
            "Rank stage complete | scored=%d | returning top_%d",
            len(ranked),
            len(top_ranked),
        )

        today = datetime.now().strftime("%Y-%m-%d")

        # Save top-N ranked jobs
        ranked_path = _DATA_DIR / "processed" / f"ranked_jobs_{today}.json"
        save_json([r.to_dict() for r in top_ranked], ranked_path)
        self._logger.info("Saved top-%d ranked jobs to %s", len(top_ranked), ranked_path)

        # Save full trace for agent trace appendix
        trace_path = _DATA_DIR / "processed" / f"rank_trace_{today}.json"
        save_json(trace, trace_path)
        self._logger.info("Saved rank trace (%d entries) to %s", len(trace), trace_path)

        return top_ranked
