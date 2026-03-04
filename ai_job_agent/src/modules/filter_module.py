"""Filter module: removes FAANG+, startups, and off-location jobs.

Each rejection is logged with an explicit reason so the full agent trace
(required by the assignment report appendix) can be reconstructed from
data/processed/filter_trace_<date>.json.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ai_job_agent.src.models.job import Job
from ai_job_agent.src.utils.logger import setup_logger
from ai_job_agent.src.utils.storage import save_json

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


class FilterModule:
    """Applies three sequential filters to a list of Job objects.

    Filters (applied in order):
        1. Company blacklist — FAANG+ and big-tech exclusions.
        2. Startup heuristic — description keyword signals.
        3. Location preference — preferred states, remote, hybrid.

    Args:
        toggles: Runtime overrides for YAML toggles, e.g.
                 ``{"iowa_only": True}`` from a CLI ``--toggle iowa_only`` flag.
    """

    def __init__(self, toggles: Optional[Dict[str, bool]] = None) -> None:
        self._logger: logging.Logger = setup_logger("filter_module")
        self._filter_cfg = self._load_yaml("filter_config.yaml")
        self._loc_cfg = self._load_yaml("locations.yaml")

        # Merge runtime toggles on top of YAML defaults
        self._toggles: Dict[str, bool] = {
            **self._filter_cfg.get("toggles", {}),
            **(toggles or {}),
        }

    # ── config helpers ────────────────────────────────────────────────────────

    def _load_yaml(self, filename: str) -> dict:
        path = _CONFIG_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ── filter predicates ─────────────────────────────────────────────────────

    def _check_blacklist(self, job: Job) -> Optional[str]:
        """Return rejection reason string if company is blacklisted, else None."""
        name = job.company_name.lower()
        for entry in self._filter_cfg.get("blacklist_companies", []):
            if entry in name:
                return f"blacklisted company: {entry}"
        return None

    def _check_startup(self, job: Job) -> Optional[str]:
        """Return rejection reason if startup keywords found in description."""
        text = (job.description or "").lower()
        for kw in self._filter_cfg.get("startup_keywords", []):
            if kw in text:
                return f"startup signal: '{kw}'"
        return None

    def _check_location(self, job: Job) -> Optional[str]:
        """Return rejection reason if location doesn't meet preference rules."""
        loc = job.location.lower()
        loc_cfg = self._loc_cfg

        is_remote = "remote" in loc
        is_hybrid = "hybrid" in loc

        # remote_only toggle: only allow fully remote jobs
        if self._toggles.get("remote_only") and not is_remote:
            return f"remote_only toggle active; location='{job.location}'"

        # Remote / hybrid — acceptable by config
        if is_remote and loc_cfg.get("remote_acceptable", True):
            return None
        if is_hybrid and loc_cfg.get("hybrid_acceptable", True):
            return None

        # iowa_only toggle: restrict preferred states to just Iowa
        if self._toggles.get("iowa_only"):
            allowed = ["IA"]
        else:
            allowed = loc_cfg.get("preferred_states", [])

        # Check if any preferred state abbreviation appears in the location string
        for state in allowed:
            if f", {state}" in job.location or f" {state}" in job.location:
                return None

        return f"location not in preferred states; location='{job.location}'"

    # ── public API ────────────────────────────────────────────────────────────

    def filter(self, jobs: List[Job]) -> List[Job]:
        """Filter jobs and return only those that pass all criteria.

        Writes a structured trace log to
        ``data/processed/filter_trace_<date>.json`` for the agent trace
        appendix required by the assignment report.

        Args:
            jobs: List of Job objects from the search stage.

        Returns:
            Subset of jobs that passed all three filters.
        """
        self._logger.info(
            "Filter stage started | input_jobs=%d | toggles=%s",
            len(jobs),
            self._toggles,
        )

        passed: List[Job] = []
        trace: List[dict] = []
        stats = {"blacklisted": 0, "startup": 0, "location": 0, "passed": 0}

        for job in jobs:
            rejection_reason: Optional[str] = None

            # --- Filter 1: blacklist ---
            reason = self._check_blacklist(job)
            if reason:
                rejection_reason = reason
                stats["blacklisted"] += 1

            # --- Filter 2: startup heuristic ---
            if rejection_reason is None:
                reason = self._check_startup(job)
                if reason:
                    rejection_reason = reason
                    stats["startup"] += 1

            # --- Filter 3: location ---
            if rejection_reason is None:
                reason = self._check_location(job)
                if reason:
                    rejection_reason = reason
                    stats["location"] += 1

            decision = "PASS" if rejection_reason is None else "REJECT"

            # Structured trace entry (used for agent trace appendix)
            trace.append(
                {
                    "job_id": job.job_id,
                    "title": job.title,
                    "company": job.company_name,
                    "location": job.location,
                    "decision": decision,
                    "reason": rejection_reason or "passed all filters",
                }
            )

            if decision == "PASS":
                stats["passed"] += 1
                passed.append(job)
                self._logger.debug(
                    "PASS | %s @ %s | %s", job.title, job.company_name, job.location
                )
            else:
                self._logger.debug(
                    "REJECT | %s @ %s | reason: %s",
                    job.title,
                    job.company_name,
                    rejection_reason,
                )

        self._logger.info("Filter stage complete | stats=%s", stats)

        today = datetime.now().strftime("%Y-%m-%d")

        # Save filtered jobs
        filtered_path = _DATA_DIR / "processed" / f"filtered_jobs_{today}.json"
        save_json([j.to_dict() for j in passed], filtered_path)
        self._logger.info(
            "Saved %d filtered jobs to %s", len(passed), filtered_path
        )

        # Save full trace for agent trace appendix
        trace_path = _DATA_DIR / "processed" / f"filter_trace_{today}.json"
        save_json(trace, trace_path)
        self._logger.info("Saved filter trace (%d entries) to %s", len(trace), trace_path)

        return passed
