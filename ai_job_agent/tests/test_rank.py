"""Unit tests for RankModule.

Covers:
- Skill score: Jaccard similarity (perfect match, no match, partial)
- Location score: remote > hybrid > preferred state > middle america > other
- Recency score: recent jobs score higher; exponential decay; edge cases
- _parse_days_old: all unit strings + edge cases
- Composite scoring: weights applied correctly
- Sorting: results returned in descending order
- Top-N slicing: returns at most top_n results
- Explanation string: format matches demo requirement
- Trace and output files written
"""
import json
import math
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ai_job_agent.src.models.job import Job
from ai_job_agent.src.models.ranked_job import RankedJob
from ai_job_agent.src.modules.rank_module import RankModule


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

RANK_CFG = {
    "weights": {"skill_match": 0.5, "location": 0.3, "recency": 0.2},
    "location_scores": {
        "remote": 100,
        "hybrid": 90,
        "preferred_state": 80,
        "middle_america": 60,
        "other": 20,
    },
    "recency_decay_days": 30,
    "top_n": 10,
    "top_n_tailor": 3,
}

LOC_CFG = {
    "preferred_states": ["IA", "NE", "MO", "IL", "OH", "MI", "MN", "WI", "IN", "KS"],
    "middle_america_states": ["IA", "NE", "MO", "IL", "OH", "MI", "MN", "WI", "IN", "KS",
                               "ND", "SD", "OK", "AR", "TN", "KY"],
    "remote_acceptable": True,
    "hybrid_acceptable": True,
}

USER_SKILLS = ["python", "tensorflow", "mlflow", "aws", "docker", "sql"]


@pytest.fixture
def rank_module(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "rank_config.yaml").write_text(yaml.dump(RANK_CFG))
    (cfg_dir / "locations.yaml").write_text(yaml.dump(LOC_CFG))

    import ai_job_agent.src.modules.rank_module as rm_mod
    monkeypatch.setattr(rm_mod, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(rm_mod, "_CONFIG_DIR", cfg_dir)

    (tmp_path / "processed").mkdir(parents=True)
    return RankModule(user_skills=USER_SKILLS)


def _make_job(
    job_id="j1",
    title="AI Engineer",
    company="Acme",
    location="Des Moines, IA",
    required_skills=None,
    preferred_skills=None,
    posted_date="3 days ago",
) -> Job:
    return Job(
        job_id=job_id,
        title=title,
        company_name=company,
        location=location,
        description="",
        job_url="https://example.com",
        required_skills=required_skills or [],
        preferred_skills=preferred_skills or [],
        posted_date=posted_date,
    )


# ---------------------------------------------------------------------------
# _parse_days_old
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("just now", 0),
    ("today", 0),
    ("1 hour ago", 0),
    ("3 hours ago", 0),
    ("1 day ago", 1),
    ("5 days ago", 5),
    ("2 weeks ago", 14),
    ("1 month ago", 30),
    ("2 months ago", 60),
    (None, None),
    ("", None),
    ("unknown", None),
])
def test_parse_days_old(text, expected):
    assert RankModule._parse_days_old(text) == expected


# ---------------------------------------------------------------------------
# Skill score
# ---------------------------------------------------------------------------

def test_skill_score_perfect_match(rank_module):
    job = _make_job(required_skills=["python", "tensorflow", "mlflow", "aws", "docker", "sql"])
    score = rank_module._skill_score(job)
    assert score == pytest.approx(100.0)


def test_skill_score_no_match(rank_module):
    job = _make_job(required_skills=["cobol", "fortran", "pascal"])
    score = rank_module._skill_score(job)
    assert score == pytest.approx(0.0)


def test_skill_score_partial_match(rank_module):
    # job needs python + cobol; user has python — intersection=1, union=2+others
    job = _make_job(required_skills=["python", "cobol"])
    score = rank_module._skill_score(job)
    # intersection = {python}, union = {python, cobol, tensorflow, mlflow, aws, docker, sql}
    expected = (1 / 7) * 100
    assert score == pytest.approx(expected, rel=0.01)


def test_skill_score_uses_both_required_and_preferred(rank_module):
    job = _make_job(
        required_skills=["python"],
        preferred_skills=["tensorflow"],
    )
    score_both = rank_module._skill_score(job)

    job_req_only = _make_job(required_skills=["python"])
    score_req_only = rank_module._skill_score(job_req_only)

    assert score_both > score_req_only


def test_skill_score_no_skills_returns_zero(rank_module):
    job = _make_job(required_skills=[], preferred_skills=[])
    assert rank_module._skill_score(job) == 0.0


# ---------------------------------------------------------------------------
# Location score
# ---------------------------------------------------------------------------

def test_location_score_remote_is_highest(rank_module):
    job = _make_job(location="Remote")
    assert rank_module._location_score(job) == 100.0


def test_location_score_hybrid(rank_module):
    job = _make_job(location="Hybrid - Chicago, IL")
    assert rank_module._location_score(job) == 90.0


def test_location_score_preferred_state(rank_module):
    job = _make_job(location="Des Moines, IA")
    assert rank_module._location_score(job) == 80.0


def test_location_score_middle_america(rank_module):
    job = _make_job(location="Nashville, TN")
    assert rank_module._location_score(job) == 60.0


def test_location_score_other(rank_module):
    job = _make_job(location="San Francisco, CA")
    assert rank_module._location_score(job) == 20.0


def test_location_score_remote_beats_preferred(rank_module):
    remote = _make_job(location="Remote")
    preferred = _make_job(location="Des Moines, IA")
    assert rank_module._location_score(remote) > rank_module._location_score(preferred)


# ---------------------------------------------------------------------------
# Recency score
# ---------------------------------------------------------------------------

def test_recency_score_today_is_100(rank_module):
    job = _make_job(posted_date="just now")
    assert rank_module._recency_score(job) == pytest.approx(100.0)


def test_recency_score_recent_beats_stale(rank_module):
    fresh = _make_job(posted_date="1 day ago")
    stale = _make_job(posted_date="45 days ago")
    assert rank_module._recency_score(fresh) > rank_module._recency_score(stale)


def test_recency_score_exponential_decay(rank_module):
    decay = RANK_CFG["recency_decay_days"]
    job = _make_job(posted_date="30 days ago")
    expected = 100.0 * math.exp(-30 / decay)
    assert rank_module._recency_score(job) == pytest.approx(expected, rel=0.001)


def test_recency_score_unknown_date_returns_neutral(rank_module):
    job = _make_job(posted_date=None)
    assert rank_module._recency_score(job) == 50.0


# ---------------------------------------------------------------------------
# Explanation string
# ---------------------------------------------------------------------------

def test_explanation_contains_skill_pct(rank_module):
    job = _make_job(required_skills=["python", "tensorflow"], posted_date="3 days ago")
    rj = rank_module.rank([job])[0]
    assert "%" in rj.explanation
    assert "skill match" in rj.explanation


def test_explanation_contains_location_label(rank_module):
    job = _make_job(location="Remote", posted_date="1 day ago")
    rj = rank_module.rank([job])[0]
    assert "remote" in rj.explanation.lower()


def test_explanation_contains_recency(rank_module):
    job = _make_job(posted_date="5 days ago")
    rj = rank_module.rank([job])[0]
    assert "5 days ago" in rj.explanation


def test_explanation_format_matches_demo(rank_module):
    """Explanation must follow: '<X>% skill match | <location> | posted <N> days ago'"""
    job = _make_job(
        required_skills=["python", "tensorflow"],
        location="Des Moines, IA",
        posted_date="3 days ago",
    )
    rj = rank_module.rank([job])[0]
    parts = rj.explanation.split(" | ")
    assert len(parts) == 3
    assert "skill match" in parts[0]
    assert "days ago" in parts[2] or "today" in parts[2]


# ---------------------------------------------------------------------------
# Composite scoring and sorting
# ---------------------------------------------------------------------------

def test_results_sorted_descending(rank_module):
    jobs = [
        _make_job(job_id="j1", location="San Francisco, CA", required_skills=[]),
        _make_job(job_id="j2", location="Remote", required_skills=["python", "tensorflow"]),
        _make_job(job_id="j3", location="Des Moines, IA", required_skills=["python"]),
    ]
    results = rank_module.rank(jobs)
    scores = [r.total_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_high_skill_match_scores_higher(rank_module):
    good = _make_job(job_id="good", required_skills=["python", "tensorflow", "mlflow", "aws"])
    bad = _make_job(job_id="bad", required_skills=["cobol", "fortran"])
    results = rank_module.rank([good, bad])
    assert results[0].job.job_id == "good"


def test_composite_weights_applied(rank_module):
    """Manually compute expected score and verify."""
    job = _make_job(
        required_skills=["python"],   # partial match
        location="Remote",            # score=100
        posted_date="just now",       # score=100
    )
    skill_s = rank_module._skill_score(job)
    expected_total = 0.5 * skill_s + 0.3 * 100 + 0.2 * 100
    rj = rank_module.rank([job])[0]
    assert rj.total_score == pytest.approx(expected_total, rel=0.01)


# ---------------------------------------------------------------------------
# Top-N slicing
# ---------------------------------------------------------------------------

def test_top_n_limits_results(rank_module):
    jobs = [_make_job(job_id=f"j{i}", location="Remote") for i in range(20)]
    results = rank_module.rank(jobs)
    assert len(results) <= RANK_CFG["top_n"]


def test_fewer_jobs_than_top_n(rank_module):
    jobs = [_make_job(job_id=f"j{i}") for i in range(3)]
    results = rank_module.rank(jobs)
    assert len(results) == 3


def test_empty_input_returns_empty(rank_module):
    results = rank_module.rank([])
    assert results == []


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def test_ranked_jobs_file_written(rank_module, tmp_path):
    jobs = [_make_job(required_skills=["python"])]
    rank_module.rank(jobs)

    today = datetime.now().strftime("%Y-%m-%d")
    out_file = tmp_path / "processed" / f"ranked_jobs_{today}.json"
    assert out_file.exists()

    data = json.loads(out_file.read_text())
    assert len(data) == 1
    assert "total_score" in data[0]
    assert "explanation" in data[0]


def test_rank_trace_file_written(rank_module, tmp_path):
    jobs = [
        _make_job(job_id="j1", required_skills=["python"]),
        _make_job(job_id="j2", required_skills=["cobol"]),
    ]
    rank_module.rank(jobs)

    today = datetime.now().strftime("%Y-%m-%d")
    trace_file = tmp_path / "processed" / f"rank_trace_{today}.json"
    assert trace_file.exists()

    trace = json.loads(trace_file.read_text())
    assert len(trace) == 2
    job_ids = {entry["job_id"] for entry in trace}
    assert job_ids == {"j1", "j2"}
