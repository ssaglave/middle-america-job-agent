"""Unit tests for FilterModule.

Covers:
- FAANG+ blacklist rejection
- Startup keyword heuristic rejection
- Location filtering (preferred state, remote, hybrid, out-of-region)
- Iowa-only toggle
- Remote-only toggle
- Structured trace output (decision + reason per job)
- Stats accuracy
"""
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ai_job_agent.src.models.job import Job
from ai_job_agent.src.modules.filter_module import FilterModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(
    title="AI Engineer",
    company="Acme Corp",
    location="Des Moines, IA",
    description="We build ML systems.",
    job_id=None,
) -> Job:
    return Job(
        job_id=job_id or f"{company}-{location}",
        title=title,
        company_name=company,
        location=location,
        description=description,
        job_url="https://example.com/job",
    )


FILTER_CFG = {
    "blacklist_companies": ["google", "amazon", "meta", "microsoft", "apple", "netflix"],
    "startup_keywords": ["series a", "seed stage", "early stage"],
    "company_size": {"min_employees": 50, "max_employees": 5000},
    "toggles": {"iowa_only": False, "remote_only": False},
}

LOC_CFG = {
    "preferred_states": ["IA", "NE", "MO", "IL", "OH", "MI", "MN", "WI", "IN", "KS"],
    "middle_america_states": ["IA", "NE", "MO", "IL", "OH", "MI", "MN", "WI", "IN", "KS"],
    "remote_acceptable": True,
    "hybrid_acceptable": True,
}


@pytest.fixture
def filter_module(tmp_path, monkeypatch):
    """FilterModule with patched config and data dirs."""
    # Write config files to tmp_path/config
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "filter_config.yaml").write_text(yaml.dump(FILTER_CFG))
    (cfg_dir / "locations.yaml").write_text(yaml.dump(LOC_CFG))

    # Patch data dir so trace/filtered files land in tmp_path
    import ai_job_agent.src.modules.filter_module as fm_mod
    monkeypatch.setattr(fm_mod, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(fm_mod, "_CONFIG_DIR", cfg_dir)

    (tmp_path / "processed").mkdir(parents=True)
    return FilterModule()


def _make_module_with_toggles(tmp_path, monkeypatch, toggles: dict) -> FilterModule:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "filter_config.yaml").write_text(yaml.dump(FILTER_CFG))
    (cfg_dir / "locations.yaml").write_text(yaml.dump(LOC_CFG))

    import ai_job_agent.src.modules.filter_module as fm_mod
    monkeypatch.setattr(fm_mod, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(fm_mod, "_CONFIG_DIR", cfg_dir)

    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    return FilterModule(toggles=toggles)


# ---------------------------------------------------------------------------
# Blacklist tests
# ---------------------------------------------------------------------------

def test_google_is_rejected(filter_module):
    jobs = [_make_job(company="Google LLC", location="Remote")]
    result = filter_module.filter(jobs)
    assert result == []


def test_amazon_is_rejected(filter_module):
    jobs = [_make_job(company="Amazon Web Services", location="Remote")]
    result = filter_module.filter(jobs)
    assert result == []


def test_blacklist_is_case_insensitive(filter_module):
    jobs = [_make_job(company="GOOGLE", location="Remote")]
    result = filter_module.filter(jobs)
    assert result == []


def test_non_blacklisted_company_passes(filter_module):
    jobs = [_make_job(company="Acme Analytics", location="Des Moines, IA")]
    result = filter_module.filter(jobs)
    assert len(result) == 1


def test_partial_name_match_rejects(filter_module):
    """'meta' inside 'Metamind Corp' should be caught."""
    jobs = [_make_job(company="Metamind Corp", location="Remote")]
    result = filter_module.filter(jobs)
    assert result == []


# ---------------------------------------------------------------------------
# Startup heuristic tests
# ---------------------------------------------------------------------------

def test_series_a_description_rejected(filter_module):
    jobs = [_make_job(
        company="FastAI Startup",
        location="Remote",
        description="We are a Series A company building the future.",
    )]
    result = filter_module.filter(jobs)
    assert result == []


def test_seed_stage_description_rejected(filter_module):
    jobs = [_make_job(
        company="NewCo",
        location="Remote",
        description="Seed stage startup looking for AI talent.",
    )]
    result = filter_module.filter(jobs)
    assert result == []


def test_no_startup_keywords_passes(filter_module):
    jobs = [_make_job(
        company="MidWest AI",
        location="Des Moines, IA",
        description="Established company with 200 employees building AI tools.",
    )]
    result = filter_module.filter(jobs)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Location tests
# ---------------------------------------------------------------------------

def test_preferred_state_passes(filter_module):
    for state in ["IA", "NE", "MO", "IL"]:
        jobs = [_make_job(location=f"Springfield, {state}")]
        result = filter_module.filter(jobs)
        assert len(result) == 1, f"Expected {state} to pass"


def test_non_preferred_location_rejected(filter_module):
    jobs = [_make_job(location="San Francisco, CA")]
    result = filter_module.filter(jobs)
    assert result == []


def test_remote_job_passes(filter_module):
    jobs = [_make_job(location="Remote")]
    result = filter_module.filter(jobs)
    assert len(result) == 1


def test_hybrid_job_passes(filter_module):
    jobs = [_make_job(location="Hybrid - Chicago, IL")]
    result = filter_module.filter(jobs)
    assert len(result) == 1


def test_remote_in_location_string_passes(filter_module):
    jobs = [_make_job(location="United States (Remote)")]
    result = filter_module.filter(jobs)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Iowa-only toggle
# ---------------------------------------------------------------------------

def test_iowa_only_toggle_allows_ia(tmp_path, monkeypatch):
    module = _make_module_with_toggles(tmp_path, monkeypatch, {"iowa_only": True})
    jobs = [_make_job(location="Des Moines, IA")]
    result = module.filter(jobs)
    assert len(result) == 1


def test_iowa_only_toggle_rejects_ne(tmp_path, monkeypatch):
    module = _make_module_with_toggles(tmp_path, monkeypatch, {"iowa_only": True})
    jobs = [_make_job(location="Omaha, NE")]
    result = module.filter(jobs)
    assert result == []


def test_iowa_only_toggle_still_allows_remote(tmp_path, monkeypatch):
    module = _make_module_with_toggles(tmp_path, monkeypatch, {"iowa_only": True})
    jobs = [_make_job(location="Remote")]
    result = module.filter(jobs)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Remote-only toggle
# ---------------------------------------------------------------------------

def test_remote_only_rejects_non_remote(tmp_path, monkeypatch):
    module = _make_module_with_toggles(tmp_path, monkeypatch, {"remote_only": True})
    jobs = [_make_job(location="Des Moines, IA")]
    result = module.filter(jobs)
    assert result == []


def test_remote_only_allows_remote(tmp_path, monkeypatch):
    module = _make_module_with_toggles(tmp_path, monkeypatch, {"remote_only": True})
    jobs = [_make_job(location="Remote")]
    result = module.filter(jobs)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Mixed batch — multiple filters applied together
# ---------------------------------------------------------------------------

def test_mixed_batch_correct_counts(filter_module):
    jobs = [
        _make_job(company="Google", location="Remote", job_id="j1"),          # blacklisted
        _make_job(company="MidAI", location="Remote", job_id="j2",
                  description="Series A company."),                            # startup
        _make_job(company="Acme", location="San Francisco, CA", job_id="j3"), # wrong location
        _make_job(company="Acme", location="Des Moines, IA", job_id="j4"),    # PASS
        _make_job(company="DataCo", location="Remote", job_id="j5"),          # PASS
    ]
    result = filter_module.filter(jobs)
    assert len(result) == 2
    passed_ids = {j.job_id for j in result}
    assert passed_ids == {"j4", "j5"}


# ---------------------------------------------------------------------------
# Trace output
# ---------------------------------------------------------------------------

def test_trace_file_is_written(filter_module, tmp_path):
    import json
    from datetime import datetime

    jobs = [
        _make_job(company="Amazon", location="Remote", job_id="j1"),
        _make_job(company="Acme", location="Des Moines, IA", job_id="j2"),
    ]
    filter_module.filter(jobs)

    today = datetime.now().strftime("%Y-%m-%d")
    trace_file = tmp_path / "processed" / f"filter_trace_{today}.json"
    assert trace_file.exists()

    trace = json.loads(trace_file.read_text())
    assert len(trace) == 2

    decisions = {entry["job_id"]: entry["decision"] for entry in trace}
    assert decisions["j1"] == "REJECT"
    assert decisions["j2"] == "PASS"


def test_trace_entries_have_reason(filter_module, tmp_path):
    import json
    from datetime import datetime

    jobs = [_make_job(company="Google", location="Remote", job_id="j1")]
    filter_module.filter(jobs)

    today = datetime.now().strftime("%Y-%m-%d")
    trace = json.loads(
        (tmp_path / "processed" / f"filter_trace_{today}.json").read_text()
    )
    entry = trace[0]
    assert entry["decision"] == "REJECT"
    assert "google" in entry["reason"].lower()
