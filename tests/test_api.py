"""Tests for vivaglint.api — the Microsoft Graph export module.

All HTTP calls are mocked via monkeypatch so no network access is required.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import pandas as pd
import pytest

import vivaglint.api as api
from vivaglint import glint_setup, read_glint_survey_api
from vivaglint.import_ import GlintSurvey


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "uuid,cid,expected",
    [
        ("u", "c", "cycle"),
        ("u", None, "survey"),
        (None, None, "daterange"),
        (None, "c", "daterange"),
    ],
)
def test_infer_mode(uuid, cid, expected):
    assert api._infer_mode(uuid, cid) == expected


def test_build_export_url_shapes():
    cyc = api._build_export_url("cycle", "exp@t", survey_uuid="U", cycle_id="C")
    assert cyc.endswith("/surveys/U/surveyCycles/C/exportSurveys")
    srv = api._build_export_url("survey", "exp@t", survey_uuid="U")
    assert srv.endswith("/surveys/U/exportSurveys")
    dr = api._build_export_url("daterange", "exp@t")
    assert dr.endswith("/exportSurveys")
    # experience name is URL-encoded
    assert "exp%40t" in dr


def test_format_datetime():
    assert api._format_datetime(None) is None
    assert api._format_datetime("") is None
    assert api._format_datetime("2025-01-01") == "2025-01-01"
    assert api._format_datetime(dt.date(2025, 1, 2)) == "2025-01-02T00:00:00Z"
    assert (
        api._format_datetime(dt.datetime(2025, 1, 2, 3, 4, 5))
        == "2025-01-02T03:04:05Z"
    )


# ---------------------------------------------------------------------------
# glint_setup
# ---------------------------------------------------------------------------

def test_glint_setup_sets_env(monkeypatch):
    for k in ("GLINT_TENANT_ID", "GLINT_CLIENT_ID", "GLINT_CLIENT_SECRET", "GLINT_EXPERIENCE_NAME"):
        monkeypatch.delenv(k, raising=False)
    assert glint_setup("t", "c", "s", "exp@t") is True
    import os
    assert os.environ["GLINT_TENANT_ID"] == "t"
    assert os.environ["GLINT_EXPERIENCE_NAME"] == "exp@t"


def test_glint_setup_validates():
    with pytest.raises(ValueError):
        glint_setup("", "c", "s", "e")


def test_glint_setup_writes_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "creds.env"
    glint_setup("t", "c", "s", "exp@t", save_to_env_file=env_file)
    text = env_file.read_text()
    assert "GLINT_TENANT_ID=t" in text
    assert "GLINT_EXPERIENCE_NAME=exp@t" in text


# ---------------------------------------------------------------------------
# read_glint_survey_api — contract checks
# ---------------------------------------------------------------------------

def test_parse_false_requires_save_zip_to(monkeypatch):
    monkeypatch.setenv("GLINT_EXPERIENCE_NAME", "exp@t")
    monkeypatch.delenv("GLINT_SAVE_ZIP_TO", raising=False)
    with pytest.raises(ValueError, match="save_zip_to"):
        read_glint_survey_api(mode="daterange", parse=False)


def test_cycle_mode_requires_ids(monkeypatch):
    monkeypatch.setenv("GLINT_EXPERIENCE_NAME", "exp@t")
    monkeypatch.delenv("GLINT_SURVEY_UUID", raising=False)
    monkeypatch.delenv("GLINT_CYCLE_ID", raising=False)
    with pytest.raises(ValueError, match="survey_uuid"):
        read_glint_survey_api(mode="cycle")


# ---------------------------------------------------------------------------
# Full pipeline — mocked HTTP
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, *, ok=True, status_code=200, json_data=None, content=b"", text=""):
        self.ok = ok
        self.status_code = status_code
        self._json = json_data or {}
        self.content = content
        self.text = text

    def json(self):
        return self._json


def _make_zip(df_map: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, df in df_map.items():
            zf.writestr(name, df.to_csv(index=False))
    return buf.getvalue()


@pytest.fixture
def _api_env(monkeypatch):
    monkeypatch.setenv("GLINT_TENANT_ID", "t")
    monkeypatch.setenv("GLINT_CLIENT_ID", "c")
    monkeypatch.setenv("GLINT_CLIENT_SECRET", "s")
    monkeypatch.setenv("GLINT_EXPERIENCE_NAME", "exp@t")
    # Reset the module-level token cache so _ensure_token fetches fresh.
    api._TOKEN_CACHE["token"] = None
    api._TOKEN_CACHE["expires_at"] = 0.0


def _install_http_mocks(monkeypatch, zip_bytes):
    """Wire requests.post/get to a fake token + export lifecycle."""

    def fake_post(url, **kwargs):
        if "oauth2" in url or "login.microsoftonline.com" in url:
            return _FakeResponse(json_data={"access_token": "TOKEN", "expires_in": 3600})
        # start export
        return _FakeResponse(json_data={"id": "job-123", "status": "notStarted"})

    def fake_get(url, **kwargs):
        if url.endswith("/content"):
            return _FakeResponse(content=zip_bytes)
        # status poll -> immediately succeeded
        return _FakeResponse(json_data={"status": "succeeded"})

    monkeypatch.setattr(api.requests, "post", fake_post)
    monkeypatch.setattr(api.requests, "get", fake_get)
    monkeypatch.setattr(api.time, "sleep", lambda *_: None)


def _valid_survey_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Employment ID": ["A1", "A2"],
            "First Name": ["Ann", "Bob"],
            "Last Name": ["A", "B"],
            "Email": ["a@x.com", "b@x.com"],
            "Status": ["ACTIVE", "ACTIVE"],
            "Manager ID": ["M1", "M1"],
            "Survey Cycle Completion Date": ["26-03-2024 09:34", "26-03-2024 09:34"],
            "My work is meaningful": [4, 3],
            "My work is meaningful_COMMENT": ["Good", ""],
            "My work is meaningful_COMMENT_TOPICS": ["Culture", ""],
            "My work is meaningful_SENSITIVE_COMMENT_FLAG": ["", ""],
        }
    )


def test_pipeline_single_csv_returns_glint_survey(monkeypatch, _api_env):
    zip_bytes = _make_zip({"cycle_export.csv": _valid_survey_df()})
    _install_http_mocks(monkeypatch, zip_bytes)

    result = read_glint_survey_api(
        mode="cycle", survey_uuid="U", cycle_id="C"
    )
    assert isinstance(result, GlintSurvey)
    assert result.metadata["n_respondents"] == 2
    assert result.metadata["n_questions"] == 1


def test_pipeline_multi_csv_returns_dict(monkeypatch, _api_env):
    df1 = _valid_survey_df()
    df2 = _valid_survey_df()
    zip_bytes = _make_zip({"survey_A.csv": df1, "survey_B.csv": df2})
    _install_http_mocks(monkeypatch, zip_bytes)

    result = read_glint_survey_api(mode="survey", survey_uuid="U")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"survey_A", "survey_B"}
    assert all(isinstance(v, GlintSurvey) for v in result.values())


def test_pipeline_parse_false_returns_zip_path(monkeypatch, _api_env, tmp_path):
    zip_bytes = _make_zip({"cycle_export.csv": _valid_survey_df()})
    _install_http_mocks(monkeypatch, zip_bytes)

    out_dir = tmp_path / "archive"
    out_dir.mkdir()
    path = read_glint_survey_api(
        mode="daterange", parse=False, save_zip_to=str(out_dir)
    )
    assert path is not None
    from pathlib import Path

    saved = Path(path)
    assert saved.exists()
    assert saved.name == "glint-export-job-123.zip"
    assert saved.read_bytes() == zip_bytes
