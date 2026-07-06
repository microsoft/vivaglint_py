"""
vivaglint.api
-------------
Import Viva Glint survey data directly from the Microsoft Graph beta API.

Ported from ``R/api.R``.  Provides :func:`glint_setup` for configuring
credentials and :func:`read_glint_survey_api` for exporting, downloading, and
parsing survey data — an alternative to :func:`vivaglint.import_.read_glint_survey`
when you want to pull data straight from Viva Glint instead of importing a
local CSV export.

Authentication uses the OAuth2 client-credentials flow against Microsoft
Entra ID.  All configuration is read from environment variables (the Python
equivalent of the R package's ``.Renviron`` support):

* ``GLINT_TENANT_ID``       — Entra tenant ID
* ``GLINT_CLIENT_ID``       — app registration (client) ID
* ``GLINT_CLIENT_SECRET``   — app registration client secret
* ``GLINT_EXPERIENCE_NAME`` — Viva Glint experience name (e.g. ``name@tenant``)
* ``GLINT_SURVEY_UUID``     — survey UUID (cycle/survey modes)
* ``GLINT_CYCLE_ID``        — survey cycle ID (cycle mode)
* ``GLINT_MODE``            — one of ``cycle`` / ``survey`` / ``daterange``
* ``GLINT_START_DATE``      — export window start (ISO 8601)
* ``GLINT_END_DATE``        — export window end (ISO 8601)
* ``GLINT_SAVE_ZIP_TO``     — optional path to persist the raw export zip

No survey data is transmitted anywhere except between your process and
Microsoft Graph; all parsing and analysis happens locally.
"""

from __future__ import annotations

import datetime as _dt
import io
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote

import pandas as pd
import requests

from vivaglint.import_ import GlintSurvey, build_glint_survey

logger = logging.getLogger(__name__)

# Base URL for the Microsoft Graph beta sentiment/experiences endpoints.
GLINT_GRAPH_BASE = (
    "https://graph.microsoft.com/beta/employeeExperience/sentiment/experiences"
)

# Module-level token cache: (access_token, epoch_expiry_seconds).
_TOKEN_CACHE: dict[str, object] = {"token": None, "expires_at": 0.0}


# ---------------------------------------------------------------------------
# Environment-variable helpers
# ---------------------------------------------------------------------------

def _env_required(var_name: str, label: str) -> str:
    """Read a required Glint env var, raising a helpful error if missing."""
    val = os.environ.get(var_name, "")
    if not val:
        raise ValueError(
            f"{label} is not set. Run glint_setup() first to configure credentials."
        )
    return val


def _env_optional(var_name: str) -> Optional[str]:
    """Read an optional Glint env var, returning ``None`` when unset/empty."""
    val = os.environ.get(var_name, "")
    return val if val else None


# ---------------------------------------------------------------------------
# glint_setup
# ---------------------------------------------------------------------------

def glint_setup(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    experience_name: str,
    save_to_env_file: Union[bool, str, Path] = False,
) -> bool:
    """Configure Viva Glint API credentials for the current process.

    Stores the four required credentials in ``os.environ`` for the current
    Python session.  Optionally appends them to an environment file (the
    Python analogue of the R package's ``.Renviron`` support) so they can be
    loaded in future sessions.

    Ported from ``R/api.R::glint_setup``.

    Parameters
    ----------
    tenant_id:
        Microsoft Entra tenant ID.
    client_id:
        App registration (client) ID.
    client_secret:
        App registration client secret value.
    experience_name:
        Viva Glint experience name (e.g. ``"contoso@demo"``).
    save_to_env_file:
        If falsy (default), credentials are set only for this process.  If
        ``True``, they are appended to ``./.env``.  If a path is given, they
        are appended to that file instead.  Lines are written as
        ``KEY=value``.  Load such a file in a later session with, e.g.,
        ``python-dotenv`` or by exporting the variables in your shell.

    Returns
    -------
    bool
        ``True`` after credentials are saved.

    Raises
    ------
    ValueError
        If any credential is empty.
    """
    if not tenant_id:
        raise ValueError("tenant_id must be provided.")
    if not client_id:
        raise ValueError("client_id must be provided.")
    if not client_secret:
        raise ValueError("client_secret must be provided.")
    if not experience_name:
        raise ValueError("experience_name must be provided.")

    os.environ["GLINT_TENANT_ID"] = tenant_id
    os.environ["GLINT_CLIENT_ID"] = client_id
    os.environ["GLINT_CLIENT_SECRET"] = client_secret
    os.environ["GLINT_EXPERIENCE_NAME"] = experience_name

    logger.info("Glint credentials saved to environment variables for this session.")

    if save_to_env_file:
        env_path = (
            Path(save_to_env_file)
            if not isinstance(save_to_env_file, bool)
            else Path.cwd() / ".env"
        )
        lines = [
            "",
            "# --- Viva Glint API credentials ---",
            f"GLINT_TENANT_ID={tenant_id}",
            f"GLINT_CLIENT_ID={client_id}",
            f"GLINT_CLIENT_SECRET={client_secret}",
            f"GLINT_EXPERIENCE_NAME={experience_name}",
        ]
        with open(env_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        logger.info(
            "Also written to %s - load it (e.g. via python-dotenv) in future sessions.",
            env_path,
        )

    return True


# ---------------------------------------------------------------------------
# Mode + URL helpers
# ---------------------------------------------------------------------------

def _infer_mode(survey_uuid: Optional[str], cycle_id: Optional[str]) -> str:
    """Infer the export mode from which identifiers are provided.

    Backward-compatible fallback used when no explicit ``mode`` is supplied.
    Both IDs present => ``cycle``; survey UUID only => ``survey``;
    neither => ``daterange``.  Ported from ``R/api.R::infer_mode``.
    """
    has_survey = bool(survey_uuid)
    has_cycle = bool(cycle_id)
    if has_survey and has_cycle:
        return "cycle"
    if has_survey:
        return "survey"
    return "daterange"


def _build_export_url(
    mode: str,
    experience_name: str,
    survey_uuid: Optional[str] = None,
    cycle_id: Optional[str] = None,
) -> str:
    """Construct the Microsoft Graph ``exportSurveys`` URL for a given mode.

    Ported from ``R/api.R::build_export_url``.
    """
    base = f"{GLINT_GRAPH_BASE}/{quote(experience_name, safe='')}"
    if mode == "cycle":
        return f"{base}/surveys/{survey_uuid}/surveyCycles/{cycle_id}/exportSurveys"
    if mode == "survey":
        return f"{base}/surveys/{survey_uuid}/exportSurveys"
    if mode == "daterange":
        return f"{base}/exportSurveys"
    raise ValueError(f"Unknown export mode: '{mode}'.")


def _format_datetime(value) -> Optional[str]:
    """Format a date/datetime/str as an ISO 8601 UTC string (or None).

    Ported from ``R/api.R::glint_format_datetime``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, _dt.datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return str(value)


# ---------------------------------------------------------------------------
# Token acquisition
# ---------------------------------------------------------------------------

def _get_token() -> str:
    """Acquire a fresh Graph access token via the client-credentials flow.

    Ported from ``R/api.R::glint_get_token``.
    """
    tid = _env_required("GLINT_TENANT_ID", "Tenant ID")
    cid = _env_required("GLINT_CLIENT_ID", "Client ID")
    csec = _env_required("GLINT_CLIENT_SECRET", "Client Secret")

    url = f"https://login.microsoftonline.com/{tid}/oauth2/v2.0/token"
    resp = requests.post(
        url,
        data={
            "client_id": cid,
            "client_secret": csec,
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials",
        },
    )
    if not resp.ok:
        raise RuntimeError(
            f"Token request failed (HTTP {resp.status_code}): {resp.text}"
        )

    token_data = resp.json()
    access_token = token_data.get("access_token")
    if not access_token:
        raise RuntimeError("Token response did not include an access token.")

    expires_in = token_data.get("expires_in", 0)
    _TOKEN_CACHE["token"] = access_token
    _TOKEN_CACHE["expires_at"] = time.time() + float(expires_in)
    return access_token


def _ensure_token() -> str:
    """Return a cached token if still valid (>60s), otherwise fetch a new one.

    Ported from ``R/api.R::glint_ensure_token``.
    """
    token = _TOKEN_CACHE.get("token")
    expires_at = float(_TOKEN_CACHE.get("expires_at") or 0.0)
    if token and (expires_at - time.time()) > 60:
        return str(token)
    return _get_token()


# ---------------------------------------------------------------------------
# Export pipeline
# ---------------------------------------------------------------------------

def _start_export(
    export_url: str,
    start_date=None,
    end_date=None,
) -> str:
    """Start an export job and return its job ID.

    Ported from ``R/api.R::glint_start_export``.
    """
    token = _ensure_token()
    body: dict = {}
    start_val = _format_datetime(start_date)
    end_val = _format_datetime(end_date)
    if start_val is not None:
        body["startDateTime"] = start_val
    if end_val is not None:
        body["endDateTime"] = end_val

    resp = requests.post(
        export_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=body,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Export request failed (HTTP {resp.status_code}): {resp.text}"
        )

    result = resp.json()
    job_id = result.get("id")
    if not job_id:
        raise RuntimeError("Export request did not return a job ID.")

    status = result.get("status")
    logger.info(
        "Export job started - ID: %s%s",
        job_id,
        f" | Status: {status}" if status else "",
    )
    return job_id


def _poll_status(
    job_id: str,
    experience_name: str,
    poll_interval: int = 10,
    max_attempts: int = 60,
) -> bool:
    """Poll an export job until it succeeds, fails, or times out.

    Ported from ``R/api.R::glint_poll_status``.
    """
    status_url = (
        f"{GLINT_GRAPH_BASE}/{quote(experience_name, safe='')}"
        f"/operations('{job_id}')"
    )
    for _ in range(max_attempts):
        token = _ensure_token()
        resp = requests.get(status_url, headers={"Authorization": f"Bearer {token}"})
        if not resp.ok:
            raise RuntimeError(
                f"Status check failed (HTTP {resp.status_code}): {resp.text}"
            )

        result = resp.json()
        status = str(result.get("status") or "").lower()
        if status == "succeeded":
            return True
        if status == "failed":
            detail = result.get("statusDetail") or result.get("status")
            raise RuntimeError(f"Export job failed: {detail}")

        time.sleep(poll_interval)

    raise TimeoutError(f"Timed out after {max_attempts * poll_interval} seconds.")


def _download_export(job_id: str, experience_name: str) -> requests.Response:
    """Download the completed export's content (the raw zip bytes).

    Ported from ``R/api.R::glint_download_export``.
    """
    token = _ensure_token()
    download_url = (
        f"{GLINT_GRAPH_BASE}/{quote(experience_name, safe='')}"
        f"/operations('{job_id}')/content"
    )
    resp = requests.get(download_url, headers={"Authorization": f"Bearer {token}"})
    if not resp.ok:
        raise RuntimeError(
            f"Download failed (HTTP {resp.status_code}): {resp.text}"
        )
    return resp


def _import_export_zip(
    content: bytes,
    encoding: str = "UTF-8",
    combine: bool = True,
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Extract the export zip and load its CSVs.

    Ported from ``R/api.R::glint_import_export_zip``.

    Parameters
    ----------
    content:
        Raw zip bytes.
    encoding:
        Character encoding passed to :func:`pandas.read_csv`.
    combine:
        When ``True`` (default), CSVs sharing an identical column schema are
        concatenated.  When ``False``, every multi-CSV export is returned as a
        dict of DataFrames keyed by filename (without ``.csv``).  Single-CSV
        exports always return a single DataFrame.

    Returns
    -------
    pd.DataFrame or dict[str, pd.DataFrame]
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV files found in the ZIP archive.")

        dfs: dict[str, pd.DataFrame] = {}
        for name in csv_names:
            with zf.open(name) as fh:
                df = pd.read_csv(io.TextIOWrapper(fh, encoding=encoding))
            key = Path(name).stem
            dfs[key] = df

    if len(dfs) == 1:
        return next(iter(dfs.values()))

    if combine:
        first_cols = list(next(iter(dfs.values())).columns)
        same_schema = all(list(df.columns) == first_cols for df in dfs.values())
        if same_schema:
            return pd.concat(list(dfs.values()), ignore_index=True)

    return dfs


def _persist_export_zip(content: bytes, path: Union[str, Path], job_id: str) -> str:
    """Write the raw export zip to disk before parsing.

    Path semantics mirror ``R/api.R::persist_export_zip``: if *path* ends with
    a separator or is an existing directory, the zip is written as
    ``glint-export-{job_id}.zip`` inside it; otherwise *path* is treated as a
    full file path.  Parent directories are created as needed.

    Returns
    -------
    str
        The path the zip was written to.
    """
    path_str = str(path)
    ends_with_sep = path_str.endswith(("/", "\\"))
    is_existing_dir = Path(path_str).is_dir()

    if ends_with_sep or is_existing_dir:
        dest = Path(path_str.rstrip("/\\")) / f"glint-export-{job_id}.zip"
    else:
        dest = Path(path_str)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    logger.info("Export zip saved to: %s", dest)
    return str(dest)


def _run_export_pipeline(
    export_url: str,
    experience_name: str,
    start_date=None,
    end_date=None,
    poll_interval: int = 10,
    max_attempts: int = 60,
    encoding: str = "UTF-8",
    combine: bool = True,
    save_zip_to: Optional[Union[str, Path]] = None,
    parse: bool = True,
):
    """Run the full export → poll → download → (persist | parse) pipeline.

    Ported from ``R/api.R::glint_run_export_pipeline``.
    """
    job_id = _start_export(export_url, start_date=start_date, end_date=end_date)
    _poll_status(
        job_id,
        experience_name=experience_name,
        poll_interval=poll_interval,
        max_attempts=max_attempts,
    )
    resp = _download_export(job_id, experience_name=experience_name)
    content = resp.content

    # Persist before parsing so a parse error doesn't lose the bytes.
    saved_path: Optional[str] = None
    if save_zip_to:
        saved_path = _persist_export_zip(content, save_zip_to, job_id)

    if not parse:
        return saved_path

    return _import_export_zip(content, encoding=encoding, combine=combine)


# ---------------------------------------------------------------------------
# read_glint_survey_api
# ---------------------------------------------------------------------------

def read_glint_survey_api(
    survey_uuid: Optional[str] = None,
    cycle_id: Optional[str] = None,
    mode: Optional[str] = None,
    emp_id_col: Optional[str] = "Employment ID",
    first_name_col: Optional[str] = "First Name",
    last_name_col: Optional[str] = "Last Name",
    email_col: Optional[str] = "Email",
    status_col: Optional[str] = "Status",
    completion_date_col: Optional[str] = "Survey Cycle Completion Date",
    sent_date_col: Optional[str] = None,
    manager_id_col: Optional[str] = "Manager ID",
    start_date=None,
    end_date=None,
    encoding: str = "UTF-8",
    poll_interval: int = 10,
    max_attempts: int = 60,
    save_zip_to: Optional[Union[str, Path]] = None,
    parse: bool = True,
    experience_name: Optional[str] = None,
) -> Union[GlintSurvey, dict, str, None]:
    """Read Viva Glint survey data via the Microsoft Graph beta API.

    Exports survey data through the Graph API, downloads the resulting ZIP
    archive, and returns either a :class:`GlintSurvey` object or a dict of them
    depending on what the export contains.  This is the API-based alternative
    to :func:`vivaglint.import_.read_glint_survey`.

    Ported from ``R/api.R::read_glint_survey_api``.

    Modes
    -----
    * ``"cycle"`` — a single survey cycle.  Requires *survey_uuid* and
      *cycle_id*.
    * ``"survey"`` — every cycle of one survey.  Requires *survey_uuid*.
    * ``"daterange"`` — every survey active between *start_date* and *end_date*.
      Both dates are optional (the API defaults to roughly the last six months).

    When *mode* is ``None`` it is read from ``GLINT_MODE``; if that is also
    unset it is inferred from which identifiers are populated (both IDs =>
    ``cycle``, survey UUID only => ``survey``, neither => ``daterange``).

    Environment-variable fallbacks
    ------------------------------
    Any input not supplied as an argument is read from the matching environment
    variable: *survey_uuid* ← ``GLINT_SURVEY_UUID``, *cycle_id* ←
    ``GLINT_CYCLE_ID``, *mode* ← ``GLINT_MODE``, *start_date* ←
    ``GLINT_START_DATE``, *end_date* ← ``GLINT_END_DATE``, *save_zip_to* ←
    ``GLINT_SAVE_ZIP_TO``.  Explicit arguments always win.

    Parameters
    ----------
    survey_uuid, cycle_id:
        Survey/cycle identifiers (see modes above).
    mode:
        ``"cycle"`` / ``"survey"`` / ``"daterange"`` or ``None`` to resolve
        from env/inference.
    emp_id_col:
        Employee identifier column.  Defaults to ``"Employment ID"`` — the name
        Microsoft Graph emits in API exports.
    first_name_col, last_name_col, email_col, status_col, completion_date_col,
    sent_date_col, manager_id_col:
        Standard column names.  ``sent_date_col`` defaults to ``None`` because
        the API export does not include that column.
    start_date, end_date:
        Optional export-window bounds.  Accept an ISO 8601 string, a
        :class:`datetime.date`, or a :class:`datetime.datetime`.
    encoding:
        CSV encoding (default ``"UTF-8"``).
    poll_interval:
        Seconds between status checks (default ``10``).
    max_attempts:
        Maximum number of polling attempts (default ``60``).
    save_zip_to:
        Optional path.  When set, the raw export zip is written to disk before
        parsing.  Required when ``parse=False``.
    parse:
        When ``True`` (default), the zip is unpacked into :class:`GlintSurvey`
        objects.  When ``False``, the zip is only downloaded and its saved path
        is returned (``save_zip_to`` must be set).
    experience_name:
        Optional override for ``GLINT_EXPERIENCE_NAME``.

    Returns
    -------
    GlintSurvey
        Single-CSV exports (typical for ``cycle`` mode).
    dict
        Multi-CSV exports (typical for ``survey`` / ``daterange`` modes) — a
        dict keyed by source CSV filename.  Entries that fit the standard
        schema are wrapped as :class:`GlintSurvey`; others are returned as raw
        DataFrames (with a warning).
    str
        When ``parse=False`` — the path the raw export zip was written to.

    Raises
    ------
    ValueError, RuntimeError, TimeoutError
        On invalid inputs or API/pipeline failures.
    """
    import warnings

    # Resolve env-var fallbacks for each input the caller didn't pass.
    survey_uuid = survey_uuid or _env_optional("GLINT_SURVEY_UUID")
    cycle_id = cycle_id or _env_optional("GLINT_CYCLE_ID")
    start_date = start_date or _env_optional("GLINT_START_DATE")
    end_date = end_date or _env_optional("GLINT_END_DATE")
    mode = mode or _env_optional("GLINT_MODE")
    save_zip_to = save_zip_to or _env_optional("GLINT_SAVE_ZIP_TO")

    # parse=False without a save destination would download and discard the
    # zip.  Fail fast before any API call so the contract is unambiguous.
    if not parse and not save_zip_to:
        raise ValueError(
            "parse=False requires save_zip_to to be set (or GLINT_SAVE_ZIP_TO "
            "in the environment); otherwise the downloaded zip has nowhere to go."
        )

    if mode is None:
        mode = _infer_mode(survey_uuid, cycle_id)
    if mode not in ("cycle", "survey", "daterange"):
        raise ValueError(
            f"mode must be one of 'cycle', 'survey', 'daterange'; got {mode!r}."
        )

    # Per-mode required-input validation.
    if mode == "cycle":
        if not survey_uuid:
            raise ValueError(
                "survey_uuid must be provided for cycle mode (or set GLINT_SURVEY_UUID)."
            )
        if not cycle_id:
            raise ValueError(
                "cycle_id must be provided for cycle mode (or set GLINT_CYCLE_ID)."
            )
    elif mode == "survey":
        if not survey_uuid:
            raise ValueError(
                "survey_uuid must be provided for survey mode (or set GLINT_SURVEY_UUID)."
            )

    exp_name = experience_name or _env_required(
        "GLINT_EXPERIENCE_NAME", "Experience Name"
    )

    export_url = _build_export_url(
        mode,
        experience_name=exp_name,
        survey_uuid=survey_uuid,
        cycle_id=cycle_id,
    )

    # combine=False so multi-CSV exports come back as a dict of DataFrames;
    # we wrap each entry below.  parse=False returns the saved zip path.
    data = _run_export_pipeline(
        export_url,
        experience_name=exp_name,
        start_date=start_date,
        end_date=end_date,
        poll_interval=poll_interval,
        max_attempts=max_attempts,
        encoding=encoding,
        combine=False,
        save_zip_to=save_zip_to,
        parse=parse,
    )

    if not parse:
        return data  # the saved zip path (str) or None

    build_kwargs = dict(
        emp_id_col=emp_id_col,
        first_name_col=first_name_col,
        last_name_col=last_name_col,
        email_col=email_col,
        status_col=status_col,
        completion_date_col=completion_date_col,
        sent_date_col=sent_date_col,
        manager_id_col=manager_id_col,
        file_path=None,
    )

    # Single CSV (typical for cycle mode) -> single GlintSurvey.
    if isinstance(data, pd.DataFrame):
        return build_glint_survey(data, **build_kwargs)

    # Multi-CSV -> dict.  Each entry is wrapped as a GlintSurvey when its
    # schema fits; entries that don't fit (supplementary metadata, attribute
    # files, etc.) come back as raw DataFrames with a warning.
    result: dict = {}
    for name, df in data.items():
        try:
            result[name] = build_glint_survey(df, **build_kwargs)
        except Exception as exc:
            warnings.warn(
                f"Could not wrap '{name}' as a GlintSurvey object ({exc}). "
                "Returning the raw DataFrame for this entry.",
                UserWarning,
                stacklevel=2,
            )
            result[name] = df

    return result
