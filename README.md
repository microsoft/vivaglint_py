# vivaglint <img src="man/figures/vivaglint-badge.png" align="right" height="138" alt="vivaglint" />

A Python toolkit for analyzing Microsoft Viva Glint survey data exports.

**Version:** 0.1.1 | **Last Updated:** July 2026 | **License:** MIT

---

## Overview

**vivaglint** simplifies the analysis of Microsoft Viva Glint survey exports by providing a complete toolkit for data import, validation, statistical analysis, and reporting. It handles the repetitive data wrangling that Glint's native UI doesn't support — multi-cycle trend analysis, manager roll-ups, demographic segmentation, attrition risk scoring, and comment search.

All local analysis happens within your Python environment. The API import
functions (`glint_setup`, `read_glint_survey_api`) connect to Microsoft Graph to
export and download survey data; all analysis is still performed locally. See
[Data Privacy & Security](#data-privacy--security) below.

> **Ported from R?** This is a direct Python port of the [vivaglint R package](https://github.com/microsoft/vivaglint). All 16 exported functions have identical names, parameter names, and default values. R code translates to Python with minimal changes — see [PACKAGE_USAGE.md](PACKAGE_USAGE.md) for a side-by-side comparison.

---

## Installation

```bash
pip install vivaglint
```

**Dependencies:** `pandas`, `numpy`, `scipy`, `rapidfuzz`, `requests`

**Optional extras:** `factor_analyzer` (factor analysis), `matplotlib` + `seaborn` (plots).
Install everything with `pip install "vivaglint[all]"`.

---

## Quick Start

```python
from vivaglint import read_glint_survey, summarize_survey

# Load and validate your Glint CSV export
survey = read_glint_survey("survey_export.csv", emp_id_col="Employee ID")

# Summarise all questions
summary = summarize_survey(survey, scale_points=5)
print(summary)
```

Or pull data straight from Viva Glint via the Microsoft Graph API:

```python
from vivaglint import glint_setup, read_glint_survey_api

glint_setup(
    tenant_id="your-tenant-id",
    client_id="your-client-id",
    client_secret="your-client-secret",
    experience_name="your-experience-name",
)

# Three export modes — pick whichever matches the question you want to ask.
# Inputs you don't pass are read from env vars (GLINT_SURVEY_UUID,
# GLINT_CYCLE_ID, GLINT_MODE, GLINT_START_DATE, GLINT_END_DATE). The default
# column mappings match the standard Glint API export, so typical calls need
# no column arguments.

# Cycle mode (single survey cycle):
survey = read_glint_survey_api(
    survey_uuid="your-survey-uuid",
    cycle_id="your-cycle-id",
)

# Survey mode (every cycle of one survey, returned as a dict):
all_cycles = read_glint_survey_api(
    mode="survey",
    survey_uuid="your-survey-uuid",
)

# Date-range mode (every survey active in the window, returned as a dict):
recent = read_glint_survey_api(
    mode="daterange",
    start_date="2025-01-01",
    end_date="2025-03-31",
)

# Any of the above can also persist the raw zip alongside the parsed data
# by setting save_zip_to (or the GLINT_SAVE_ZIP_TO env var):
recent = read_glint_survey_api(
    mode="daterange",
    save_zip_to="./glint-archive",   # writes glint-export-{job_id}.zip in that folder
)

# Or skip the parse entirely if you only want the zip on disk
# (save_zip_to is required when parse=False):
zip_path = read_glint_survey_api(
    mode="daterange",
    parse=False,
    save_zip_to="./glint-archive",
)
```

---

## Key Capabilities

| Capability | Functions |
|---|---|
| **Import & Validate** | `read_glint_survey`, `extract_questions`, `join_attributes` |
| **API Import** | `glint_setup`, `read_glint_survey_api` |
| **Reshape** | `pivot_long`, `split_survey_data` |
| **Core Analytics** | `summarize_survey`, `get_response_dist`, `compare_cycles` |
| **Advanced Analytics** | `get_correlations`, `extract_survey_factors`, `search_comments` |
| **Segmentation** | `analyze_by_attributes`, `analyze_attrition` |
| **Hierarchy** | `aggregate_by_manager` |

---

## Documentation

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** — function signatures and common patterns
- **[PACKAGE_USAGE.md](PACKAGE_USAGE.md)** — detailed usage guide with R-to-Python translation notes
- **[FUNCTION_OUTPUTS.md](FUNCTION_OUTPUTS.md)** — output column reference for every function

---

## Data Privacy & Security

By default, this package processes survey data locally and does not transmit data to external services. If you use the API import functions (e.g., `read_glint_survey_api()`), the package connects to Microsoft Graph to export and download survey data. All analysis is still performed locally. Ensure compliance with your organization's data handling policies when working with employee survey data.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please report security issues via [SECURITY.md](SECURITY.md).
