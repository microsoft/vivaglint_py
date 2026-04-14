# vivaglint

A Python toolkit for analyzing Microsoft Viva Glint survey data exports.

**Version:** 0.1.0 | **Last Updated:** March 2026 | **License:** MIT

---

## Overview

**vivaglint** simplifies the analysis of Microsoft Viva Glint survey exports by providing a complete toolkit for data import, validation, statistical analysis, and reporting. It handles the repetitive data wrangling that Glint's native UI doesn't support — multi-cycle trend analysis, manager roll-ups, demographic segmentation, attrition risk scoring, and comment search.

All processing happens locally within your Python environment. No employee data is transmitted to any external service, including Microsoft.

> **Ported from R?** This is a direct Python port of the [vivaglint R package](https://github.com/microsoft/vivaglint). All 14 exported functions have identical names, parameter names, and default values. R code translates to Python with minimal changes — see [PACKAGE_USAGE.md](PACKAGE_USAGE.md) for a side-by-side comparison.

---

## Installation

```bash
pip install vivaglint
```

**Dependencies:** `pandas`, `numpy`, `scipy`, `factor_analyzer`, `rapidfuzz`, `matplotlib`, `seaborn`

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

---

## Key Capabilities

| Capability | Functions |
|---|---|
| **Import & Validate** | `read_glint_survey`, `extract_questions`, `join_attributes` |
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
- **[vignettes/vivaglint-intro.ipynb](vignettes/vivaglint-intro.ipynb)** — end-to-end Jupyter notebook walkthrough
- **[man/](man/)** — per-function reference pages (build with Sphinx)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please report security issues via [SECURITY.md](SECURITY.md).
