# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Development setup

```bash
git clone https://github.com/microsoft/vivaglint_py.git
cd vivaglint_py
python -m pip install -e ".[dev]"
```

## Running the tests

```bash
pytest
pytest --cov=vivaglint   # with coverage
```

All new functionality should be covered by tests in `tests/`. Tests must not make
real network calls — the Microsoft Graph API layer (`vivaglint/api.py`) is exercised
with mocked HTTP responses (see `tests/test_api.py`).

## Relationship to the R package

vivaglint (Python) is a direct port of the [vivaglint R package](https://github.com/microsoft/vivaglint).
When changing behaviour, keep the two in sync: exported function names, parameter names,
default values, and return-column names should match the R package unless there is a
Python-specific reason to diverge (document any such divergence in the docstring and in
`PACKAGE_USAGE.md`).

## Style

- Target Python 3.9+.
- Use type hints and NumPy-style docstrings, matching the existing modules.
- Reference the R source (e.g. `Ported from R/api.R::read_glint_survey_api`) in docstrings
  for functions that mirror an R counterpart.
