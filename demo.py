"""
vivaglint demo script — Python equivalent of demo.R

Demonstrates all major package functions using a sample survey file.
Run from the repo root:

    python demo.py
"""

from vivaglint import (
    read_glint_survey,
    extract_questions,
    summarize_survey,
    get_response_dist,
    pivot_long,
    get_correlations,
    extract_survey_factors,
    aggregate_by_manager,
    analyze_by_attributes,
    analyze_attrition,
    search_comments,
    split_survey_data,
    compare_cycles,
    join_attributes,
)

# ---------------------------------------------------------------------------
# 1. Load survey data
# ---------------------------------------------------------------------------
survey = read_glint_survey(
    "tests/fixtures/sample_survey.csv",
    emp_id_col="Employee ID",
)
print(survey)

# ---------------------------------------------------------------------------
# 2. List all questions
# ---------------------------------------------------------------------------
questions = extract_questions(survey)
print("\nQuestions:")
print(questions)
print("There are:", len(questions), "questions in the survey.")

# ---------------------------------------------------------------------------
# 3. Summarise all questions
# ---------------------------------------------------------------------------
summary = summarize_survey(survey, scale_points=5)
print("\nSummary:")
print(summary)

# ---------------------------------------------------------------------------
# 3b. Response distributions
# ---------------------------------------------------------------------------
distributions = get_response_dist(survey)
print("\nResponse distributions:")
print(distributions)

# ---------------------------------------------------------------------------
# 4. Analyse a specific question
# ---------------------------------------------------------------------------
detailed = summarize_survey(
    survey,
    scale_points=5,
    questions=["My work is meaningful"],
)
print("\nDetailed (single question):")
print(detailed)

# ---------------------------------------------------------------------------
# 5. Extract comments in long format
# ---------------------------------------------------------------------------
comments = pivot_long(survey, data_type="comments")
print("Total non-empty comments:", len(comments), "\n")
print("\nComments (long format):")
print(comments.head())

# ---------------------------------------------------------------------------
# 6. Reshape to long format — all responses
# ---------------------------------------------------------------------------
df_long_return = pivot_long(survey, data_type="all")
print("\nLong format (all):")
long_data = df_long_return[["First Name", "question", "response"]]
print(long_data.head())

# ---------------------------------------------------------------------------
# 7. Both formats at once
# ---------------------------------------------------------------------------
both = pivot_long(survey, data_type="both")
print("\n All Responses:\n", both["all"].head())
print("\n Comments Only:\n", both["comments"].head())

# ---------------------------------------------------------------------------
# 8. Correlations
# ---------------------------------------------------------------------------
correlations_long   = get_correlations(survey)
correlations_matrix = get_correlations(survey, format="matrix")
print("\nCorrelations (long):")
print(correlations_long)

# ---------------------------------------------------------------------------
# 9. Factor analysis
# ---------------------------------------------------------------------------
try:
    from factor_analyzer import FactorAnalyzer  # noqa: F401
    factors = extract_survey_factors(survey, n_factors=1, rotation="oblimin")
    print("\nFactor summary:")
    print(factors["factor_summary"])
except ImportError:
    print("\nSkipping factor analysis — factor_analyzer not installed.")

# ---------------------------------------------------------------------------
# 10. Manager roll-up
# ---------------------------------------------------------------------------
try:
    managers = aggregate_by_manager(survey, scale_points=5)
    print("\nManager summary:")
    print(managers.head())
except Exception as e:
    print(f"\nSkipping manager analysis: {e}")

# ---------------------------------------------------------------------------
# 11. Demographic segmentation
# ---------------------------------------------------------------------------
try:
    survey_enriched = join_attributes(survey, "tests/fixtures/sample_attributes.csv")
    by_dept = analyze_by_attributes(
        survey_enriched,
        scale_points=5,
        attribute_cols="Department",
    )
    print("\nBy department:")
    print(by_dept.head())
except Exception as e:
    print(f"\nSkipping attribute analysis: {e}")

# ---------------------------------------------------------------------------
# 12. Attrition risk
# ---------------------------------------------------------------------------
try:
    attrition = analyze_attrition(
        survey,
        attrition_file="tests/fixtures/sample_attrition.csv",
        emp_id_col="Employee ID",
        term_date_col="Termination Date",
        scale_points=5,
    )
    print("\nAttrition:")
    print(attrition.head())
except Exception as e:
    print(f"\nSkipping attrition analysis: {e}")

# ---------------------------------------------------------------------------
# 13. Comment search
# ---------------------------------------------------------------------------
results = search_comments(survey, "flexibility")
print("\nComment search results:")
print(results)

# ---------------------------------------------------------------------------
# 14. Split quantitative / qualitative
# ---------------------------------------------------------------------------
parts = split_survey_data(survey)
print("\nQuantitative columns:", list(parts["quantitative"].columns))
print("Qualitative columns:",  list(parts["qualitative"].columns))
