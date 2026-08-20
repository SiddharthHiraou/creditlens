"""ECOA reason codes. These are compliance behaviours, not preferences."""

from __future__ import annotations

import re

import numpy as np
import pytest

from src.explainability.reason_codes import DEFAULT_TOP_N, ReasonCodeMapper, default_mapper
from src.features.build import build


@pytest.fixture(scope="module")
def mapper() -> ReasonCodeMapper:
    return default_mapper()


def test_every_built_feature_is_mapped_or_explicitly_suppressed(mapper):
    """An unmapped feature is silently dropped from a legally required
    disclosure. This must fail the build, not warn."""
    assert mapper.unmapped(build().feature_names) == []


def test_protected_attributes_are_never_disclosed(mapper):
    """Age, sex and family status must not appear as a basis for denial."""
    names = ["DAYS_BIRTH", "CODE_GENDER", "STAB_age_years", "RATIO_annuity_to_income"]
    shap = np.array([10.0, 9.0, 8.0, 1.0])  # protected attributes dominate
    codes = mapper.explain(names, shap)
    assert len(codes) == 1
    assert codes[0].family == "affordability"
    for suppressed in ("DAYS_BIRTH", "CODE_GENDER", "STAB_age_years"):
        assert all(suppressed not in c.driving_features for c in codes)


def test_age_bearing_interaction_is_disclosed_as_a_score_reason_not_an_age_one(mapper):
    """`EXT_mean_x_age` is the model's strongest feature and contains age. It
    must surface as an external-score reason so the disclosure is actionable
    and never names age."""
    assert mapper.feature_family["EXT_mean_x_age"] == "external_score"
    codes = mapper.explain(["EXT_mean_x_age"], np.array([5.0]))
    assert codes[0].family == "external_score"
    # Word boundary, not substring: "credit reporting agency" contains "age".
    assert not re.search(r"\bage\b", codes[0].phrase, flags=re.IGNORECASE)


def test_only_contributions_pushing_toward_default_become_reasons(mapper):
    names = ["RATIO_annuity_to_income", "EXT_SOURCE_2"]
    codes = mapper.explain(names, np.array([2.0, -5.0]))
    assert [c.family for c in codes] == ["affordability"]


def test_reasons_are_deduplicated_to_distinct_families(mapper):
    """Four spellings of high debt-to-income is one reason, not four."""
    names = [
        "RATIO_annuity_to_income",
        "RATIO_credit_to_income",
        "RATIO_goods_to_income",
        "XSRC_total_debt_to_income",
        "BB_n_late_total",
    ]
    codes = mapper.explain(names, np.array([1.0, 1.0, 1.0, 1.0, 0.5]))
    assert len({c.family for c in codes}) == len(codes)
    assert codes[0].family == "affordability"
    assert codes[0].contribution == pytest.approx(4.0)


def test_families_rank_by_summed_contribution(mapper):
    names = ["RATIO_annuity_to_income", "RATIO_credit_to_income", "BB_n_late_total"]
    codes = mapper.explain(names, np.array([1.0, 1.0, 1.5]))
    # affordability sums to 2.0, delinquency is 1.5
    assert codes[0].family == "affordability"
    assert codes[1].family == "bureau_delinquency"


def test_at_most_four_reasons_are_returned(mapper):
    names = [
        "RATIO_annuity_to_income",
        "BB_n_late_total",
        "CC_util_mean",
        "EXT_SOURCE_2",
        "PREV_n_refused",
        "STAB_employed_years",
    ]
    codes = mapper.explain(names, np.ones(len(names)))
    assert len(codes) == DEFAULT_TOP_N


def test_ranks_are_sequential_from_one(mapper):
    names = ["RATIO_annuity_to_income", "BB_n_late_total", "CC_util_mean"]
    codes = mapper.explain(names, np.ones(3))
    assert [c.rank for c in codes] == [1, 2, 3]


def test_length_mismatch_is_caught_rather_than_silently_misaligned(mapper):
    with pytest.raises(ValueError, match="out of alignment"):
        mapper.explain(["RATIO_annuity_to_income"], np.array([1.0, 2.0]))


def test_every_family_has_a_phrase_and_an_actionable_flag(mapper):
    for name, meta in mapper.families.items():
        assert meta.get("phrase", "").strip(), f"{name} has no phrase"
        assert "actionable" in meta, f"{name} has no actionable flag"


def test_all_mapped_features_point_at_a_real_family(mapper):
    unknown = {f: fam for f, fam in mapper.feature_family.items() if fam not in mapper.families}
    assert unknown == {}
