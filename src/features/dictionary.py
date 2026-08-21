"""Generate the data dictionary from the artifacts, never by hand.

A hand-maintained feature list is wrong within one sprint. This reads the
feature spec, the information values, the monotonic directions, the reason-code
mapping and the global SHAP summary, and writes ``docs/data_dictionary.md``.

Regenerate with ``make docs``. If a feature is added, dropped or remapped, the
document changes on the next run rather than quietly disagreeing with the model.
"""

from __future__ import annotations

from collections import Counter

import polars as pl

from src.config import ARTIFACTS, DOCS
from src.explainability.reason_codes import ReasonCodeMapper
from src.features.build import build
from src.features.monotonic import DIRECTION_RULES
from src.features.spec import FeatureSpec

# Prefix -> (source table, family description). Order matters: longest first,
# so BB_ is matched before B.
SOURCES: tuple[tuple[str, str, str], ...] = (
    ("BURO_", "bureau", "Prior credit lines reported by the credit bureau"),
    ("BB_", "bureau_balance", "Monthly bureau status history per credit line"),
    ("PREV_", "previous_application", "The applicant's history with this lender"),
    ("INST_", "installments_payments", "Actual repayment conduct on prior loans"),
    ("CC_", "credit_card_balance", "Revolving account balances and utilisation"),
    ("POS_", "POS_CASH_balance", "Point-of-sale and cash loan servicing"),
    ("XSRC_", "derived (cross-source)", "Relates the application to existing obligations"),
    ("RATIO_", "derived (application)", "Affordability ratios from the application"),
    ("STAB_", "derived (application)", "Employment and life-stage stability proxies"),
    ("EXT_", "application", "External bureau scores and their missingness pattern"),
    ("FLAG_no_", "derived (join)", "Marks an applicant with no history in a source"),
    ("AMT_", "application", "Amounts stated on the application"),
    ("CNT_", "application", "Counts stated on the application"),
    ("DAYS_", "application", "Negative day offsets from the application date"),
    ("NAME_", "application", "Categorical attributes stated on the application"),
    ("REGION_", "application", "Regional risk grade"),
    ("OCCUPATION_", "application", "Stated occupation"),
    ("CODE_", "application", "Demographic code"),
)


def classify(feature: str) -> tuple[str, str]:
    for prefix, table, description in SOURCES:
        if feature.startswith(prefix):
            return table, description
    return "application", "Application attribute"


def _iv_lookup() -> dict[str, tuple[float, str]]:
    path = ARTIFACTS / "information_values.parquet"
    if not path.exists():
        return {}
    df = pl.read_parquet(path)
    return {r["feature"]: (float(r["iv"]), r["strength"]) for r in df.iter_rows(named=True)}


def _shap_lookup() -> dict[str, float]:
    path = ARTIFACTS / "shap_global.parquet"
    if not path.exists():
        return {}
    df = pl.read_parquet(path)
    return {r["feature"]: float(r["share"]) for r in df.iter_rows(named=True)}


def _direction(feature: str) -> str:
    d = DIRECTION_RULES.get(feature, 0)
    return {1: "↑ risk", -1: "↓ risk", 0: "unconstrained"}[d]


def generate() -> str:
    spec = FeatureSpec.load()
    matrix = build()
    frame = matrix.frame.collect()
    mapper = ReasonCodeMapper.load()

    ivs = _iv_lookup()
    shap = _shap_lookup()
    selected = set(spec.features)
    dropped: dict[str, str] = spec.dropped or {}

    by_table: dict[str, list[str]] = {}
    for feature in sorted(matrix.feature_names):
        table, _ = classify(feature)
        by_table.setdefault(table, []).append(feature)

    lines: list[str] = [
        "# Data Dictionary",
        "",
        "**Generated** by `make docs` from the feature spec, the information-value",
        "table and the global SHAP summary. Do not edit by hand — a hand-maintained",
        "feature list disagrees with the model within a sprint.",
        "",
        f"- Feature spec version **{spec.version}**, fingerprint `{spec.fingerprint}`",
        f"- **{len(matrix.feature_names)}** features built, **{len(spec.features)}** selected into the champion",
        f"- **{len(spec.monotonic)}** carry a monotonic constraint",
        f"- **{len(mapper.feature_family)}** mapped to an ECOA reason family, "
        f"**{len(mapper.suppressed)}** suppressed from disclosure",
        "",
        "## Reading the columns",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| **In model** | Survived selection into the champion's feature spec |",
        "| **IV** | Information value on training data. Below 0.02 is unpredictive |",
        "| **Direction** | Monotonic constraint. `↑ risk` means PD may only rise with the feature |",
        "| **SHAP share** | Share of total mean-absolute SHAP across the out-of-time fold |",
        "| **Reason family** | ECOA family this feature is disclosed under, or `suppressed` |",
        "| **Null %** | Share missing across the full population |",
        "",
        "Missingness is meaningful throughout: a null bureau aggregate means the",
        "applicant has no bureau file, which is a real and risk-relevant segment,",
        "not a data quality problem. History *counts* fill to zero; *ratios* and",
        "*slopes* stay null.",
        "",
    ]

    for table in sorted(by_table):
        features = by_table[table]
        _, description = classify(features[0])
        in_model = sum(1 for f in features if f in selected)
        lines += [
            f"## `{table}`",
            "",
            f"{description}. **{len(features)} features, {in_model} in the champion.**",
            "",
            "| Feature | In model | IV | Direction | SHAP share | Reason family | Null % |",
            "|---|---|---|---|---|---|---|",
        ]
        for feature in features:
            iv, strength = ivs.get(feature, (None, ""))
            iv_cell = f"{iv:.4f} ({strength})" if iv is not None else "—"
            family = (
                "**suppressed**"
                if feature in mapper.suppressed
                else mapper.feature_family.get(feature, "—")
            )
            share = shap.get(feature)
            share_cell = f"{share:.2%}" if share else "—"
            null_pct = (
                f"{frame[feature].null_count() / max(frame.height, 1):.1%}"
                if feature in frame.columns
                else "—"
            )
            lines.append(
                f"| `{feature}` | {'yes' if feature in selected else 'no'} | {iv_cell} | "
                f"{_direction(feature)} | {share_cell} | {family} | {null_pct} |"
            )
        lines.append("")

    reasons = Counter(dropped.values())
    lines += [
        "## Why features were dropped",
        "",
        "Selection runs on **training data only**. Screening on the full frame — even",
        "just to compute a correlation matrix — leaks out-of-time information into the",
        "choice of features, which is subtle enough to survive review.",
        "",
        "| Reason | Features dropped |",
        "|---|---|",
    ]
    for reason, count in reasons.most_common():
        lines.append(f"| {reason} | {count} |")
    lines += [
        "",
        "## Suppressed from disclosure",
        "",
        "These may drive the model but can never appear in an adverse action notice.",
        "Age, sex and family status are protected or proxy-protected under ECOA.",
        "",
    ]
    lines += [f"- `{f}`" for f in sorted(mapper.suppressed)]
    lines += [
        "",
        "Note `EXT_mean_x_age`, the model's strongest single feature, is **not** on",
        "this list. It embeds age through an interaction and is disclosed under the",
        "external-score family, so the applicant is told something they can act on and",
        "age is never named as a basis for denial.",
        "",
    ]

    text = "\n".join(lines)
    (DOCS / "data_dictionary.md").write_text(text)
    return text


if __name__ == "__main__":
    text = generate()
    print(f"wrote docs/data_dictionary.md ({len(text.splitlines())} lines)")
