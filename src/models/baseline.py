"""Phase 1 baseline: logistic regression on application-level fields only.

This is deliberately the weakest defensible model. It uses nothing from the
bureau or repayment-history tables, so the lift that Phase 2's relational
aggregations deliver is measurable against an honest floor rather than against
a strawman.

Design notes worth defending:

* Median imputation with an explicit ``*_was_missing`` indicator. Missingness
  in ``EXT_SOURCE_1`` is informative -- thin-file applicants are likelier to
  lack an external score -- so discarding the pattern throws away signal.
* ``class_weight="balanced"`` rather than resampling. SMOTE on credit data
  fabricates borrowers who never applied and reliably degrades calibration.
* Standardisation, because L2 regularisation is not scale-invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_SEED

NUMERIC = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_ID_PUBLISH",
    "REGION_RATING_CLIENT",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

CATEGORICAL = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE",
]

# The four ratios every credit shop computes. A baseline without DTI is not a
# baseline anyone would accept.
RATIOS = [
    "ratio_annuity_to_income",
    "ratio_credit_to_income",
    "ratio_credit_to_goods",
    "ratio_annuity_to_credit",
]

MISSING_INDICATORS = ["EXT_SOURCE_1", "EXT_SOURCE_3"]


def add_baseline_ratios(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Attach the standard affordability ratios."""
    income = pl.col("AMT_INCOME_TOTAL").clip(lower_bound=1.0)
    goods = pl.col("AMT_GOODS_PRICE").clip(lower_bound=1.0)
    credit = pl.col("AMT_CREDIT").clip(lower_bound=1.0)
    return lf.with_columns(
        ratio_annuity_to_income=(pl.col("AMT_ANNUITY") / income),
        ratio_credit_to_income=(pl.col("AMT_CREDIT") / income),
        ratio_credit_to_goods=(credit / goods),
        ratio_annuity_to_credit=(pl.col("AMT_ANNUITY") / credit),
    )


def add_missing_indicators(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Preserve the informative-missingness pattern before imputation."""
    return lf.with_columns(
        [pl.col(c).is_null().cast(pl.Int8).alias(f"{c}_was_missing") for c in MISSING_INDICATORS]
    )


def prepare(lf: pl.LazyFrame) -> pl.LazyFrame:
    return add_missing_indicators(add_baseline_ratios(lf))


@dataclass
class BaselineModel:
    pipeline: Pipeline
    feature_names: list[str] = field(default_factory=list)

    def predict_pd(self, df: pl.DataFrame) -> np.ndarray:
        # sklearn consumes polars frames directly (>=1.4); no pandas round-trip.
        return self.pipeline.predict_proba(df.select(self.feature_names))[:, 1]

    def coefficients(self) -> pl.DataFrame:
        """Signed coefficients on the standardised scale, largest effect first."""
        pre = self.pipeline.named_steps["preprocess"]
        clf = self.pipeline.named_steps["clf"]
        names = list(pre.get_feature_names_out())
        return (
            pl.DataFrame({"feature": names, "coefficient": clf.coef_[0]})
            .with_columns(abs_coefficient=pl.col("coefficient").abs())
            .sort("abs_coefficient", descending=True)
        )


def build_pipeline() -> Pipeline:
    numeric_cols = NUMERIC + RATIOS
    indicator_cols = [f"{c}_was_missing" for c in MISSING_INDICATORS]

    numeric_tf = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_tf = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False),
            ),
        ]
    )

    pre = ColumnTransformer(
        [
            ("num", numeric_tf, numeric_cols),
            ("cat", categorical_tf, CATEGORICAL),
            ("ind", "passthrough", indicator_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        [
            ("preprocess", pre),
            (
                "clf",
                LogisticRegression(
                    # sklearn 1.8 deprecated `penalty`; l1_ratio=0 is ridge.
                    l1_ratio=0.0,
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def feature_columns() -> list[str]:
    return NUMERIC + RATIOS + CATEGORICAL + [f"{c}_was_missing" for c in MISSING_INDICATORS]


def fit(train: pl.DataFrame, *, label_col: str = "label") -> BaselineModel:
    cols = feature_columns()
    pipe = build_pipeline()
    pipe.fit(train.select(cols), train[label_col].to_numpy())
    return BaselineModel(pipeline=pipe, feature_names=cols)
