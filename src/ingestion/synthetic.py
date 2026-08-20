"""Synthetic Home-Credit-shaped data with a real origination calendar.

Why this exists: the Kaggle sources need an authenticated download and, for
Home Credit, acceptance of the competition rules in a browser. This generator
emits the *same table and column names* so every downstream module -- schemas,
aggregations, splits, training -- is written against the real contract and
keeps working unchanged once the CSVs land in ``data/raw/``.

Two deliberate departures from the real Home Credit files, both improvements
for this project's purposes:

* ``application`` carries an ``origination_date``. Real Home Credit has no
  application timestamp at all, which is why out-of-time evaluation in this
  project runs on Lending Club (see docs/target_definition.md).
* ``application`` carries ``max_dpd_in_window``, the observed delinquency
  depth, so the target rule is applied rather than assumed.

A single latent risk factor per applicant drives the application attributes,
the bureau history and the repayment behaviour, so relational aggregations
carry genuine signal and Phase 2 feature work is not chasing noise.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.config import RANDOM_SEED, SYNTHETIC

EDUCATION = [
    "Lower secondary",
    "Secondary / secondary special",
    "Incomplete higher",
    "Higher education",
    "Academic degree",
]
FAMILY = ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"]
OCCUPATION = [
    "Laborers",
    "Sales staff",
    "Core staff",
    "Managers",
    "Drivers",
    "Accountants",
    "Medicine staff",
    "High skill tech staff",
]
CREDIT_TYPE = ["Consumer credit", "Credit card", "Car loan", "Mortgage", "Microloan"]
CONTRACT_STATUS = ["Approved", "Refused", "Canceled", "Unused offer"]


@dataclass(frozen=True)
class SyntheticConfig:
    n_applicants: int = 50_000
    start: dt.date = dt.date(2020, 1, 1)
    end: dt.date = dt.date(2023, 6, 30)
    base_bad_rate: float = 0.09
    seed: int = RANDOM_SEED


def _dates(rng: np.random.Generator, n: int, cfg: SyntheticConfig) -> np.ndarray:
    span = (cfg.end - cfg.start).days
    # Mild volume growth over time, as a real book would show.
    w = np.linspace(0.7, 1.3, span + 1)
    offs = rng.choice(span + 1, size=n, p=w / w.sum())
    # datetime64[D] maps straight onto polars Date; a list of datetime.date
    # objects would land as an unwritable Object column.
    return np.datetime64(cfg.start, "D") + offs.astype("timedelta64[D]")


def generate(cfg: SyntheticConfig | None = None) -> dict[str, pl.DataFrame]:
    """Build every table. Returns a name -> DataFrame mapping."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_applicants
    ids = np.arange(100_001, 100_001 + n, dtype=np.int64)

    # ---- latent risk -----------------------------------------------------
    z = rng.normal(size=n)
    origination = _dates(rng, n, cfg)
    orig_year = origination.astype("datetime64[Y]").astype(int) + 1970
    # A vintage effect: the 2022 cohort underwrites worse. This gives drift
    # and vintage analysis something true to find later.
    vintage_shift = np.where(orig_year == 2022, 0.25, 0.0)
    risk = z + vintage_shift

    # ---- application table ----------------------------------------------
    age_years = np.clip(rng.normal(43, 11, n) - 2.5 * risk, 21, 69)
    employed_years = np.clip(rng.gamma(2.0, 3.0, n) - 1.2 * risk, 0, 45)
    income = np.exp(rng.normal(11.9, 0.55, n) - 0.18 * risk).round(-2)
    credit = np.exp(rng.normal(13.0, 0.60, n) + 0.10 * risk).round(-2)
    goods = (credit * rng.uniform(0.75, 1.0, n)).round(-2)
    annuity = (credit / rng.uniform(12, 60, n)).round(2)
    children = rng.poisson(0.55, n).clip(0, 8)

    # EXT_SOURCE_* are the dominant real Home Credit features: external bureau
    # scores, higher is safer. Correlated with latent risk but noisy.
    ext1 = np.clip(0.5 - 0.16 * risk + rng.normal(0, 0.17, n), 0.01, 0.99)
    ext2 = np.clip(0.5 - 0.20 * risk + rng.normal(0, 0.15, n), 0.01, 0.99)
    ext3 = np.clip(0.5 - 0.14 * risk + rng.normal(0, 0.19, n), 0.01, 0.99)
    # Realistic missingness, and it is not missing-at-random: thin-file
    # applicants are likelier to lack an external score.
    for arr, rate in ((ext1, 0.56), (ext2, 0.02), (ext3, 0.20)):
        miss = rng.random(n) < np.clip(rate + 0.05 * risk, 0.0, 0.95)
        arr[miss] = np.nan

    edu_idx = np.clip((rng.normal(2.0, 1.0, n) - 0.45 * risk).round(), 0, 4).astype(int)
    region_rating = np.clip((rng.normal(2.0, 0.6, n) + 0.25 * risk).round(), 1, 3).astype(int)

    application = pl.DataFrame(
        {
            "SK_ID_CURR": ids,
            "origination_date": origination,
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
            "CODE_GENDER": rng.choice(["F", "M"], n, p=[0.66, 0.34]),
            "FLAG_OWN_CAR": np.where(rng.random(n) < 0.34 - 0.04 * (risk > 0), "Y", "N"),
            "FLAG_OWN_REALTY": np.where(rng.random(n) < 0.69, "Y", "N"),
            "CNT_CHILDREN": children.astype(np.int32),
            "CNT_FAM_MEMBERS": (children + rng.integers(1, 3, n)).astype(np.int32),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods,
            "NAME_EDUCATION_TYPE": np.array(EDUCATION)[edu_idx],
            "NAME_FAMILY_STATUS": rng.choice(FAMILY, n, p=[0.22, 0.55, 0.11, 0.07, 0.05]),
            "OCCUPATION_TYPE": rng.choice(OCCUPATION, n),
            "REGION_RATING_CLIENT": region_rating.astype(np.int32),
            "DAYS_BIRTH": (-age_years * 365.25).astype(np.int32),
            "DAYS_EMPLOYED": (-employed_years * 365.25).astype(np.int32),
            "DAYS_ID_PUBLISH": (-rng.uniform(0, 4500, n)).astype(np.int32),
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
        }
    )

    # ---- outcome ---------------------------------------------------------
    # Log-odds built from the same latent factor plus true feature effects, so
    # a well-specified model can recover roughly a 0.72-0.76 AUC -- in the
    # range real Home Credit models reach, not an implausible 0.95.
    dti = annuity / np.maximum(income, 1.0)
    ltv = credit / np.maximum(goods, 1.0)
    intercept = np.log(cfg.base_bad_rate / (1 - cfg.base_bad_rate))
    logit = (
        intercept
        + 0.95 * risk
        + 2.4 * (dti - np.nanmean(dti))
        + 0.55 * (ltv - 1.0)
        - 0.020 * (age_years - 43)
        - 0.030 * employed_years
        + 0.22 * (region_rating - 2)
        + rng.normal(0, 0.45, n)
    )
    p_bad = 1.0 / (1.0 + np.exp(-logit))
    is_bad = rng.random(n) < p_bad

    # Turn the binary outcome into a delinquency depth so the DPD-based target
    # rule has something real to bite on, including a 30-89 indeterminate band.
    max_dpd = np.zeros(n, dtype=np.int32)
    max_dpd[is_bad] = rng.integers(90, 271, is_bad.sum())
    grey = (~is_bad) & (rng.random(n) < 0.06 + 0.03 * (risk > 0.5))
    max_dpd[grey] = rng.integers(30, 90, grey.sum())
    mild = (~is_bad) & (~grey) & (rng.random(n) < 0.15)
    max_dpd[mild] = rng.integers(1, 30, mild.sum())
    application = application.with_columns(max_dpd_in_window=pl.Series(max_dpd))
    # A CSV read of the real files yields nulls, not NaN, for missing scores.
    # Match that here so downstream null handling is exercised honestly.
    application = application.with_columns(
        pl.col("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3").fill_nan(None)
    )

    tables = {"application": application}
    tables.update(_bureau_tables(rng, ids, risk, origination))
    tables.update(_previous_tables(rng, ids, risk))
    return tables


def _bureau_tables(
    rng: np.random.Generator, ids: np.ndarray, risk: np.ndarray, origination: np.ndarray
) -> dict[str, pl.DataFrame]:
    # Riskier applicants carry more prior credit lines; thin files exist too.
    n_lines = rng.poisson(np.clip(4.0 + 1.1 * risk, 0.3, None)).astype(int)
    total = int(n_lines.sum())
    owner = np.repeat(ids, n_lines)
    owner_risk = np.repeat(risk, n_lines)
    bureau_id = np.arange(500_001, 500_001 + total, dtype=np.int64)

    days_credit = -rng.uniform(30, 2900, total).astype(np.int32)
    active = rng.random(total) < 0.42
    amt_sum = np.exp(rng.normal(11.6, 1.0, total)).round(-1)
    debt = np.where(active, amt_sum * rng.uniform(0.05, 1.05, total), 0.0).round(-1)
    overdue_flag = rng.random(total) < np.clip(0.05 + 0.05 * owner_risk, 0.005, 0.6)
    overdue_amt = np.where(overdue_flag, amt_sum * rng.uniform(0.01, 0.35, total), 0.0).round(-1)

    bureau = pl.DataFrame(
        {
            "SK_ID_CURR": owner,
            "SK_ID_BUREAU": bureau_id,
            "CREDIT_ACTIVE": np.where(active, "Active", "Closed"),
            "CREDIT_TYPE": rng.choice(CREDIT_TYPE, total, p=[0.55, 0.25, 0.1, 0.06, 0.04]),
            "DAYS_CREDIT": days_credit,
            "DAYS_CREDIT_ENDDATE": (days_credit + rng.uniform(180, 2200, total)).astype(np.int32),
            "CREDIT_DAY_OVERDUE": np.where(overdue_flag, rng.integers(1, 180, total), 0).astype(
                np.int32
            ),
            "AMT_CREDIT_SUM": amt_sum,
            "AMT_CREDIT_SUM_DEBT": debt,
            "AMT_CREDIT_SUM_OVERDUE": overdue_amt,
            "CNT_CREDIT_PROLONG": rng.poisson(0.05, total).astype(np.int32),
        }
    )

    # Monthly status strings per bureau line: the raw material for the trend
    # features that carry the most lift in the real dataset.
    months = rng.integers(6, 49, total)
    bb_bureau = np.repeat(bureau_id, months)
    bb_risk = np.repeat(owner_risk, months)
    bb_month = -np.concatenate([np.arange(m) for m in months]).astype(np.int32)
    p_late = np.clip(0.04 + 0.045 * bb_risk, 0.002, 0.65)
    draw = rng.random(bb_month.size)
    status = np.where(
        draw < p_late * 0.45,
        "1",
        np.where(
            draw < p_late * 0.70,
            "2",
            np.where(draw < p_late * 0.85, "3", np.where(draw < p_late, "5", "C")),
        ),
    )
    status = np.where(rng.random(bb_month.size) < 0.18, "X", status)

    bureau_balance = pl.DataFrame(
        {"SK_ID_BUREAU": bb_bureau, "MONTHS_BALANCE": bb_month, "STATUS": status}
    )
    return {"bureau": bureau, "bureau_balance": bureau_balance}


def _previous_tables(
    rng: np.random.Generator, ids: np.ndarray, risk: np.ndarray
) -> dict[str, pl.DataFrame]:
    n_prev = rng.poisson(np.clip(3.2 + 0.5 * risk, 0.2, None)).astype(int)
    total = int(n_prev.sum())
    owner = np.repeat(ids, n_prev)
    owner_risk = np.repeat(risk, n_prev)
    prev_id = np.arange(900_001, 900_001 + total, dtype=np.int64)

    p_refused = np.clip(0.14 + 0.08 * owner_risk, 0.01, 0.85)
    draw = rng.random(total)
    status = np.where(
        draw < p_refused,
        "Refused",
        np.where(
            draw < p_refused + 0.08,
            "Canceled",
            np.where(draw < p_refused + 0.13, "Unused offer", "Approved"),
        ),
    )
    amt_app = np.exp(rng.normal(12.4, 0.8, total)).round(-2)

    previous_application = pl.DataFrame(
        {
            "SK_ID_PREV": prev_id,
            "SK_ID_CURR": owner,
            "NAME_CONTRACT_TYPE": rng.choice(
                ["Cash loans", "Consumer loans", "Revolving loans"], total
            ),
            "NAME_CONTRACT_STATUS": status,
            "DAYS_DECISION": (-rng.uniform(30, 2800, total)).astype(np.int32),
            "AMT_APPLICATION": amt_app,
            "AMT_CREDIT": np.where(
                status == "Approved", amt_app * rng.uniform(0.7, 1.1, total), 0.0
            ).round(-2),
            "AMT_ANNUITY": (amt_app / rng.uniform(12, 48, total)).round(2),
            "CNT_PAYMENT": rng.integers(6, 61, total).astype(np.int32),
        }
    )

    approved = prev_id[status == "Approved"]
    appr_risk = owner_risk[status == "Approved"]
    appr_owner = owner[status == "Approved"]

    # installments_payments: the source of late-payment counts and payment
    # shortfall ratios.
    n_inst = rng.integers(4, 25, approved.size)
    inst_prev = np.repeat(approved, n_inst)
    inst_curr = np.repeat(appr_owner, n_inst)
    inst_risk = np.repeat(appr_risk, n_inst)
    k = inst_prev.size
    days_inst = -rng.uniform(30, 2500, k)
    late_days = np.where(
        rng.random(k) < np.clip(0.13 + 0.07 * inst_risk, 0.01, 0.8),
        rng.gamma(2.0, 9.0, k),
        -rng.uniform(0, 12, k),
    )
    amt_inst = np.exp(rng.normal(9.6, 0.7, k)).round(2)
    shortfall = np.where(
        rng.random(k) < np.clip(0.06 + 0.05 * inst_risk, 0.002, 0.6),
        rng.uniform(0.35, 0.98, k),
        1.0,
    )

    installments_payments = pl.DataFrame(
        {
            "SK_ID_PREV": inst_prev,
            "SK_ID_CURR": inst_curr,
            "NUM_INSTALMENT_NUMBER": np.concatenate([np.arange(1, m + 1) for m in n_inst]).astype(
                np.int32
            ),
            "DAYS_INSTALMENT": days_inst.astype(np.int32),
            "DAYS_ENTRY_PAYMENT": (days_inst + late_days).astype(np.int32),
            "AMT_INSTALMENT": amt_inst,
            "AMT_PAYMENT": (amt_inst * shortfall).round(2),
        }
    )

    # credit_card_balance: utilisation trajectory lives here.
    is_card = rng.random(approved.size) < 0.3
    card_prev = approved[is_card]
    card_owner = appr_owner[is_card]
    card_risk = appr_risk[is_card]
    n_mo = rng.integers(6, 37, card_prev.size)
    cb_prev = np.repeat(card_prev, n_mo)
    cb_curr = np.repeat(card_owner, n_mo)
    cb_risk = np.repeat(card_risk, n_mo)
    cb_month = -np.concatenate([np.arange(m) for m in n_mo]).astype(np.int32)
    limit = np.repeat(np.exp(rng.normal(11.5, 0.6, card_prev.size)).round(-2), n_mo)
    # Utilisation drifts up over time for risky borrowers: the trend signal.
    util = np.clip(
        rng.normal(0.42, 0.22, cb_month.size) + 0.11 * cb_risk + 0.010 * cb_month * cb_risk,
        0.0,
        1.35,
    )

    credit_card_balance = pl.DataFrame(
        {
            "SK_ID_PREV": cb_prev,
            "SK_ID_CURR": cb_curr,
            "MONTHS_BALANCE": cb_month,
            "AMT_BALANCE": (limit * util).round(2),
            "AMT_CREDIT_LIMIT_ACTUAL": limit,
            "AMT_DRAWINGS_CURRENT": (limit * rng.uniform(0, 0.3, cb_month.size)).round(2),
            "AMT_PAYMENT_CURRENT": (limit * rng.uniform(0, 0.25, cb_month.size)).round(2),
            "SK_DPD": np.where(
                rng.random(cb_month.size) < np.clip(0.05 + 0.05 * cb_risk, 0.002, 0.5),
                rng.integers(1, 150, cb_month.size),
                0,
            ).astype(np.int32),
        }
    )

    # POS_CASH_balance
    is_pos = ~is_card
    pos_prev_ids = approved[is_pos]
    pos_owner = appr_owner[is_pos]
    pos_risk = appr_risk[is_pos]
    n_pm = rng.integers(4, 31, pos_prev_ids.size)
    pc_prev = np.repeat(pos_prev_ids, n_pm)
    pc_curr = np.repeat(pos_owner, n_pm)
    pc_risk = np.repeat(pos_risk, n_pm)
    pc_month = -np.concatenate([np.arange(m) for m in n_pm]).astype(np.int32)

    pos_cash_balance = pl.DataFrame(
        {
            "SK_ID_PREV": pc_prev,
            "SK_ID_CURR": pc_curr,
            "MONTHS_BALANCE": pc_month,
            "CNT_INSTALMENT": rng.integers(6, 61, pc_month.size).astype(np.int32),
            "CNT_INSTALMENT_FUTURE": rng.integers(0, 40, pc_month.size).astype(np.int32),
            "NAME_CONTRACT_STATUS": rng.choice(
                ["Active", "Completed", "Signed"], pc_month.size, p=[0.6, 0.35, 0.05]
            ),
            "SK_DPD": np.where(
                rng.random(pc_month.size) < np.clip(0.04 + 0.05 * pc_risk, 0.002, 0.5),
                rng.integers(1, 150, pc_month.size),
                0,
            ).astype(np.int32),
        }
    )

    return {
        "previous_application": previous_application,
        "installments_payments": installments_payments,
        "credit_card_balance": credit_card_balance,
        "POS_CASH_balance": pos_cash_balance,
    }


def write(tables: dict[str, pl.DataFrame], out_dir=SYNTHETIC) -> dict[str, str]:
    """Persist as parquet -- 5-10x smaller and far faster to reload than CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        df.write_parquet(path)
        written[name] = f"{df.height:,} rows x {df.width} cols -> {path.name}"
    return written
