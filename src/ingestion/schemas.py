"""Pandera schemas enforced at every dataframe boundary.

These are contracts, not documentation. A silently renamed column or a bureau
row with a positive ``DAYS_CREDIT`` (a credit line opened in the future) should
stop the pipeline at ingestion, not surface three weeks later as an unexplained
drop in AUC.

Coverage is deliberately uneven: columns the model depends on are constrained
tightly, pass-through columns loosely.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class ApplicationSchema(pa.DataFrameModel):
    """Applicant-level table: one row per credit application."""

    SK_ID_CURR: int = pa.Field(unique=True, nullable=False)
    origination_date: pl.Date = pa.Field(nullable=False)
    NAME_CONTRACT_TYPE: str = pa.Field(isin=["Cash loans", "Revolving loans"])
    CODE_GENDER: str = pa.Field(isin=["F", "M", "XNA"])
    CNT_CHILDREN: int = pa.Field(ge=0, le=20)
    AMT_INCOME_TOTAL: float = pa.Field(gt=0)
    AMT_CREDIT: float = pa.Field(gt=0)
    AMT_ANNUITY: float = pa.Field(gt=0, nullable=True)
    AMT_GOODS_PRICE: float = pa.Field(gt=0, nullable=True)
    # Home Credit encodes these as negative days relative to application date.
    DAYS_BIRTH: int = pa.Field(le=0)
    EXT_SOURCE_1: float = pa.Field(ge=0, le=1, nullable=True)
    EXT_SOURCE_2: float = pa.Field(ge=0, le=1, nullable=True)
    EXT_SOURCE_3: float = pa.Field(ge=0, le=1, nullable=True)
    max_dpd_in_window: int = pa.Field(ge=0, nullable=True)

    class Config:
        strict = False
        coerce = True

    @pa.check("DAYS_BIRTH", name="plausible_applicant_age")
    def age_between_18_and_100(cls, data):
        """Reject ages outside 18-100; ECOA makes age a protected attribute,
        so a nonsense value here would silently poison the fairness report."""
        years = -pl.col(data.key) / 365.25
        return data.lazyframe.select((years >= 18) & (years <= 100))


class BureauSchema(pa.DataFrameModel):
    SK_ID_CURR: int = pa.Field(nullable=False)
    SK_ID_BUREAU: int = pa.Field(unique=True, nullable=False)
    CREDIT_ACTIVE: str = pa.Field(isin=["Active", "Closed", "Sold", "Bad debt"])
    DAYS_CREDIT: int = pa.Field(le=0, description="Never positive: no future-dated credit lines.")
    AMT_CREDIT_SUM: float = pa.Field(ge=0, nullable=True)
    AMT_CREDIT_SUM_DEBT: float = pa.Field(nullable=True)
    AMT_CREDIT_SUM_OVERDUE: float = pa.Field(ge=0, nullable=True)
    CREDIT_DAY_OVERDUE: int = pa.Field(ge=0)

    class Config:
        strict = False
        coerce = True


class BureauBalanceSchema(pa.DataFrameModel):
    SK_ID_BUREAU: int = pa.Field(nullable=False)
    MONTHS_BALANCE: int = pa.Field(le=0)
    STATUS: str = pa.Field(isin=["C", "X", "0", "1", "2", "3", "4", "5"])

    class Config:
        strict = False
        coerce = True


class PreviousApplicationSchema(pa.DataFrameModel):
    SK_ID_PREV: int = pa.Field(unique=True, nullable=False)
    SK_ID_CURR: int = pa.Field(nullable=False)
    NAME_CONTRACT_STATUS: str = pa.Field(isin=["Approved", "Refused", "Canceled", "Unused offer"])
    DAYS_DECISION: int = pa.Field(le=0)
    AMT_APPLICATION: float = pa.Field(ge=0, nullable=True)

    class Config:
        strict = False
        coerce = True


class InstallmentsSchema(pa.DataFrameModel):
    SK_ID_PREV: int = pa.Field(nullable=False)
    SK_ID_CURR: int = pa.Field(nullable=False)
    DAYS_INSTALMENT: int = pa.Field(le=0)
    AMT_INSTALMENT: float = pa.Field(ge=0, nullable=True)
    AMT_PAYMENT: float = pa.Field(ge=0, nullable=True)

    class Config:
        strict = False
        coerce = True


class CreditCardBalanceSchema(pa.DataFrameModel):
    SK_ID_PREV: int = pa.Field(nullable=False)
    SK_ID_CURR: int = pa.Field(nullable=False)
    MONTHS_BALANCE: int = pa.Field(le=0)
    AMT_CREDIT_LIMIT_ACTUAL: float = pa.Field(ge=0, nullable=True)
    SK_DPD: int = pa.Field(ge=0)

    class Config:
        strict = False
        coerce = True


class PosCashSchema(pa.DataFrameModel):
    SK_ID_PREV: int = pa.Field(nullable=False)
    SK_ID_CURR: int = pa.Field(nullable=False)
    MONTHS_BALANCE: int = pa.Field(le=0)
    SK_DPD: int = pa.Field(ge=0)

    class Config:
        strict = False
        coerce = True


SCHEMAS: dict[str, type[pa.DataFrameModel]] = {
    "application": ApplicationSchema,
    "bureau": BureauSchema,
    "bureau_balance": BureauBalanceSchema,
    "previous_application": PreviousApplicationSchema,
    "installments_payments": InstallmentsSchema,
    "credit_card_balance": CreditCardBalanceSchema,
    "POS_CASH_balance": PosCashSchema,
}
