"""Precompute applicant history features into the serving cache.

The 161 relational features cost a scan of 9.6M child rows to compute. Doing
that per request is impossible inside a 150ms budget, so they are computed once
here — the same aggregations the training pipeline uses — and looked up by
``SK_ID_CURR`` at inference.

Only features that appear in the feature spec are cached. Storing the other
149 would triple the memory for values no model reads.
"""

from __future__ import annotations

from src.api.cache import FeatureCache
from src.api.settings import get_settings
from src.features.application import EXT_SOURCES
from src.features.build import build
from src.features.spec import FeatureSpec

# Application-level features are derived from the request payload at scoring
# time, so caching them would be both redundant and wrong -- the cached value
# would be from whenever the warm ran, not from this application.
_APPLICATION_PREFIXES = ("RATIO_", "STAB_", "EXT_")
_APPLICATION_COLUMNS = frozenset(
    {
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
        *EXT_SOURCES,
    }
)


def history_features(spec: FeatureSpec) -> list[str]:
    """Spec features that come from relational history rather than the payload."""
    return [
        f
        for f in spec.features
        if f not in _APPLICATION_COLUMNS and not f.startswith(_APPLICATION_PREFIXES)
    ]


def warm_into(cache: FeatureCache, spec: FeatureSpec, limit: int | None = None) -> int:
    """Populate a specific cache instance. Returns rows written."""
    columns = history_features(spec)
    frame = build().frame.select(["SK_ID_CURR", *columns])
    if limit is not None:
        frame = frame.head(limit)

    rows: dict[int, dict[str, float | None]] = {}
    for record in frame.collect().iter_rows(named=True):
        key = int(record.pop("SK_ID_CURR"))
        rows[key] = {
            name: (None if value is None else float(value)) for name, value in record.items()
        }
    return cache.put_many(rows)


def warm(limit: int | None = None) -> tuple[int, int, str]:
    """Populate the configured cache. Returns (rows, features per row, backend)."""
    settings = get_settings()
    spec = FeatureSpec.load()
    cache = FeatureCache.connect(settings.redis_url)

    columns = history_features(spec)
    written = warm_into(cache, spec, limit)
    return written, len(columns), cache.backend


def sample_payloads(n: int = 20) -> list[dict]:
    """Realistic request bodies drawn from the out-of-time fold.

    Used by the load test and the smoke check so both exercise the cache-hit
    path rather than scoring thin-file strangers.
    """
    from src.config import SYNTHETIC_SPLIT, SYNTHETIC_TARGET
    from src.ingestion.loaders import load
    from src.ingestion.splits import split_by_time
    from src.ingestion.target import assign_labels_from_dpd, modelling_population

    pop = modelling_population(assign_labels_from_dpd(load("application"), SYNTHETIC_TARGET))
    test = split_by_time(pop, SYNTHETIC_SPLIT).test.head(n).collect()

    payloads = []
    for row in test.iter_rows(named=True):
        payloads.append(
            {
                "sk_id_curr": int(row["SK_ID_CURR"]),
                "amt_income_total": float(row["AMT_INCOME_TOTAL"]),
                "amt_credit": float(row["AMT_CREDIT"]),
                "amt_annuity": float(row["AMT_ANNUITY"]),
                "amt_goods_price": float(row["AMT_GOODS_PRICE"]),
                "days_birth": int(row["DAYS_BIRTH"]),
                "days_employed": int(max(row["DAYS_EMPLOYED"], -36525)),
                "days_id_publish": int(row["DAYS_ID_PUBLISH"]),
                "cnt_children": int(row["CNT_CHILDREN"]),
                "cnt_fam_members": int(max(row["CNT_FAM_MEMBERS"], row["CNT_CHILDREN"] + 1)),
                "region_rating_client": int(row["REGION_RATING_CLIENT"]),
                "name_contract_type": row["NAME_CONTRACT_TYPE"],
                "code_gender": row["CODE_GENDER"],
                "flag_own_car": row["FLAG_OWN_CAR"],
                "flag_own_realty": row["FLAG_OWN_REALTY"],
                "name_education_type": row["NAME_EDUCATION_TYPE"],
                "name_family_status": row["NAME_FAMILY_STATUS"],
                "occupation_type": row["OCCUPATION_TYPE"],
                "ext_source_1": _opt(row["EXT_SOURCE_1"]),
                "ext_source_2": _opt(row["EXT_SOURCE_2"]),
                "ext_source_3": _opt(row["EXT_SOURCE_3"]),
            }
        )
    return payloads


def _opt(value) -> float | None:
    return None if value is None else float(value)


def write_sample_payloads(n: int = 200) -> int:
    """Persist sample payloads for the load test to replay."""
    import json

    from src.config import ARTIFACTS

    payloads = sample_payloads(n)
    (ARTIFACTS / "sample_payloads.json").write_text(json.dumps(payloads))
    # A single pretty-printed payload, so the README's curl example is a real
    # file the reader can actually post rather than something to retype.
    (ARTIFACTS / "one_payload.json").write_text(json.dumps(payloads[0], indent=2))
    return len(payloads)


if __name__ == "__main__":
    written, n_features, backend = warm()
    print(f"cached {written:,} applicants x {n_features} history features into {backend}")
    print(f"wrote {write_sample_payloads()} sample payloads")
