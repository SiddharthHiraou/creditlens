"""API configuration, all overridable by environment variable."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import ARTIFACTS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CREDITLENS_", env_file=".env", extra="ignore")

    app_name: str = "CreditLens"
    version: str = "0.4.0"
    environment: str = "local"

    # Storage. Both fall back to something that works with nothing running, so
    # `uvicorn src.api.main:app` is a viable dev loop without docker compose.
    database_url: str = Field(default=f"sqlite:///{ARTIFACTS / 'creditlens.db'}")
    redis_url: str | None = None

    # Auth. Keys are seeded here for local use only; in any real deployment
    # they come from the environment and are hashed at rest.
    api_keys: str = "demo-key-underwriter,demo-key-risk"
    rate_limit_per_minute: int = 600

    # With Redis the cache is shared and warmed out of band. The in-process
    # fallback starts empty, so warm it at startup or every applicant looks
    # thin-file -- a silently wrong answer rather than a visible failure.
    warm_cache_on_startup: bool = True
    batch_max_rows: int = 1000
    shap_top_k: int = 10
    use_onnx: bool = True
    # SHAP needs the booster loaded, which is the only reason the serving image
    # carries CatBoost. Disabling it gives a decision-only tier that serves from
    # ONNX alone -- faster, and with no reason codes.
    enable_shap: bool = True

    @property
    def key_set(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
