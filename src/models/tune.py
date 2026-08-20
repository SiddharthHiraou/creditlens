"""Optuna hyperparameter search with TPE and pruning.

Bayesian search over the space, not a grid. Unpromising trials are pruned at
intermediate boosting rounds rather than run to completion, which is where most
of the wall-clock saving comes from.

Two rules the search obeys:

**Tune against validation, never test.** The out-of-time test fold is touched
exactly once, after the champion is chosen. A hyperparameter search that peeks
at test produces a number nobody can reproduce on next quarter's book.

**Monotonic constraints are fixed, not searched.** They encode domain knowledge
and regulatory expectation; letting the optimiser trade them away for AUC would
defeat the purpose of having them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

from src.config import RANDOM_SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    n_pruned: int
    study: optuna.Study

    def trials_dataframe(self):
        import polars as pl

        rows = [
            {
                "number": t.number,
                "value": t.value,
                "state": str(t.state.name),
                **{f"param_{k}": v for k, v in t.params.items()},
            }
            for t in self.study.trials
        ]
        return pl.DataFrame(rows)


def _lightgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def tune_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    monotone: list[int] | None = None,
    n_trials: int = 100,
    timeout: int | None = None,
    seed: int = RANDOM_SEED,
) -> TuningResult:
    """TPE search maximising validation AUC, with median pruning."""
    import lightgbm as lgb

    from src.models.gbdt import scale_pos_weight

    spw = scale_pos_weight(y_train)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 3000,
            "scale_pos_weight": spw,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": seed,
            **_lightgbm_space(trial),
        }
        if monotone is not None and any(monotone):
            params["monotone_constraints"] = monotone
            # See the note in gbdt.fit_lightgbm: 'basic' is the method that
            # actually guarantees the constraint.
            params["monotone_constraints_method"] = "basic"

        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_train,
            y_train,
            eval_X=x_valid,
            eval_y=y_valid,
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
                # Report at each round so the pruner can kill a bad trial early.
                LightGBMPruningCallback(trial, "auc"),
            ],
        )
        return float(roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1]))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed, n_startup_trials=15),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=50, interval_steps=25),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))

    pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    return TuningResult(
        best_params=study.best_params,
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        n_pruned=pruned,
        study=study,
    )


class LightGBMPruningCallback:
    """Feed the validation metric to Optuna each round so trials can be pruned.

    Written out rather than using ``optuna.integration`` so the project does
    not depend on the optional integrations package, which lags LightGBM
    releases and has broken across versions before.
    """

    def __init__(self, trial: optuna.Trial, metric: str, valid_name: str = "valid_0"):
        self.trial = trial
        self.metric = metric
        self.valid_name = valid_name

    def __call__(self, env) -> None:
        for name, metric, value, _ in env.evaluation_result_list:
            if name == self.valid_name and metric == self.metric:
                self.trial.report(value, step=env.iteration)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
                return
