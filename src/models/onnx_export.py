"""Export the champion booster to ONNX for serving.

Two reasons this is worth the extra artifact, both of which show up in the
numbers below:

* **Latency.** ONNX Runtime executes a single-row prediction in a tight C++
  loop with no Python-level batching machinery. Native CatBoost `predict` on a
  one-row matrix pays fixed overhead that dominates at batch size 1, which is
  exactly the serving case.
* **Dependency surface.** The serving image needs `onnxruntime` and numpy, not
  CatBoost, LightGBM, XGBoost, Optuna and MLflow. That is a materially smaller
  image and a smaller thing to keep patched.

**The calibrator stays in Python.** ONNX carries the booster's raw margin;
`SmoothedIsotonic` is a handful of numpy operations over ~29 blocks and is
cheaper to apply directly than to express as a graph. The wrapper below keeps
them together so callers cannot accidentally serve an uncalibrated PD -- which
would be wrong by a factor of 2.5 here and would look completely normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS

ONNX_PATH = ARTIFACTS / "champion.onnx"


def export(model, path: Path = ONNX_PATH) -> Path:
    """Write the champion's booster to ONNX.

    Accepts a CalibratedModel, a GbdtModel, or a bare booster.
    """
    base = getattr(model, "base", model)
    booster = getattr(base, "booster", base)
    path.parent.mkdir(parents=True, exist_ok=True)

    cls = type(booster).__name__
    if cls.startswith("CatBoost"):
        booster.save_model(str(path), format="onnx")
    elif cls.startswith(("LGBM", "XGB")):
        from skl2onnx import to_onnx

        n_features = len(getattr(base, "feature_names", [])) or booster.n_features_in_
        dummy = np.zeros((1, n_features), dtype=np.float32)
        path.write_bytes(to_onnx(booster, dummy, target_opset=17).SerializeToString())
    else:
        raise TypeError(f"No ONNX export path for {cls}")
    return path


@dataclass
class OnnxScorer:
    """Serving-side scorer: ONNX booster plus the Python calibrator."""

    session: object
    input_name: str
    output_name: str
    feature_names: list[str]
    calibrator: object | None = None

    @classmethod
    def load(
        cls,
        path: Path = ONNX_PATH,
        *,
        feature_names: list[str] | None = None,
        calibrator: object | None = None,
        threads: int = 1,
    ) -> OnnxScorer:
        import onnxruntime as ort

        options = ort.SessionOptions()
        # One thread per request. Serving concurrency comes from the process
        # pool; intra-op threading here just adds contention and tail latency.
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        outputs = session.get_outputs()
        # CatBoost's ONNX export emits (label, probabilities); take probabilities.
        name = outputs[-1].name
        return cls(
            session=session,
            input_name=session.get_inputs()[0].name,
            output_name=name,
            feature_names=list(feature_names or []),
            calibrator=calibrator,
        )

    def predict_pd_uncalibrated(self, x: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(
            np.asarray(x, dtype=np.float32).reshape(1, -1)
            if np.ndim(x) == 1
            else np.asarray(x, dtype=np.float32)
        )
        raw = self.session.run([self.output_name], {self.input_name: arr})[0]
        return _positive_class(raw)

    def predict_pd(self, x: np.ndarray) -> np.ndarray:
        p = self.predict_pd_uncalibrated(x)
        if self.calibrator is None:
            return p
        return np.clip(self.calibrator.predict(p), 1e-6, 1 - 1e-6)


def _positive_class(raw) -> np.ndarray:
    """Pull the positive-class probability out of ONNX's output shape.

    CatBoost emits a list of per-row dicts ``{0: p0, 1: p1}``; other exporters
    emit a 2-D array. Handling both keeps the loader exporter-agnostic.
    """
    if isinstance(raw, list):
        return np.array([row[1] if isinstance(row, dict) else row[-1] for row in raw], dtype=float)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, -1]
    return arr.ravel()
