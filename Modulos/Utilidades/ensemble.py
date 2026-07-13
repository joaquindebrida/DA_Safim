"""
ensemble.py
─────────────────────────────────────────────────────────────────────────────
Anomaly detection ensemble: training, scoring and prediction utilities.

Public API
──────────
    robust_scale_score(score, p5, p95)
        → scaled, p5, p95

    fit_ensemble(df, features, modelos, scaler_class)
        → pipeline : dict

    predict_ensemble(pipeline, df_new)
        → output : pd.DataFrame

Structure of pipeline dict
──────────────────────────
    features            : list[str]
    scaler              : fitted scaler instance
    modelos             : dict {name: fitted model}
    score_cols          : list[str]
    scaling_params      : dict {col: {p5, p95}}
    p80_scores          : pd.Series
    train_composite_scores : pd.Series
    resultados_train    : pd.DataFrame  (raw scores on train)
    df_scores_train     : pd.DataFrame  (scaled scores on train)
    df_labels_train     : pd.DataFrame  (binary labels on train)
    X_train_scaled      : np.ndarray

Output columns of predict_ensemble
───────────────────────────────────
    Score_IF_raw  Score_KNN_raw  Score_PCA_raw  Score_LOF_raw
    Score_IF_label  Score_KNN_label  Score_PCA_label  Score_LOF_label
    Composite_label
    Score_IF  Score_KNN  Score_PCA  Score_LOF
    Composite_score
    Risk_percentile
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1. SCORE SCALING
# ─────────────────────────────────────────────────────────────────────────────

def robust_scale_score(
    score: np.ndarray,
    p5:  float | None = None,
    p95: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Robust min-max scaling using the 5th and 95th percentiles as anchors.
    Output is clipped to [0, 1].

    Parameters
    ----------
    score : array-like
        Raw anomaly scores.
    p5 : float or None
        Lower anchor. If None, computed from `score`.
    p95 : float or None
        Upper anchor. If None, computed from `score`.

    Returns
    -------
    scaled : np.ndarray  clipped to [0, 1]
    p5     : float
    p95    : float
    """
    score = np.asarray(score, dtype=float)

    if p5  is None: p5  = np.percentile(score, 5)
    if p95 is None: p95 = np.percentile(score, 95)

    denom = p95 - p5

    if denom == 0:
        scaled = np.zeros_like(score)
    else:
        scaled = (score - p5) / denom

    return np.clip(scaled, 0, 1), p5, p95


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def fit_ensemble(
    df: pd.DataFrame,
    features: list[str],
    modelos: dict,
    scaler_class=StandardScaler,
) -> dict:
    """
    Train an ensemble of anomaly detection models on df[features].

    Parameters
    ----------
    df : pd.DataFrame
        Training set. All rows are assumed to be normal (no labels needed).
    features : list of str
        Feature columns to use.
    modelos : dict {name: model_instance}
        Models must implement .fit(X) and expose .decision_scores_ after fitting.
        Compatible with PyOD models (IForest, KNN, PCA, LOF, etc.).
    scaler_class : sklearn scaler class
        Default: StandardScaler. Pass RobustScaler if preferred.

    Returns
    -------
    pipeline : dict
        All objects needed for prediction and explainability.
    """
    X = df[features].copy()

    # ── Scale ─────────────────────────────────────────────────────
    scaler   = scaler_class()
    X_scaled = scaler.fit_transform(X)

    # ── Train models ──────────────────────────────────────────────
    resultados = pd.DataFrame(index=df.index)
    score_cols = []

    for nombre, modelo in modelos.items():
        modelo.fit(X_scaled)
        col = f"Score_{nombre}"
        resultados[col] = modelo.decision_scores_
        score_cols.append(col)

    # ── Scale scores ──────────────────────────────────────────────
    df_scores      = pd.DataFrame(index=df.index)
    scaling_params = {}

    for col in score_cols:
        scaled, p5, p95    = robust_scale_score(resultados[col])
        df_scores[col]     = scaled
        scaling_params[col] = {"p5": p5, "p95": p95}

    df_scores["Composite_score"] = df_scores[score_cols].mean(axis=1)

    # ── Labels (p80 threshold per model) ─────────────────────────
    p80_scores = resultados[score_cols].quantile(0.80)
    df_labels  = resultados[score_cols].ge(p80_scores).astype(int)
    df_labels["Anomalia"] = df_labels[score_cols].sum(axis=1)

    # ── Pipeline ──────────────────────────────────────────────────
    pipeline = {
        "features":               features,
        "scaler":                 scaler,
        "modelos":                modelos,
        "score_cols":             score_cols,
        "scaling_params":         scaling_params,
        "p80_scores":             p80_scores,
        "train_composite_scores": df_scores["Composite_score"],
        "resultados_train":       resultados,
        "df_scores_train":        df_scores,
        "df_labels_train":        df_labels,
        "X_train_scaled":         X_scaled,
        "df_train":               X
    }

    _print_fit_summary(pipeline, len(df))

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_ensemble(
    pipeline: dict,
    df_new: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score new observations using a trained ensemble pipeline.

    Parameters
    ----------
    pipeline : dict
        Output of fit_ensemble().
    df_new : pd.DataFrame
        New data. Must contain all columns in pipeline['features'].

    Returns
    -------
    output : pd.DataFrame
        One row per observation. See module docstring for column list.
    """
    features               = pipeline["features"]
    scaler                 = pipeline["scaler"]
    modelos                = pipeline["modelos"]
    score_cols             = pipeline["score_cols"]
    scaling_params         = pipeline["scaling_params"]
    p80_scores             = pipeline["p80_scores"]
    train_composite_scores = pipeline["train_composite_scores"]

    # ── Validate features ─────────────────────────────────────────
    missing = [f for f in features if f not in df_new.columns]
    if missing:
        raise ValueError(f"Missing features in df_new: {missing}")

    # ── Scale ─────────────────────────────────────────────────────
    X        = df_new[features].copy()
    X_scaled = scaler.transform(X)

    # ── Raw scores ────────────────────────────────────────────────
    resultados = pd.DataFrame(index=df_new.index)

    for nombre, modelo in modelos.items():
        col = f"Score_{nombre}"
        resultados[col] = modelo.decision_function(X_scaled)

    # ── Labels ────────────────────────────────────────────────────
    df_labels = resultados[score_cols].ge(p80_scores).astype(int)
    df_labels["Anomalia_compuesta"] = df_labels[score_cols].sum(axis=1)

    # ── Scaled scores ─────────────────────────────────────────────
    df_scores = pd.DataFrame(index=df_new.index)

    for col in score_cols:
        p5  = scaling_params[col]["p5"]
        p95 = scaling_params[col]["p95"]
        scaled, _, _  = robust_scale_score(resultados[col], p5, p95)
        df_scores[col] = scaled

    df_scores["Composite_score"] = df_scores[score_cols].mean(axis=1)

    # Percentile vs training distribution
    df_scores["Risk_percentile"] = df_scores["Composite_score"].apply(
        lambda x: (train_composite_scores <= x).mean() * 100
    )

    # ── Assemble output ───────────────────────────────────────────
    output = pd.concat([resultados, df_labels, df_scores], axis=1)

    n_models = len(modelos)
    new_names = (
        [f"Score_{n}_raw"   for n in modelos] +   # raw scores
        [f"Score_{n}_label" for n in modelos] +   # binary labels
        ["Composite_label"] +                      # sum of labels
        [f"Score_{n}"       for n in modelos] +   # scaled scores
        ["Composite_score", "Risk_percentile"]
    )

    if len(new_names) != len(output.columns):
        warnings.warn(
            f"Column rename mismatch: expected {len(new_names)}, "
            f"got {len(output.columns)}. Skipping rename."
        )
    else:
        output.columns = new_names

    return output


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print_fit_summary(pipeline: dict, n_train: int) -> None:
    W = 52
    score_cols = pipeline["score_cols"]
    p80        = pipeline["p80_scores"]
    composite  = pipeline["train_composite_scores"]

    print("=" * W)
    print("  ENSEMBLE ENTRENADO")
    print("=" * W)
    print(f"  Planes de entrenamiento : {n_train}")
    print(f"  Features                : {len(pipeline['features'])}")
    print(f"  Modelos                 : {', '.join(pipeline['modelos'].keys())}")
    print(f"  {'─'*40}")
    print(f"  Composite score — train")
    print(f"    media  : {composite.mean():.4f}")
    print(f"    p50    : {composite.quantile(0.50):.4f}")
    print(f"    p80    : {composite.quantile(0.80):.4f}")
    print(f"    p95    : {composite.quantile(0.95):.4f}")
    print("=" * W)
