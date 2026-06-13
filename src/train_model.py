"""
Train the hygiene-risk model, evaluate it honestly, and save the artifact the
Streamlit app loads.

Design notes (defensible in an interview):
  * Target is rare (~4.5% need improvement), so we judge the model on
    ROC-AUC and average precision (PR-AUC) -- NOT accuracy, which a
    predict-everything-passes model would ace.
  * RandomForest with class_weight='balanced' so the rare positives aren't
    drowned out.
  * Honest framing: contextual features (type, region, recency) only shift risk
    moderately, so the model is a *screening aid*, not an oracle. We report the
    real numbers and let them speak.

Run (after prepare_data.py):
    python src/train_model.py
Outputs:
    model/model.joblib           (pipeline + metadata for the app)
    figures/*.png                (ROC, PR, calibration, importances, SHAP)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_table.parquet"
MODEL_OUT = ROOT / "model" / "model.joblib"
FIGDIR = ROOT / "figures"

CAT = ["business_type", "region", "settlement_type"]
# years_since_rating is deliberately EXCLUDED: it's a strong predictor only via
# reverse causation (failing places get re-inspected sooner), and a prospective
# business has no inspection history. We model only what's knowable up front.
NUM: list[str] = []
NAVY, RED, GREY = "#1f3a5f", "#c0392b", "#95a5a6"
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def build_pipeline() -> Pipeline:
    transformers = [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)]
    if NUM:
        transformers.append(("num", "passthrough", NUM))
    pre = ColumnTransformer(transformers)
    # No class_weight: we want predict_proba CALIBRATED to the true ~4.5% base
    # rate so the app's risk % is meaningful. Balancing would inflate every
    # probability toward 50%. We handle imbalance at scoring time via the
    # operating threshold instead.
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=14, min_samples_leaf=25,
        random_state=42, n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def plot_roc(y, p, auc):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=NAVY, lw=2, label=f"model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="random")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve", fontweight="bold"); ax.legend()
    fig.savefig(FIGDIR / "roc_curve.png"); plt.close(fig)


def plot_pr(y, p, ap, base):
    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, color=NAVY, lw=2, label=f"model (AP = {ap:.3f})")
    ax.axhline(base, color=GREY, ls="--", lw=1, label=f"base rate ({base:.1%})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision-recall curve", fontweight="bold"); ax.legend()
    fig.savefig(FIGDIR / "pr_curve.png"); plt.close(fig)


def plot_calibration(y, p):
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean, frac, "o-", color=NAVY, label="model")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="perfect")
    ax.set_xlabel("predicted risk"); ax.set_ylabel("observed failure rate")
    ax.set_title("Calibration", fontweight="bold"); ax.legend()
    fig.savefig(FIGDIR / "calibration.png"); plt.close(fig)


def grouped_importance(pipe) -> dict[str, float]:
    """Sum one-hot importances back up to the original feature."""
    ohe_names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pipe.named_steps["clf"].feature_importances_
    grouped: dict[str, float] = {}
    for name, val in zip(ohe_names, imp):
        key = next((orig for orig in CAT + NUM if orig in name), name)
        grouped[key] = grouped.get(key, 0.0) + float(val)
    return grouped


def plot_importance(grouped: dict[str, float]):
    s = pd.Series(grouped).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(s.index, s.values, color=NAVY)
    ax.set_xlabel("total importance"); ax.set_title("Feature importance", fontweight="bold")
    fig.savefig(FIGDIR / "feature_importance.png"); plt.close(fig)


def plot_shap(pipe, X_sample):
    try:
        import shap
        pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
        Xt = pre.transform(X_sample)
        names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        vals = sv[1] if isinstance(sv, list) else sv
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        shap.summary_plot(vals, Xt, feature_names=names, show=False, max_display=12)
        plt.title("SHAP — drivers of predicted risk", fontweight="bold")
        plt.savefig(FIGDIR / "shap_summary.png", bbox_inches="tight", dpi=130)
        plt.close()
        print("  saved SHAP summary")
    except Exception as exc:  # noqa: BLE001 - SHAP is optional, never fail training
        print(f"  (skipped SHAP: {exc})")


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    MODEL_OUT.parent.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA)
    X, y = df[CAT + NUM], df["needs_improvement"]
    base = y.mean()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipe = build_pipeline().fit(X_tr, y_tr)

    p = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p)
    ap = average_precision_score(y_te, p)
    brier = brier_score_loss(y_te, p)
    print(f"Test ROC-AUC = {auc:.3f}   PR-AUC = {ap:.3f}   Brier = {brier:.4f}   base rate = {base:.2%}")
    # Screening operating point: flag every business scoring above the average.
    print(f"\nClassification report @ threshold = base rate ({base:.3f}) — 'flag above-average risk':")
    print(classification_report(y_te, (p >= base).astype(int), digits=3))

    importance = grouped_importance(pipe)
    plot_roc(y_te, p, auc)
    plot_pr(y_te, p, ap, base)
    plot_calibration(y_te, p)
    plot_importance(importance)
    plot_shap(pipe, X_te.sample(min(2000, len(X_te)), random_state=42))

    meta = {
        "features_cat": CAT,
        "features_num": NUM,
        "business_types": sorted(df["business_type"].unique().tolist()),
        "regions": sorted(df["region"].unique().tolist()),
        "settlement_types": sorted(df["settlement_type"].unique().tolist()),
        "base_rate": float(base),
        "metrics": {"roc_auc": float(auc), "pr_auc": float(ap), "brier": float(brier)},
        "feature_importance": importance,
    }
    joblib.dump({"pipeline": pipe, "meta": meta}, MODEL_OUT)
    print(f"\nSaved model -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
