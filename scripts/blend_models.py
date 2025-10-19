# scripts/blend_models.py
# Learn a convex blend between logistic and xgb model probabilities to minimize logloss.
# Deps: polars, xgboost (to score), numpy (small), scikit-learn (split, logloss)
import argparse, json, os, math
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import xgboost as xgb

def sigmoid(z): return 1.0 / (1.0 + math.exp(-z))

def score_logistic(df, model):
    feats = model["features"]
    coef = model["coef"]
    b0 = float(model["intercept"])
    X = df.select(feats).to_numpy()
    z = b0 + np.dot(X, np.array(coef, dtype=float))
    return 1.0 / (1.0 + np.exp(-z))

def score_xgb(df, model):
    feats = model["features"]
    booster_json = model["booster_json"]
    booster = xgb.Booster()
    booster.load_model(bytearray(booster_json, "utf-8"))
    dmat = xgb.DMatrix(df.select(feats).to_numpy())
    return booster.predict(dmat)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--logit_model", required=True)
    ap.add_argument("--xgb_model", required=True)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--out_metrics", required=True)
    args = ap.parse_args()

    for p in [args.features, args.logit_model, args.xgb_model]:
        if not os.path.exists(p):
            raise SystemExit(f"Missing: {p}")

    df = pl.read_csv(args.features)
    if not {"home_score","away_score"}.issubset(df.columns):
        raise SystemExit("No labeled rows; cannot blend.")
    df = df.with_columns(
        (pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()).alias("_has")
    ).filter(pl.col("_has")).drop("_has").with_columns(
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_win")
    )

    with open(args.logit_model) as f: logit = json.load(f)
    with open(args.xgb_model) as f: xgbm = json.load(f)

    # intersection of features; fill missing with 0
    feat_union = list(dict.fromkeys((logit.get("features") or []) + (xgbm.get("features") or [])))
    if not feat_union: raise SystemExit("No common features to blend.")

    for c in feat_union:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(c))
    df = df.with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in feat_union])

    # Ensure each scorer has its own features available
    X_df_log = df.select([c for c in (logit.get("features") or [])])
    X_df_xgb = df.select([c for c in (xgbm.get("features") or [])])

    if X_df_log.width == 0 or X_df_xgb.width == 0:
        raise SystemExit("One model has zero usable features; abort blending.")

    y = df.get_column("home_win").to_numpy()
    # Score models
    p_log = score_logistic(df, logit)
    p_xgb = score_xgb(df, xgbm)

    # Train/val split to pick alpha
    pL_tr, pL_va, pX_tr, pX_va, y_tr, y_va = train_test_split(
        p_log, p_xgb, y, test_size=0.25, random_state=42, stratify=y
    )

    best_alpha, best_ll = 0.5, 1e9
    for a in [i/100 for i in range(0,101)]:  # coarse grid 0..1
        pv = a*pL_va + (1-a)*pX_va
        ll = log_loss(y_va, pv, labels=[0,1])
        if ll < best_ll:
            best_ll, best_alpha = ll, a

    # Save blended model
    blended = {
        "model_type": "blend_linear",
        "alpha_logistic": best_alpha,
        "components": {
            "logistic": {"path": os.path.basename(args.logit_model), "features": logit.get("features", [])},
            "xgboost":  {"path": os.path.basename(args.xgb_model),   "features": xgbm.get("features", [])},
        }
    }
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    with open(args.out_model, "w") as f: json.dump(blended, f, indent=2)

    with open(args.out_metrics, "w") as f:
        json.dump({"val_logloss": float(best_ll), "alpha_logistic": float(best_alpha)}, f, indent=2)

    print(f"Blended alpha (logistic weight) = {best_alpha:.2f}, val_logloss={best_ll:.4f}")

if __name__ == "__main__":
    main()
