# scripts/train_xgb.py
# Train an XGBoost model on team_week_features.csv
# Deps: polars, xgboost, scikit-learn (for split/metrics)
import argparse, json, os
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import xgboost as xgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--out_metrics", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)

    df = pl.read_csv(args.features)

    # target
    if {"home_score","away_score"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_win")
        )
    else:
        with open(args.out_metrics, "w") as f:
            json.dump({"note": "No scores; cannot train XGB"}, f, indent=2)
        # write a dummy model so artifact upload works
        with open(args.out_model, "w") as f:
            json.dump({"model_type":"xgb","note":"no_train_data"}, f, indent=2)
        return

    # Feature candidates (same list as logistic where possible)
    cand = [
        "implied_margin","implied_total",
        "h_roll3_margin","a_roll3_margin",
        "h_roll3_winpct","a_roll3_winpct",
        "h_rest_days","a_rest_days",
        "spread_line","total_line",
    ]
    feats = [c for c in cand if c in df.columns]
    use_cols = feats + ["home_win"]

    df_train = df.select([c for c in use_cols if c in df.columns]).drop_nulls()
    if df_train.height < 200 or not feats:
        with open(args.out_metrics, "w") as f:
            json.dump({"note":"insufficient_rows_or_features","rows":int(df_train.height),"features":feats}, f, indent=2)
        with open(args.out_model, "w") as f:
            json.dump({"model_type":"xgb","note":"passthrough"}, f, indent=2)
        return

    X = df_train.select(feats).to_numpy()
    y = df_train.get_column("home_win").to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Lightweight, fast params
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_tr, y_tr)

    p_tr = model.predict_proba(X_tr)[:,1]
    p_te = model.predict_proba(X_te)[:,1]

    metrics = {}
    try:
        metrics["train_logloss"] = float(log_loss(y_tr, p_tr))
        metrics["test_logloss"] = float(log_loss(y_te, p_te))
    except: pass
    try:
        metrics["train_auc"] = float(roc_auc_score(y_tr, p_tr))
        metrics["test_auc"] = float(roc_auc_score(y_te, p_te))
    except: pass
    try:
        metrics["train_acc"] = float(accuracy_score(y_tr, (p_tr>=0.5).astype(int)))
        metrics["test_acc"] = float(accuracy_score(y_te, (p_te>=0.5).astype(int)))
    except: pass

    # Save model as raw booster JSON + feature list
    booster = model.get_booster().save_raw("json").decode("utf-8")
    out = {
        "model_type": "xgboost",
        "features": feats,
        "booster_json": booster,
        "params": {"n_estimators":400,"max_depth":3,"learning_rate":0.05,
                   "subsample":0.8,"colsample_bytree":0.8,"reg_lambda":1.0,
                   "tree_method":"hist","objective":"binary:logistic"}
    }
    with open(args.out_model, "w") as f:
        json.dump(out, f, indent=2)

    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Trained XGB with features:", feats)
    print("Metrics:", json.dumps(metrics))

if __name__ == "__main__":
    main()
