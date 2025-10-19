# scripts/train_model.py
import argparse, json, os
import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--out_metrics", required=True)
    ap.add_argument("--out_latest_features", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)

    # Load features (Polars)
    df = pl.read_csv(args.features)

    # Create target if scores exist
    has_scores = set(["home_score","away_score"]).issubset(df.columns)
    if has_scores:
        df = df.with_columns((pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_win"))
    else:
        # No historical labels → nothing to train
        with open(args.out_metrics, "w") as f:
            json.dump({"note": "No scores present; cannot train"}, f, indent=2)
        # Still produce latest features (everything)
        df.write_csv(args.out_latest_features)
        print("No scores present; wrote latest features only.")
        return

    # Feature candidates (keep it tiny & robust)
    feature_candidates = [c for c in ["implied_margin","implied_total"] if c in df.columns]
    if not feature_candidates:
        # Fall back to spread/total names if present
        feature_candidates = [c for c in ["spread_line","total_line"] if c in df.columns]

    # Training rows: must have target and feature columns non-null
    use_cols = feature_candidates + ["home_win"]
    df_train = df.select([c for c in use_cols if c in df.columns]).drop_nulls()

    if df_train.height < 200 or len(feature_candidates) == 0:
        # Not enough data to fit meaningfully—emit metrics/pass-through model
        with open(args.out_metrics, "w") as f:
            json.dump({
                "note": "Insufficient data or features to train",
                "rows": int(df_train.height),
                "features": feature_candidates
            }, f, indent=2)
        # Latest features = games without scores (likely upcoming)
        latest = df.filter(pl.any_horizontal([pl.col("home_score").is_null(), pl.col("away_score").is_null()]))
        latest.select([c for c in df.columns if c != "home_win"]).write_csv(args.out_latest_features)
        with open(args.out_model, "w") as f:
            json.dump({"model": "passthrough", "features": feature_candidates}, f, indent=2)
        print("Wrote passthrough artifacts.")
        return

    # Prepare numpy arrays
    X = df_train.select(feature_candidates).to_numpy()
    y = df_train.get_column("home_win").to_numpy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Predictions
    prob_train = clf.predict_proba(X_train)[:, 1]
    prob_test  = clf.predict_proba(X_test)[:, 1]

    # Metrics (guarded)
    metrics = {}
    try:
        metrics["train_logloss"] = float(log_loss(y_train, prob_train))
        metrics["test_logloss"]  = float(log_loss(y_test, prob_test))
    except Exception:
        pass
    try:
        metrics["train_auc"] = float(roc_auc_score(y_train, prob_train))
        metrics["test_auc"]  = float(roc_auc_score(y_test, prob_test))
    except Exception:
        pass
    try:
        metrics["train_acc"] = float(accuracy_score(y_train, (prob_train >= 0.5).astype(int)))
        metrics["test_acc"]  = float(accuracy_score(y_test, (prob_test  >= 0.5).astype(int)))
    except Exception:
        pass

    # Save a JSON-serializable model (coef/intercept/features)
    model_json = {
        "model_type": "logistic_regression",
        "features": feature_candidates,
        "coef": clf.coef_.ravel().tolist(),
        "intercept": float(clf.intercept_.ravel()[0]),
        "classes_": [int(c) for c in clf.classes_.tolist()]
    }
    with open(args.out_model, "w") as f:
        json.dump(model_json, f, indent=2)

    # Save metrics
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # Latest features = rows that don't have scores yet (likely upcoming games)
    if has_scores:
        latest = df.filter(pl.any_horizontal([pl.col("home_score").is_null(), pl.col("away_score").is_null()]))
    else:
        latest = df
    # Keep only needed columns for inference
    keep_for_latest = [c for c in ["season","week","game_id","home_team","away_team","implied_margin","implied_total","spread_line","total_line"] if c in latest.columns]
    if keep_for_latest:
        latest = latest.select(keep_for_latest)
    latest.write_csv(args.out_latest_features)

    print("Training complete.")
    print("Metrics:", json.dumps(metrics))

if __name__ == "__main__":
    main()
