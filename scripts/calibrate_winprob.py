# scripts/calibrate_winprob.py
# Fit an isotonic calibration curve on historical rows, save as JSON knots.
# Deps: polars, scikit-learn

import argparse
import json
import os
import polars as pl
from sklearn.isotonic import IsotonicRegression

MIN_POINTS = 200  # minimum labeled rows to fit reliably

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True, help="data/processed/team_week_features.csv")
    ap.add_argument("--model_json", required=True, help="models/nfl_model.json")
    ap.add_argument("--out_json", required=True, help="models/calibrator_isotonic.json")
    args = ap.parse_args()

    if not os.path.exists(args.features_csv):
        with open(args.out_json, "w") as f:
            json.dump({"note": "missing_features_csv", "path": args.features_csv}, f, indent=2)
        return

    if not os.path.exists(args.model_json):
        with open(args.out_json, "w") as f:
            json.dump({"note": "missing_model_json", "path": args.model_json}, f, indent=2)
        return

    df = pl.read_csv(args.features_csv)

    # Need historical games with final scores
    if not {"home_score","away_score"}.issubset(df.columns):
        with open(args.out_json, "w") as f:
            json.dump({"note": "no_score_columns"}, f, indent=2)
        return

    df = df.with_columns(
        (pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()).alias("_has_scores")
    ).filter(pl.col("_has_scores")).drop("_has_scores")

    if df.height < MIN_POINTS:
        with open(args.out_json, "w") as f:
            json.dump({"note": "too_few_scored_games", "n": int(df.height)}, f, indent=2)
        return

    # Target
    df = df.with_columns(
        (pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_win")
    )

    # Load model and prepare features
    with open(args.model_json) as f:
        model = json.load(f)

    feats = model.get("features", [])
    coef = model.get("coef", [])
    intercept = float(model.get("intercept", 0.0))

    if not feats or not coef or len(feats) != len(coef):
        with open(args.out_json, "w") as f:
            json.dump({"note": "model_has_no_usable_features"}, f, indent=2)
        return

    # Ensure feature columns exist and are float; fill nulls with 0.0
    for c in feats:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(c))
    df = df.with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in feats])

    # Build raw probability via logistic: p_raw = 1/(1 + exp(-(b0 + sum(w*x))))
    logit = pl.lit(intercept)
    for c, w in zip(feats, coef):
        logit = logit + pl.col(c) * float(w)
    df = df.with_columns(((1 / (1 + (-logit).exp()))).alias("p_raw"))

    # Drop any residual nulls in target / p_raw (defensive)
    df = df.filter(pl.col("home_win").is_not_null() & pl.col("p_raw").is_not_null())

    n = df.height
    if n < MIN_POINTS:
        with open(args.out_json, "w") as f:
            json.dump({"note": "too_few_points_after_clean", "n": int(n)}, f, indent=2)
        return

    # Prepare Python lists (avoids hard dependency on NumPy here)
    p_raw = df.get_column("p_raw").to_list()
    y = df.get_column("home_win").to_list()

    # Fit isotonic mapping
    iso = IsotonicRegression(out_of_bounds="clip")
    p_cal = iso.fit_transform(p_raw, y)  # fit + transform (transform output unused beyond fit)

    # Save knots (x_thresholds and y_values)
    calibrator = {
        "type": "isotonic",
        "x_thresholds": iso.X_thresholds_.tolist(),
        "y_values": iso.y_thresholds_.tolist(),
        "n_points": int(len(iso.X_thresholds_)),
        "train_samples": int(n),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(calibrator, f, indent=2)

    print(f"Wrote {args.out_json} with {calibrator['n_points']} knots (N={n}).")

if __name__ == "__main__":
    main()
