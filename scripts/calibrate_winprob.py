# scripts/calibrate_winprob.py
# Fit an isotonic calibration curve on historical rows, save as JSON knots.
# Deps: polars, scikit-learn

import argparse, json, os
import polars as pl
from sklearn.isotonic import IsotonicRegression

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True, help="data/processed/team_week_features.csv")
    ap.add_argument("--model_json", required=True, help="models/nfl_model.json")
    ap.add_argument("--out_json", required=True, help="models/calibrator_isotonic.json")
    args = ap.parse_args()

    if not os.path.exists(args.features_csv):
        raise SystemExit(f"Missing features CSV: {args.features_csv}")
    if not os.path.exists(args.model_json):
        raise SystemExit(f"Missing model JSON: {args.model_json}")

    df = pl.read_csv(args.features_csv)
    # Keep labeled rows only
    has_scores = set(["home_score","away_score"]).issubset(df.columns)
    if not has_scores:
        # Nothing to calibrate
        with open(args.out_json, "w") as f:
            json.dump({"note": "no_scores_to_calibrate"}, f, indent=2)
        print("No scores in features; wrote a pass-through calibrator note.")
        return

    df = df.with_columns((pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_win"))

    # Load model
    with open(args.model_json) as f:
        model = json.load(f)
    feats = model.get("features", [])
    coef = model.get("coef", [])
    intercept = float(model.get("intercept", 0.0))

    if not feats or not coef or len(feats) != len(coef):
        with open(args.out_json, "w") as f:
            json.dump({"note": "model_has_no_usable_features"}, f, indent=2)
        print("Model has no usable features; wrote note.")
        return

    # Ensure feature columns exist and are float, fill nulls with 0.0
    for c in feats:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(c))
    df = df.with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in feats])

    # Raw prob via logistic: 1 / (1 + exp(-z))
    logit = pl.lit(intercept)
    for c, w in zip(feats, coef):
        logit = logit + pl.col(c) * float(w)
    prob = (1 / (1 + (-logit).exp())).alias("p_raw")

    scored = df.select(["home_win"] + feats).with_columns(prob)
    scored_np = scored.select(["p_raw","home_win"]).to_numpy()
    if scored_np.shape[0] < 200:
        # Too few points to fit isotonic reliably
        with open(args.out_json, "w") as f:
            json.dump({"note": "too_few_points", "n": int(scored_np.shape[0])}, f, indent=2)
        print(f"Too few points ({scored_np.shape[0]}) for isotonic; wrote note.")
        return

    p_raw = scored_np[:,0]
    y = scored_np[:,1]

    # Fit isotonic mapping
    iso = IsotonicRegression(out_of_bounds="clip")
    p_cal = iso.fit_transform(p_raw, y)

    # Save knots (x_thresholds and y_values)
    calibrator = {
        "type": "isotonic",
        "x_thresholds": iso.X_thresholds_.tolist(),
        "y_values": iso.y_thresholds_.tolist(),
        "n_points": int(len(iso.X_thresholds_)),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(calibrator, f, indent=2)
    print(f"Wrote {args.out_json} with {calibrator['n_points']} knots.")

if __name__ == "__main__":
    main()
