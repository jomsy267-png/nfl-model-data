# scripts/backtest.py
# Historical backtest using saved model (+ optional isotonic calibrator)
# Deps: polars
import os, json, math, argparse, bisect
import polars as pl

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def auc_roc(probs, labels):
    # Pure-Python AUC via rank method
    paired = sorted(zip(probs, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # average ranks for ties
    ranks = []
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j+1][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based ranks
        for _ in range(i, j+1):
            ranks.append(avg_rank)
        i = j + 1
    sum_ranks_pos = sum(r for (r, (_, y)) in zip(ranks, paired) if y == 1)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc

def log_loss(probs, labels, eps=1e-12):
    ll = 0.0
    n = 0
    for p, y in zip(probs, labels):
        p = min(max(p, eps), 1 - eps)
        ll += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        n += 1
    return ll / n if n else None

def brier_score(probs, labels):
    n = len(probs)
    if n == 0: return None
    return sum((p - y)**2 for p, y in zip(probs, labels)) / n

def maybe_apply_calibrator(p_raw_col: pl.Expr, calibrator):
    if calibrator is None:
        return p_raw_col.alias("p_cal")
    xs, ys = calibrator
    def cal(p):
        i = bisect.bisect_right(xs, float(p)) - 1
        if i < 0: return float(ys[0])
        if i >= len(xs) - 1: return float(ys[-1])
        x0, x1 = xs[i], xs[i+1]
        y0, y1 = ys[i], ys[i+1]
        if x1 == x0: return float(y0)
        t = (float(p) - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0))
    return p_raw_col.map_elements(cal).alias("p_cal")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", default="data/processed/team_week_features.csv")
    ap.add_argument("--model_json", default="models/nfl_model.json")
    ap.add_argument("--calibrator_json", default="models/calibrator_isotonic.json")
    ap.add_argument("--out_dir", default="models")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load features & keep labeled rows
    df = pl.read_csv(args.features_csv)
    if not {"home_score","away_score"}.issubset(df.columns):
        raise SystemExit("Features CSV missing final scores; cannot backtest.")
    df = df.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    if df.height == 0:
        raise SystemExit("No labeled rows to backtest.")

    # Target
    df = df.with_columns((pl.col("home_score") > pl.col("away_score")).cast(pl.Int8).alias("home_win"))

    # Load model
    with open(args.model_json) as f:
        model = json.load(f)
    feats = model.get("features", [])
    coef = model.get("coef", [])
    intercept = float(model.get("intercept", 0.0))
    if not feats or not coef or len(feats) != len(coef):
        raise SystemExit("Model has no usable features for backtest.")

    # Optional calibrator
    calibrator = None
    if os.path.exists(args.calibrator_json):
        try:
            with open(args.calibrator_json) as f:
                c = json.load(f)
                if c.get("type") == "isotonic" and "x_thresholds" in c and "y_values" in c:
                    calibrator = (c["x_thresholds"], c["y_values"])
        except Exception:
            calibrator = None

    # Ensure feature columns exist & are float
    for c in feats:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(c))
    df = df.with_columns([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in feats])

    # Raw prob
    logit = pl.lit(intercept)
    for c, w in zip(feats, coef):
        logit = logit + pl.col(c) * float(w)
    p_raw = (1 / (1 + (-logit).exp())).alias("p_raw")

    # Calibrated prob (or pass-through)
    p_cal = maybe_apply_calibrator(p_raw, calibrator)

    base_cols = [c for c in ["season","week","game_id","gameday","home_team","away_team","spread_line","total_line","implied_margin","implied_total"] if c in df.columns]
    out = df.select(base_cols + ["home_win", p_raw, p_cal])

    # Save by-game
    out_path = os.path.join(args.out_dir, "backtest_by_game.csv")
    out.write_csv(out_path)

    # Metrics overall (use calibrated where available)
    probs = out.get_column("p_cal").fill_null(out.get_column("p_raw")).to_list()
    labels = out.get_column("home_win").to_list()
    metrics = {
        "n_games": int(len(labels)),
        "brier": brier_score(probs, labels),
        "logloss": log_loss(probs, labels),
        "auc": auc_roc(probs, labels),
    }

    # By-season aggregates (Polars 0.20+ friendly)
    by_season = (
        out
        .with_columns(pl.col("p_cal").fill_null(pl.col("p_raw")).alias("p_use"))
        .group_by("season")
        .agg([
            pl.len().alias("n"),
            ((pl.col("p_use") - pl.col("home_win"))**2).mean().alias("brier"),
            pl.when(pl.col("home_win") == 1)
              .then(-(pl.col("p_use") + pl.lit(1e-12)).log())
              .otherwise(-((pl.lit(1.0) - pl.col("p_use") + pl.lit(1e-12)).log()))
              .mean()
              .alias("logloss"),
            ((pl.col("p_use") >= 0.5).cast(pl.Int8) == pl.col("home_win")).cast(pl.Float64).mean().alias("accuracy"),
        ])
        .sort("season")
    )
    by_season_path = os.path.join(args.out_dir, "backtest_by_season.csv")
    by_season.write_csv(by_season_path)

    # By-week aggregates (if season/week present)
    by_week_cols = [c for c in ["season","week"] if c in out.columns]
    if by_week_cols:
        by_week = (
            out
            .with_columns(pl.col("p_cal").fill_null(pl.col("p_raw")).alias("p_use"))
            .group_by(by_week_cols)
            .agg([
                pl.len().alias("n"),
                ((pl.col("p_use") - pl.col("home_win"))**2).mean().alias("brier"),
                pl.when(pl.col("home_win") == 1)
                  .then(-(pl.col("p_use") + pl.lit(1e-12)).log())
                  .otherwise(-((pl.lit(1.0) - pl.col("p_use") + pl.lit(1e-12)).log()))
                  .mean()
                  .alias("logloss"),
                ((pl.col("p_use") >= 0.5).cast(pl.Int8) == pl.col("home_win")).cast(pl.Float64).mean().alias("accuracy"),
            ])
            .sort(by_week_cols)
        )
        by_week_path = os.path.join(args.out_dir, "backtest_by_week.csv")
        by_week.write_csv(by_week_path)
    else:
        by_week_path = None

    # Write metrics JSON
    metrics_path = os.path.join(args.out_dir, "backtest_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote:\n- {out_path}\n- {by_season_path}\n- {by_week_path or '(no week groups)'}\n- {metrics_path}")

if __name__ == "__main__":
    main()
