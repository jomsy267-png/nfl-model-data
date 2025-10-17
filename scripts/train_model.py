#!/usr/bin/env python3
import os, gzip, glob
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

RAW_PBP   = Path("data/raw/nflverse__nflfastR-play-by-play-data")
PROC_FEAT = Path("data/processed/team_week_features.csv")
EXT_DIR   = Path("data/external")
OUT_DIR   = Path("models"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_gz_csv(path):
    return pd.read_csv(path, compression="infer", low_memory=False)

def load_results_and_lines():
    # concat all seasons found
    games, lines = [], []
    for p in sorted(EXT_DIR.glob("games_*.csv.gz")):
        games.append(read_gz_csv(p))
    for p in sorted(EXT_DIR.glob("lines_*.csv.gz")):
        lines.append(read_gz_csv(p))
    if not games or not lines:
        raise SystemExit("No games/lines data found. Run fetch_results.yml first.")
    g = pd.concat(games, ignore_index=True)
    l = pd.concat(lines, ignore_index=True)
    # keep only needed fields
    g = g[["game_id","season","week","home_team","away_team","home_score","away_score"]]
    # standardize team columns to match features
    # nflverse uses same abbreviations as nfldata; we assume features 'team' uses the same.
    # build team-week rows (home & away as separate observations)
    g["margin_home"] = g["home_score"] - g["away_score"]
    home_rows = g.rename(columns={"home_team":"team"})[["game_id","season","week","team","margin_home"]].copy()
    home_rows["is_home"] = 1
    home_rows = home_rows.rename(columns={"margin_home":"margin"})
    away_rows = g.rename(columns={"away_team":"team"})[["game_id","season","week","team"]].copy()
    away_rows["margin"] = -g["margin_home"]
    away_rows["is_home"] = 0
    long = pd.concat([home_rows, away_rows], ignore_index=True)

    # closing lines: spread toward home, moneylines for home/away present
    keep = ["game_id","spread_line","home_moneyline","away_moneyline","total_line"]
    l = l[[c for c in keep if c in l.columns]].drop_duplicates("game_id")
    df = long.merge(l, on="game_id", how="left")
    return df

def add_features(df_feat, df_outcomes):
    # df_feat: team-week features (your CSV)
    f = df_feat.copy()
    cols = [c for c in [
        "season","week","team",
        "off_epa_per_play_4w","def_epa_per_play_allowed_4w",
        "off_success_rate_4w","def_success_allowed_4w",
        "pace_plays_4w","proe_4w"
    ] if c in f.columns]
    f = f[cols].dropna(subset=["season","week","team"])
    # keep latest snapshot per team-week
    f = f.sort_values(["season","week"]).drop_duplicates(["season","week","team"], keep="last")
    df = df_outcomes.merge(f, on=["season","week","team"], how="left")
    # market features (toward this team)
    # Convert closing spread (home perspective) to this team perspective:
    # If is_home==1, closing_spread_for_team = -spread_line (book quotes "home by -x" as negative)
    # If away, closing_spread_for_team = +spread_line
    if "spread_line" in df.columns:
        df["spread_for_team"] = np.where(df["is_home"]==1, -df["spread_line"], df["spread_line"])
    else:
        df["spread_for_team"] = np.nan
    # Moneyline fair prob from closing ML (approx, ignoring vig)
    def ml_to_prob(ml):
        ml = pd.to_numeric(ml, errors="coerce")
        # +200 => 1/(1+200/100); -200 => 1/(1+100/200)
        p = np.where(ml>0, 100/(ml+100), np.where(ml<0, (-ml)/(100+(-ml)), np.nan))
        return p
    if {"home_moneyline","away_moneyline"} <= set(df.columns):
        df["p_home_close"] = ml_to_prob(df["home_moneyline"])
        df["p_away_close"] = ml_to_prob(df["away_moneyline"])
        df["p_team_close"] = np.where(df["is_home"]==1, df["p_home_close"], df["p_away_close"])
    else:
        df["p_team_close"] = np.nan

    # labels
    df["won"]   = (df["margin"] > 0).astype(int)
    df["cover"] = (df["margin"] - (-df["spread_for_team"]) > 0).astype(int)  # margin vs team spread

    return df

def train_models(df):
    # Feature matrix
    feats = [c for c in [
        "off_epa_per_play_4w","def_epa_per_play_allowed_4w",
        "off_success_rate_4w","def_success_allowed_4w",
        "pace_plays_4w","proe_4w",
        "spread_for_team","p_team_close","is_home"
    ] if c in df.columns]
    X = df[feats].copy()
    X = X.fillna(X.median(numeric_only=True))
    y_margin = df["margin"].values
    y_win    = df["won"].values

    # 1) Margin model (Ridge)
    ridge = Ridge(alpha=2.0, fit_intercept=True)
    ridge.fit(X, y_margin)

    # 2) Win prob model (Logistic) using same features
    logit = LogisticRegression(max_iter=1000, solver="lbfgs")
    logit.fit(X, y_win)
    p_raw = logit.predict_proba(X)[:,1]

    # 3) Calibrate win prob with isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y_win)
    p_cal = iso.transform(p_raw)

    # 4) Blended “market + model” prob (market prior with residual edge)
    # Blend weight chosen by minimizing Brier on train (simple baseline):
    p_market = df["p_team_close"].fillna(p_raw)
    best_brier, best_w = 1e9, 0.0
    for w in np.linspace(0.0, 1.0, 21):
        p_blend = w*p_cal + (1-w)*p_market
        b = brier_score_loss(y_win, p_blend)
        if b < best_brier:
            best_brier, best_w = b, w
    p_blend = best_w*p_cal + (1-best_w)*p_market

    metrics = {
        "mae_margin": float(mean_absolute_error(y_margin, ridge.predict(X))),
        "brier_raw":  float(brier_score_loss(y_win, p_raw)),
        "brier_iso":  float(brier_score_loss(y_win, p_cal)),
        "brier_blend":float(brier_score_loss(y_win, p_blend)),
        "blend_weight_model": float(best_w)
    }

    # Save “model” as simple JSON (coeffs) for transparency and portability
    model = {
        "features": feats,
        "ridge_coef": ridge.coef_.tolist(),
        "ridge_intercept": float(ridge.intercept_),
        "logit_coef": logit.coef_[0].tolist(),
        "logit_intercept": float(logit.intercept_[0]),
        "iso_x_": iso.X_thresholds_.tolist(),
        "iso_y_": iso.y_thresholds_.tolist(),
        "blend_w_model": float(best_w)
    }
    return model, metrics

def main():
    # Load inputs
    feat = pd.read_csv(PROC_FEAT)
    outcomes = load_results_and_lines()
    df = add_features(feat, outcomes)

    model, metrics = train_models(df)

    # Persist
    pd.Series(model).to_json(OUT_DIR/"nfl_model.json")
    pd.Series(metrics).to_json(OUT_DIR/"training_metrics.json")

    # Also output today-ready “current” table (latest snapshot per team)
    latest = feat.sort_values(["season","week"]).drop_duplicates(["season","team"], keep="last")
    latest.to_csv(OUT_DIR/"latest_team_features.csv", index=False)
    print("Saved models/nfl_model.json, models/training_metrics.json, models/latest_team_features.csv")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
