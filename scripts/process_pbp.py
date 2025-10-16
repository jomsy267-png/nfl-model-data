#!/usr/bin/env python3
"""
Build team-week model features from nflverse play-by-play CSVs.

Inputs   : data/raw/nflverse__nflfastR-play-by-play-data/play_by_play_YYYY.csv.gz
Outputs  : data/processed/team_week_features.csv

Features :
- Offense: plays, EPA/play, success rate
- Defense: EPA/play allowed, success rate allowed
- Neutral pass rate & PROE (team neutral pass% - league neutral pass%)
- Pace proxy: offensive plays per game
- Rolling (last 4 weeks within season) for off/def EPA/play & success

Assumptions:
- Uses only a small set of columns; handles missing cols gracefully.
- Streams year-by-year to keep memory light.
"""

import os
import gzip
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw/nflverse__nflfastR-play-by-play-data")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "team_week_features.csv"

USECOLS = [
    # identifiers
    "season","week","game_id","play_id","posteam","defteam","home_team","away_team",
    # play/type
    "pass","rush","qb_dropback",
    # metrics
    "epa","success",
    # context for neutrality
    "down","ydstogo","yardline_100","score_differential","game_seconds_remaining"
]

def load_year(path: Path) -> pd.DataFrame:
    # pandas handles .gz with compression='infer'. Some years/cols differ; use usecols+errors='ignore'
    df = pd.read_csv(path, compression="infer", low_memory=False, usecols=lambda c: c in USECOLS)
    # enforce columns
    for c in USECOLS:
        if c not in df.columns:
            df[c] = np.nan
    return df

def neutral_filter(df: pd.DataFrame) -> pd.Series:
    # neutral definition: 1st/2nd down, -7 <= score <= +7, 20 <= yardline_100 <= 80, not end-game hurry (>= 120s)
    cond = (
        df["down"].isin([1,2]) &
        df["score_differential"].between(-7,7, inclusive="both") &
        df["yardline_100"].between(20,80, inclusive="both") &
        (df["game_seconds_remaining"] >= 120)
    )
    cond = cond.fillna(False)
    return cond

def build_team_week(df_all: pd.DataFrame) -> pd.DataFrame:
    # basic cleaning
    df = df_all.dropna(subset=["season","week","posteam","defteam","epa"])
    # coerce
    for col in ["pass","rush","qb_dropback","success"]:
        if col in df.columns:
            df[col] = (df[col] == 1) if df[col].dtype != bool else df[col].fillna(False)
        else:
            df[col] = False

    # offense aggregates (team = posteam)
    off = df.groupby(["season","week","posteam"], as_index=False).agg(
        off_plays=("play_id","count"),
        off_epa_per_play=("epa","mean"),
        off_success_rate=("success","mean"),
        off_pass_attempts=("pass","sum"),
        off_rush_attempts=("rush","sum"),
        off_dropbacks=("qb_dropback","sum"),
    ).rename(columns={"posteam":"team"})

    # defense aggregates (team = defteam)
    dfn = df.groupby(["season","week","defteam"], as_index=False).agg(
        def_plays=("play_id","count"),
        def_epa_per_play_allowed=("epa","mean"),
        def_success_allowed=("success","mean"),
    ).rename(columns={"defteam":"team"})

    # merge off+def same key (outer to keep bye weeks if any appear later)
    tw = pd.merge(off, dfn, on=["season","week","team"], how="outer")

    # Neutral pass rate & PROE
    neutr = df[neutral_filter(df)].copy()
    neutr["is_pass"] = neutr["pass"].astype(float)
    nr_team = neutr.groupby(["season","week","posteam"], as_index=False).agg(
        neutral_pass_plays=("is_pass","sum"),
        neutral_plays=("is_pass","count"),
    ).rename(columns={"posteam":"team"})
    tw = tw.merge(nr_team, on=["season","week","team"], how="left")
    tw["neutral_pass_rate"] = (tw["neutral_pass_plays"] / tw["neutral_plays"]).replace([np.inf,np.nan], np.nan)

    # League neutral pass rate per season-week
    league_nr = nr_team.groupby(["season","week"], as_index=False).apply(
        lambda g: pd.Series({"league_neutral_pass_rate": g["neutral_pass_plays"].sum()/max(g["neutral_plays"].sum(),1)})
    )
    tw = tw.merge(league_nr, on=["season","week"], how="left")
    tw["proe"] = tw["neutral_pass_rate"] - tw["league_neutral_pass_rate"]

    # Pace proxy: offensive plays per game (team-week)
    tw["pace_plays"] = tw["off_plays"]

    # Fill zeros for counts where appropriate
    for c in ["off_plays","def_plays","neutral_pass_plays","neutral_plays","off_pass_attempts","off_rush_attempts","off_dropbacks"]:
        if c in tw.columns:
            tw[c] = tw[c].fillna(0).astype(int)

    # Rolling windows (last 4 weeks within season per team)
    tw = tw.sort_values(["team","season","week"])
    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["off_epa_per_play_4w"] = g["off_epa_per_play"].rolling(4, min_periods=1).mean()
        g["off_success_rate_4w"] = g["off_success_rate"].rolling(4, min_periods=1).mean()
        g["def_epa_per_play_allowed_4w"] = g["def_epa_per_play_allowed"].rolling(4, min_periods=1).mean()
        g["def_success_allowed_4w"] = g["def_success_allowed"].rolling(4, min_periods=1).mean()
        g["pace_plays_4w"] = g["pace_plays"].rolling(4, min_periods=1).mean()
        g["proe_4w"] = g["proe"].rolling(4, min_periods=1).mean()
        return g
    tw = tw.groupby(["team","season"], group_keys=False).apply(_roll)

    # Final ordering & types
    cols = [
        "team","season","week",
        "off_plays","off_epa_per_play","off_success_rate",
        "def_plays","def_epa_per_play_allowed","def_success_allowed",
        "neutral_plays","neutral_pass_plays","neutral_pass_rate","league_neutral_pass_rate","proe",
        "pace_plays",
        "off_epa_per_play_4w","off_success_rate_4w",
        "def_epa_per_play_allowed_4w","def_success_allowed_4w",
        "pace_plays_4w","proe_4w",
        "off_pass_attempts","off_rush_attempts","off_dropbacks",
    ]
    cols = [c for c in cols if c in tw.columns]
    tw = tw[cols].sort_values(["season","week","team"]).reset_index(drop=True)
    return tw

def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"Raw folder not found: {RAW_DIR}")

    # load years one by one, concat minimal cols
    frames = []
    for p in sorted(RAW_DIR.glob("play_by_play_*.csv.gz")):
        print(f"Reading {p.name} ...")
        try:
            df = load_year(p)
            frames.append(df)
        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    if not frames:
        raise SystemExit("No play_by_play_*.csv.gz files found.")

    all_df = pd.concat(frames, ignore_index=True)
    features = build_team_week(all_df)
    features.to_csv(OUT_PATH, index=False)
    print(f"Saved features → {OUT_PATH}  (rows={len(features):,})")

if __name__ == "__main__":
    main()
