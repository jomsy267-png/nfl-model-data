# scripts/build_features.py
# Builds expanded game-level features (home-team perspective) from schedules_* CSVs.
# Dependencies: polars

import glob, os, sys
import polars as pl

def build_team_history(sched: pl.DataFrame) -> pl.DataFrame:
    """
    From game-level schedules (home/away rows in one table), build a per-team
    chronological table with team-centric stats, then compute lagged rolling features.
    Returns team-centric rows (one row per team-game).
    """
    # Home perspective
    home = sched.select([
        pl.col("season"),
        pl.col("week"),
        pl.col("game_id"),
        pl.col("gameday").alias("date"),
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opp"),
        pl.col("home_score").alias("pts_for"),
        pl.col("away_score").alias("pts_against"),
        pl.lit(1).alias("is_home")
    ])

    # Away perspective
    away = sched.select([
        pl.col("season"),
        pl.col("week"),
        pl.col("game_id"),
        pl.col("gameday").alias("date"),
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opp"),
        pl.col("away_score").alias("pts_for"),
        pl.col("home_score").alias("pts_against"),
        pl.lit(0).alias("is_home")
    ])

    team_games = pl.concat([home, away], how="vertical_relaxed")

    # Ensure proper date type
    if team_games["date"].dtype not in (pl.Date, pl.Datetime):
        team_games = team_games.with_columns(
            pl.col("date").str.strptime(pl.Date, strict=False)
        )

    # Sort chronologically within team
    team_games = team_games.sort(["team", "date", "week"])

    # Add raw outcomes
    team_games = team_games.with_columns([
        (pl.col("pts_for") - pl.col("pts_against")).alias("margin"),
        (pl.col("pts_for") > pl.col("pts_against")).cast(pl.Int8).alias("win")
    ])

    # ✅ Version-proof rest days:
    # Convert Date to epoch-days (Int32), diff within team, yields integer day gaps.
    team_games = team_games.with_columns(
        pl.col("date").cast(pl.Int32).diff().over("team").alias("rest_days")
    )

    # Rolling features (exclude current game: shift(1))
    def add_rolls(g: pl.DataFrame) -> pl.DataFrame:
        g = g.with_columns([
            pl.col("pts_for").shift(1).alias("_pf_lag"),
            pl.col("pts_against").shift(1).alias("_pa_lag"),
            pl.col("margin").shift(1).alias("_m_lag"),
            pl.col("win").shift(1).alias("_w_lag"),
        ])
        g = g.with_columns([
            pl.col("_pf_lag").rolling_mean(window_size=3).alias("roll3_pts_for"),
            pl.col("_pa_lag").rolling_mean(window_size=3).alias("roll3_pts_against"),
            pl.col("_m_lag").rolling_mean(window_size=3).alias("roll3_margin"),
            pl.col("_w_lag").rolling_mean(window_size=3).alias("roll3_winpct"),

            pl.col("_pf_lag").rolling_mean(window_size=5).alias("roll5_pts_for"),
            pl.col("_pa_lag").rolling_mean(window_size=5).alias("roll5_pts_against"),
            pl.col("_m_lag").rolling_mean(window_size=5).alias("roll5_margin"),
            pl.col("_w_lag").rolling_mean(window_size=5).alias("roll5_winpct"),
        ])
        return g.drop(["_pf_lag","_pa_lag","_m_lag","_w_lag"])

    team_games = (
        team_games
        .groupby("team", maintain_order=True)
        .map_groups(add_rolls)
    )

    return team_games

def main():
    os.makedirs("data/processed", exist_ok=True)

    # Load all schedules_*.csv
    sched_paths = sorted(glob.glob("data/external/schedules_*.csv"))
    if not sched_paths:
        print("ERROR: No schedules_*.csv found", file=sys.stderr)
        sys.exit(2)

    df = pl.concat([pl.read_csv(p) for p in sched_paths], how="vertical_relaxed")

    # Ensure expected base columns exist
    keep_base = [c for c in [
        "season","week","game_id","gameday",
        "home_team","away_team","home_score","away_score",
        "spread_line","total_line"
    ] if c in df.columns]
    df = df.select(keep_base)

    # Build team-centric history with rolling features
    team_hist = build_team_history(df)

    # Bring back game-level columns (market priors)
    game_cols = df.select([
        "game_id","season","week","gameday","home_team","away_team","home_score","away_score",
        "spread_line","total_line"
    ])

    # Home rolls (prefix h_)
    home_rolls = (
        team_hist
        .rename({
            "team":"home_team","opp":"away_team",
            "rest_days":"h_rest_days",
            "roll3_pts_for":"h_roll3_pts_for",
            "roll3_pts_against":"h_roll3_pts_against",
            "roll3_margin":"h_roll3_margin",
            "roll3_winpct":"h_roll3_winpct",
            "roll5_pts_for":"h_roll5_pts_for",
            "roll5_pts_against":"h_roll5_pts_against",
            "roll5_margin":"h_roll5_margin",
            "roll5_winpct":"h_roll5_winpct"
        })
        .select(["season","week","game_id","h_rest_days",
                 "h_roll3_pts_for","h_roll3_pts_against","h_roll3_margin","h_roll3_winpct",
                 "h_roll5_pts_for","h_roll5_pts_against","h_roll5_margin","h_roll5_winpct"])
    )

    # Away rolls (prefix a_)
    away_rolls = (
        team_hist
        .rename({
            "team":"away_team","opp":"home_team",
            "rest_days":"a_rest_days",
            "roll3_pts_for":"a_roll3_pts_for",
            "roll3_pts_against":"a_roll3_pts_against",
            "roll3_margin":"a_roll3_margin",
            "roll3_winpct":"a_roll3_winpct",
            "roll5_pts_for":"a_roll5_pts_for",
            "roll5_pts_against":"a_roll5_pts_against",
            "roll5_margin":"a_roll5_margin",
            "roll5_winpct":"a_roll5_winpct"
        })
        .select(["season","week","game_id","a_rest_days",
                 "a_roll3_pts_for","a_roll3_pts_against","a_roll3_margin","a_roll3_winpct",
                 "a_roll5_pts_for","a_roll5_pts_against","a_roll5_margin","a_roll5_winpct"])
    )

    # Merge home/away rolls into game-level features
    feats = (
        game_cols
        .join(home_rolls, on=["season","week","game_id"], how="left")
        .join(away_rolls, on=["season","week","game_id"], how="left")
    )

    # Final engineered columns compatible with your trainer
    feats = feats.with_columns([
        (pl.col("home_score") - pl.col("away_score")).alias("final_margin"),
        pl.col("spread_line").alias("implied_margin"),
        pl.col("total_line").alias("implied_total"),
    ])

    out = "data/processed/team_week_features.csv"
    feats.write_csv(out)
    print(f"Wrote {out} ({feats.height} rows, {len(feats.columns)} cols)")

if __name__ == "__main__":
    main()
