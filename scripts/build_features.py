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
        pl.col("date").cast(pl.Int32).diff().over("tea
