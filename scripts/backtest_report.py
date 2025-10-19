# scripts/backtest_report.py
import os
from datetime import datetime, timezone
import pandas as pd

# ---- Config (edit paths if your repo differs) ----
PREDICTIONS_CSV = "data/predictions.csv"    # requires game_id, pred_winner, p_use, pred_odds, team_pred, etc.
RESULTS_CSV     = "data/results.csv"        # requires game_id, actual_winner, home_win, etc.
OUT_DIR         = "reports"
OUT_STEM        = f"backtest_{datetime.now(timezone.utc).strftime('%Y%m%d')}"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    preds   = pd.read_csv(PREDICTIONS_CSV)
    actuals = pd.read_csv(RESULTS_CSV)

    # Ensure required columns exist (soft guardrails)
    needed_preds   = {"game_id", "pred_winner"}
    needed_actuals = {"game_id", "actual_winner"}
    missing = (needed_preds - set(preds.columns)) | (needed_actuals - set(actuals.columns))
    if missing:
        raise SystemExit(f"Missing required columns for report: {sorted(missing)}")

    df = preds.merge(actuals, on="game_id", suffixes=("_pred", "_actual"))

    # Basic metrics
    df["hit"]  = (df["pred_winner"] == df["actual_winner"]).astype(int)
    # If p_use & home_win exist, compute a Brier-like score; else skip gracefully
    if {"p_use", "home_win"}.issubset(df.columns):
        df["brier_sq"] = (df["p_use"] - df["home_win"])**2
        brier = df["brier_sq"].mean()
    else:
        brier = None

    # ROI proxy if you store fair/decimal odds in pred_odds (1 = stake)
    if "pred_odds" in df.columns:
        df["roi"] = df["hit"] * (df["pred_odds"] - 1) - (1 - df["hit"])
        avg_roi = df["roi"].mean()
    else:
        avg_roi = None

    accuracy = df["hit"].mean()
    n_games  = len(df)

    # Per-team performance if “team_pred” exists
    per_team_md = ""
    if "team_pred" in df.columns:
        per_team = (df
                    .groupby("team_pred", dropna=False)["hit"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10))
        per_team_md = per_team.to_frame("accuracy").to_markdown()

    # Build Markdown
    lines = []
    lines.append("# 🏈 NFL Backtest Report")
    lines.append(f"**Generated (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Games Tested:** {n_games}")
    lines.append(f"**Accuracy:** {accuracy:.2%}")
    if brier is not None:
        lines.append(f"**Brier Score:** {brier:.4f}")
    if avg_roi is not None:
        lines.append(f"**Average ROI (per bet):** {avg_roi:.2%}")
    lines.append("")
    if per_team_md:
        lines.append("## Top Performing Teams (by prediction accuracy)")
        lines.append(per_team_md)
        lines.append("")

    md_path  = os.path.join(OUT_DIR, f"{OUT_STEM}.md")
    csv_path = os.path.join(OUT_DIR, f"{OUT_STEM}.csv")

    # Save outputs
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    df.to_csv(csv_path, index=False)

    print(f"✅ Wrote {md_path}")
    print(f"✅ Wrote {csv_path}")

if __name__ == "__main__":
    main()
