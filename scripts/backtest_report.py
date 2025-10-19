# scripts/backtest_report.py
# Build a readable Markdown report from backtest outputs.
# Inputs (from scripts/backtest.py):
#   - models/backtest_by_game.csv
#   - models/backtest_by_season.csv
#   - models/backtest_metrics.json
# Output:
#   - models/backtest_report.md
#
# Deps: polars

import json
import os
import polars as pl


def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main():
    by_game_path = "models/backtest_by_game.csv"
    by_season_path = "models/backtest_by_season.csv"
    metrics_path = "models/backtest_metrics.json"
    out_md = "models/backtest_report.md"

    # Ensure inputs exist
    missing = [p for p in (by_game_path, by_season_path, metrics_path) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Missing backtest files: {missing}")

    # Load
    g = pl.read_csv(by_game_path)
    s = pl.read_csv(by_season_path)
    with open(metrics_path) as f:
        M = json.load(f)

    # Choose probability column (prefer calibrated)
    use_col = "p_cal" if "p_cal" in g.columns else "p_raw"
    g = g.with_columns(pl.col(use_col).alias("p_use"))

    # Reliability (deciles)
    # bin = floor(p_use*10) clipped to [0,9]
    g = g.with_columns((pl.col("p_use") * 10.0).floor().cast(pl.Int64).clip(0, 9).alias("bin"))
    rel = (
        g.group_by("bin")
         .agg([
             pl.len().alias("n"),
             pl.col("p_use").mean().alias("avg_prob"),
             pl.col("home_win").mean().alias("emp_rate"),
             ((pl.col("p_use") - pl.col("home_win")) ** 2).mean().alias("brier_bin"),
         ])
         .sort("bin")
         .with_columns((pl.col("emp_rate") - pl.col("avg_prob")).alias("calib_gap"))
    )

    # Assemble Markdown
    lines = []
    lines.append("# NFL Backtest Report")
    lines.append("")
    lines.append("## Overall metrics")
    lines.append("")
    lines.append(f"- **Games**: {M.get('n_games','')}")
    if "auc" in M:     lines.append(f"- **AUC**: {fmt(M['auc'])}")
    if "brier" in M:   lines.append(f"- **Brier score**: {fmt(M['brier'])}")
    if "logloss" in M: lines.append(f"- **Log loss**: {fmt(M['logloss'])}")
    lines.append("")

    # By season table
    lines.append("## By season")
    lines.append("")
    if s.height > 0:
        # ensure expected columns exist
        expected = ["season", "n", "brier", "logloss", "accuracy"]
        for col in expected:
            if col not in s.columns:
                s = s.with_columns(pl.lit(None).alias(col)) if col != "season" else s
        lines.append("| season | n | brier | logloss | accuracy |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in s.iter_rows(named=True):
            lines.append(
                f"| {r.get('season','')} | {r.get('n','')} | "
                f"{fmt(r.get('brier',''))} | {fmt(r.get('logloss',''))} | {fmt(r.get('accuracy',''))} |"
            )
    else:
        lines.append("_No season rows_")
    lines.append("")

    # Reliability table
    lines.append("## Reliability (calibration) by decile of predicted probability")
    lines.append("")
    if rel.height > 0:
        lines.append("| bin | n | avg_prob | empirical_rate | calib_gap | brier_bin |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for r in rel.iter_rows(named=True):
            lines.append(
                f"| {r['bin']} | {r['n']} | {fmt(r['avg_prob'])} | "
                f"{fmt(r['emp_rate'])} | {fmt(r['calib_gap'])} | {fmt(r['brier_bin'])} |"
            )
        lines.append("")
        lines.append("> `calib_gap = empirical_rate − avg_prob` (ideal ≈ 0)")
    else:
        lines.append("_No reliability data_")

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
