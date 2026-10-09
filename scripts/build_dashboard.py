"""Build a self-contained dashboard from the latest backtest CSV outputs."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd

COLORS = {
    "Seasonal naive": "#8a96a8",
    "SARIMAX": "#51a9ff",
    "Direct RF (lags)": "#c084fc",
    "SARIMAX + RF residual": "#43d9a3",
}


def _chart(summary: pd.DataFrame) -> str:
    selected = summary[summary["model"].isin(["SARIMAX", "SARIMAX + RF residual"])].copy()
    width, height = 720, 280
    left, top, chart_w, chart_h = 58, 18, 620, 205
    values = selected["mae"].astype(float)
    max_val = max(float(values.max()), 1.0) * 1.12
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="MAE by forecast horizon">'
    ]
    for tick in range(5):
        y_val = max_val * tick / 4
        y = top + chart_h - chart_h * tick / 4
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" class="axis">{y_val:.0f}</text>'
        )
    for model, group in selected.groupby("model", sort=False):
        group = group.sort_values("horizon")
        points = []
        for row in group.itertuples():
            x = left + chart_w * (int(row.horizon) - 1) / 5
            y = top + chart_h - chart_h * float(row.mae) / max_val
            points.append(f"{x:.1f},{y:.1f}")
        color = COLORS.get(model, "#fff")
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for point in points:
            x, y = point.split(",")
            parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
    for horizon in range(1, 7):
        x = left + chart_w * (horizon - 1) / 5
        parts.append(
            f'<text x="{x:.1f}" y="{top + chart_h + 24}" text-anchor="middle" class="axis">{horizon}</text>'
        )
    parts.append(
        f'<text x="{left + chart_w / 2}" y="{height - 8}" text-anchor="middle" class="axis">Forecast horizon (seasons)</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def build(summary_path: Path, output_path: Path) -> None:
    summary = pd.read_csv(summary_path)
    expected = {"model", "horizon", "n_forecasts", "mae", "rmse", "wape_pct", "mase"}
    missing = expected - set(summary.columns)
    if missing:
        raise ValueError(f"Summary is missing fields: {sorted(missing)}")

    table_rows = []
    for row in summary.sort_values(["horizon", "model"]).itertuples():
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row.model))}</td><td>{int(row.horizon)}</td>"
            f"<td>{int(row.n_forecasts)}</td><td>{float(row.mae):,.1f}</td>"
            f"<td>{float(row.rmse):,.1f}</td><td>{float(row.wape_pct):.2f}%</td>"
            f"<td>{float(row.mase):.3f}</td></tr>"
        )

    origin_count = int(summary["n_forecasts"].max())
    rows_html = "\n".join(table_rows)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Leakage-aware seasonal backtest for Sri Lankan rice production forecasts.">
<title>Sri Lanka Rice Production Forecasting — Model Evaluation</title>
<style>
:root{{--bg:#07101c;--panel:#0d1b2a;--line:#20364b;--text:#e7f0f7;--muted:#9db0c2;--green:#43d9a3;--blue:#51a9ff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at top right,#10263a,var(--bg) 55%);color:var(--text);font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}}
main{{max-width:1180px;margin:auto;padding:40px 22px 64px}}.eyebrow{{color:var(--green);font-size:12px;letter-spacing:.16em;text-transform:uppercase;font-weight:700}}
h1{{font-size:clamp(30px,4vw,48px);line-height:1.1;margin:12px 0}}h2{{font-size:20px;margin:0 0 14px}}p{{color:var(--muted);max-width:900px}}.grid{{stroke:var(--line);stroke-width:1}}.axis{{fill:var(--muted);font-size:11px}}
.hero,.panel,.notice{{background:rgba(13,27,42,.9);border:1px solid var(--line);border-radius:16px;padding:22px;margin-top:18px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:22px}}.kpi{{padding:15px;background:#0a1725;border:1px solid var(--line);border-radius:12px}}.kpi b{{display:block;font-size:23px;color:var(--green)}}.kpi span{{color:var(--muted);font-size:12px}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin:10px 0}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
.chart-wrap{{overflow-x:auto}}svg{{width:100%;min-width:520px}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:right;padding:10px 9px;border-bottom:1px solid var(--line);white-space:nowrap}}th{{color:#b7cadb;font-weight:650}}td:first-child,th:first-child{{text-align:left}}.notice{{border-left:4px solid #f2b65d}}.notice strong{{color:#ffd28c}}.foot{{font-size:12px;color:var(--muted);margin-top:28px}}
a{{color:#80c2ff}}
</style></head>
<body><main>
<div class="eyebrow">Seasonal time-series evaluation · Sri Lanka</div>
<h1>Rice production forecast<br>backtest</h1>
<p>Expanding-window evaluation of four target-history models. The target is seasonal rice production in thousand metric tonnes (not yield). Forecast inputs use observed production history and the future season label; realized test-period weather, acreage and macroeconomic values are excluded.</p>
<div class="kpis">
<div class="kpi"><b>149</b><span>seasonal observations · Yala 1950–Yala 2024</span></div>
<div class="kpi"><b>{origin_count}</b><span>forecast origins per horizon</span></div>
<div class="kpi"><b>1–6</b><span>season-ahead horizons</span></div>
<div class="kpi"><b>4</b><span>models compared on common origins</span></div>
</div>
<section class="panel"><h2>MAE by forecast horizon</h2><div class="legend"><span><i class="dot" style="background:{COLORS["SARIMAX"]}"></i>SARIMAX</span><span><i class="dot" style="background:{COLORS["SARIMAX + RF residual"]}"></i>SARIMAX + RF residual</span></div><div class="chart-wrap">{_chart(summary)}</div></section>
<section class="panel"><h2>Horizon-specific scores</h2><p>Lower MAE, RMSE, WAPE and MASE are better. Scores aggregate overlapping rolling forecasts and are descriptive, not independent confidence estimates.</p><div class="table-wrap"><table><thead><tr><th>Model</th><th>Horizon</th><th>Origins</th><th>MAE</th><th>RMSE</th><th>WAPE</th><th>MASE</th></tr></thead><tbody>{rows_html}</tbody></table></div></section>
<section class="notice"><strong>Interpretation and limits.</strong> This is a historical rolling-origin benchmark, not a guarantee of future performance. The available sample is small; the last historical years also participate in the rolling evaluation, so retain a final untouched prospective evaluation for any formal model-selection claim. The hybrid has no published prediction interval in this release. Climate-shock attribution is not established by these aggregate scores.</section>
<div class="foot">Generated from <code>outputs/backtest_summary.csv</code>. Rebuild with <code>python scripts/build_dashboard.py</code>. See <a href="docs/methodology.md">the methodology</a> and <a href="docs/data_dictionary.md">data dictionary</a>.</div>
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("outputs/backtest_summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("index.html"))
    args = parser.parse_args()
    build(args.summary, args.output)
    print(f"Dashboard written to {args.output}")


if __name__ == "__main__":
    main()
