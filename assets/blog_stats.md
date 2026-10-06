# Humanoid Balance — Training Stats (auto-generated 2026-09-03T18:30:59)

| Version | Steps (M) | Last R | Best R | Last len | Max len | Push (N) | Status |
|---|---|---|---|---|---|---|---|
| v1 v1 Baseline | 8.5 | 530.79 | 1789.74 | 63.6 | 199.4 | 0 | done — collapsed |
| v2 v2 Action Buffer | 50.0 | 10194.52 | 10249.88 | 1000.0 | 1000.0 | 50 | done — first 1000-step survival |
| v3 v3 Origin Lock (Best) | 50.0 | 10171.13 | 10197.27 | 1000.0 | 1000.0 | 100 | best model |
| v4 v4 Curriculum (Exp.) | 12.5 | 8745.29 | 8745.29 | 1000.0 | 1000.0 | 300 | experimental — updating |

## Copy-paste blog numbers
- Best model: v3 origin-locked, best R ≈ 10197.27 @ 47.5M steps.
- Max push: 300N curriculum (v4, experimental, last R ≈ 8745.29).
- Total curves JSON: `assets/progress_data.json` → paste `chart_js` straight into Chart.js.
