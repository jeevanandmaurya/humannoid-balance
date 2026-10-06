#!/usr/bin/env python3
"""
Humanoid Balance — Unified Progress Dashboard & Blog Data Generator
=================================================================
One general file that combines ALL training versions (v1..v4) in one place.

What it does
------------
1. Loads training progress from TensorBoard logs (primary) + evaluations.npz (fallback)
   for every version under ./runs/.
2. Shows a realtime desktop UI (tkinter + matplotlib) with Refresh + auto-refresh,
   so you can watch progress live instead of opening 4 separate TensorBoards.
3. Exports blog-ready data: JSON + CSV + Markdown table.
4. Syncs fresh curves into index.html (Chart.js const v1Data..v4Data + version stats)
   so the website always shows the latest data. Run with --sync-index.

Usage
-----
    python progress_dashboard.py --summary            # print table in terminal
    python progress_dashboard.py --ui                 # open realtime dashboard UI
    python progress_dashboard.py --live               # text live-watch in terminal
    python progress_dashboard.py --export             # write JSON + CSV + MD to assets/
    python progress_dashboard.py --sync-index         # refresh index.html charts from runs/
    python progress_dashboard.py --export --sync-index --summary   # do everything

    python progress_dashboard.py --ui --refresh 15    # UI auto-refresh every 15s

Requirements: numpy (required), matplotlib + tensorboard (optional but recommended).
Everything degrades gracefully to evaluations.npz if tensorboard/matplotlib missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "runs"
ASSETS_DIR = ROOT / "assets"
INDEX_HTML = ROOT / "index.html"

# ---------------------------------------------------------------------------
# 1. Version registry — edit here when you add v5, v6, ...
# ---------------------------------------------------------------------------
VERSIONS = OrderedDict([
    ("v1", {
        "label": "v1 Baseline",
        "run_dir": RUNS_ROOT / "humanoid_balance",      # v1 logs live here (no _v1 suffix)
        "tb_subdir": "tb",                               # <run_dir>/tb/PPO_* (may be empty for v1)
        "color": "#d62728",
        "status": "done — collapsed",
        "obs_dim": 350,
        "push_n": 0,                                     # max push withstood (N), for blog bar chart
        "desc": "Locomotion wrapper, no smoothing. Fell in ~74 steps.",
    }),
    ("v2", {
        "label": "v2 Action Buffer",
        "run_dir": RUNS_ROOT / "humanoid_balance_v2",
        "tb_subdir": "tb",
        "color": "#1b9e77",
        "status": "done — first 1000-step survival",
        "obs_dim": 367,
        "push_n": 50,
        "desc": "Prev-action obs (367-D) + smoothness penalty.",
    }),
    ("v3", {
        "label": "v3 Origin Lock (Best)",
        "run_dir": RUNS_ROOT / "humanoid_balance_v3",
        "tb_subdir": "tb",
        "color": "#7570b3",
        "status": "best model",
        "obs_dim": 367,
        "push_n": 100,
        "desc": "Quadratic drift penalty, origin-locked balance.",
    }),
    ("v4", {
        "label": "v4 Curriculum (Exp.)",
        "run_dir": RUNS_ROOT / "humanoid_balance_v4",
        "tb_subdir": "tb",
        "color": "#141413",
        "status": "experimental — updating",
        "obs_dim": 367,
        "push_n": 300,
        "desc": "Standing gate + settled bonus, 50N→300N curriculum.",
    }),
])

EVAL_TAG_REWARD = "eval/mean_reward"
EVAL_TAG_LEN = "eval/mean_ep_length"

# Full PPO training-loop tags (tab 2 of the Streamlit app). fps lives outside train/.
TRAIN_TAGS = [
    "train/loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/clip_range",
    "train/explained_variance",
    "train/std",
    "train/learning_rate",
    "time/fps",
]
TRAIN_TAG_LABELS = {
    "train/loss": "Total loss",
    "train/policy_gradient_loss": "Policy-gradient loss",
    "train/value_loss": "Value loss",
    "train/entropy_loss": "Entropy loss",
    "train/approx_kl": "Approx KL",
    "train/clip_fraction": "Clip fraction",
    "train/clip_range": "Clip range",
    "train/explained_variance": "Explained variance",
    "train/std": "Action std",
    "train/learning_rate": "Learning rate",
    "time/fps": "Throughput (fps)",
}

# Parallel-env + PPO config per version (all versions use 12 SubprocVecEnvs;
# v1 config predates the versioned scripts so values are marked estimated).
VERSION_TRAIN_CONFIG = {
    "v1": {"script": "humanoid_balance_gymnasium.py (early run)", "n_envs": 12,
           "total_steps": 8_500_000, "episode_len": 1000, "lr": 1e-4,
           "n_steps": 1024, "batch": 512, "epochs": 5, "gamma": 0.99,
           "gae": 0.95, "clip": 0.2, "ent": 1e-3, "vec": "SubprocVecEnv + VecNormalize",
           "estimated": True},
    "v2": {"script": "humanoid_balance_gymnasium.py", "n_envs": 12,
           "total_steps": 50_000_000, "episode_len": 1000, "lr": 1e-4,
           "n_steps": 1024, "batch": 512, "epochs": 5, "gamma": 0.99,
           "gae": 0.95, "clip": 0.2, "ent": 1e-3, "vec": "SubprocVecEnv + VecNormalize",
           "estimated": False},
    "v3": {"script": "humanoid_balance_v3.py", "n_envs": 12,
           "total_steps": 50_000_000, "episode_len": 1000, "lr": 1e-4,
           "n_steps": 1024, "batch": 512, "epochs": 5, "gamma": 0.99,
           "gae": 0.95, "clip": 0.2, "ent": 1e-3, "vec": "SubprocVecEnv + VecNormalize",
           "estimated": False},
    "v4": {"script": "humanoid_balance_v4.py", "n_envs": 12,
           "total_steps": 50_000_000, "episode_len": 1000, "lr": 1e-4,
           "n_steps": 1024, "batch": 512, "epochs": 5, "gamma": 0.99,
           "gae": 0.95, "clip": 0.2, "ent": 1e-3, "vec": "SubprocVecEnv + VecNormalize",
           "estimated": False},
}

# Recorded clips for the side-by-side Clips tab (both per-version + hero demos).
CLIPS = OrderedDict([
    ("v1", {"gif": ASSETS_DIR / "media/versions/v1_final.gif",
            "png": ASSETS_DIR / "media/versions/v1_final_fall.png",
            "caption": "v1 final — collapses under push"}),
    ("v2", {"gif": ASSETS_DIR / "media/versions/v2_final.gif",
            "png": ASSETS_DIR / "media/versions/v2_final_balanced.png",
            "caption": "v2 final — first stable stance"}),
    ("v3", {"gif": ASSETS_DIR / "media/versions/v3_best.gif",
            "png": ASSETS_DIR / "media/versions/v3_best_final.png",
            "caption": "v3 best — origin-locked balance"}),
    ("v4", {"gif": ASSETS_DIR / "media/versions/v4_experimental.gif",
            "png": ASSETS_DIR / "media/versions/v4_experimental.png",
            "caption": "v4 experimental — curriculum attempt"}),
    ("step0", {"gif": ASSETS_DIR / "media/versions/step0_free_fall.gif",
               "png": ASSETS_DIR / "media/versions/step0_free_fall.png",
               "caption": "Step 0 — untrained free fall"}),
])
# Live MuJoCo viewer commands per version (Clips tab "watch live" buttons).
WATCH_CMDS = {
    "v2": "python humanoid_balance_gymnasium.py --watch",
    "v3": "python humanoid_balance_v3.py --watch",
    "v4": "python humanoid_balance_v4.py --watch",
}


# ---------------------------------------------------------------------------
# 2. Loaders — TensorBoard (primary) + evaluations.npz (fallback)
# ---------------------------------------------------------------------------
def load_tb_curve(run_dir: Path, tb_subdir: str = "tb"):
    """Return {step: (reward, ep_len)} merged across all TB event files. {} if none."""
    tb_root = run_dir / tb_subdir
    if not tb_root.exists():
        return {}
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return {}
    merged: dict[int, list] = {}
    # Each PPO_*/ subdir may hold several event files from resumed runs; EA merges them.
    tb_dirs = [d for d in tb_root.iterdir() if d.is_dir()] or [tb_root]
    for d in tb_dirs:
        try:
            ea = event_accumulator.EventAccumulator(
                str(d), size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            if EVAL_TAG_REWARD not in tags:
                continue
            rewards = {s.step: s.value for s in ea.Scalars(EVAL_TAG_REWARD)}
            lens = {s.step: s.value for s in ea.Scalars(EVAL_TAG_LEN)} \
                if EVAL_TAG_LEN in tags else {}
            for step, r in rewards.items():
                merged[step] = [r, lens.get(step)]
        except Exception as e:
            print(f"[warn] TB read failed for {d}: {e}", file=sys.stderr)
    # drop entries without reward; keep len=None -> filled later from npz
    return {k: (float(v[0]), None if v[1] is None else float(v[1]))
            for k, v in merged.items()}


def load_npz_curve(run_dir: Path):
    """Return {step: (reward, ep_len)} from eval/evaluations.npz. {} if none."""
    npz_path = run_dir / "eval" / "evaluations.npz"
    if not npz_path.exists():
        return {}
    try:
        d = np.load(str(npz_path))
        steps = d["timesteps"].astype(int)
        rewards = d["results"].mean(axis=1)
        lens = d["ep_lengths"].mean(axis=1)
        return {int(s): (float(r), float(l))
                for s, r, l in zip(steps, rewards, lens)}
    except Exception as e:
        print(f"[warn] npz read failed for {npz_path}: {e}", file=sys.stderr)
        return {}


def load_version_curve(key: str):
    """Combined curve sorted by step. Prefers TB, backfills missing lens from npz."""
    cfg = VERSIONS[key]
    tb = load_tb_curve(cfg["run_dir"], cfg["tb_subdir"])
    npz = load_npz_curve(cfg["run_dir"])
    merged = dict(npz)
    merged.update({k: v for k, v in tb.items() if v[0] is not None})
    # backfill None lens from npz nearest step
    for step, (r, l) in list(merged.items()):
        if l is None and npz:
            nearest = min(npz, key=lambda s: abs(s - step))
            merged[step] = (r, npz[nearest][1])
    pts = sorted(merged.items())  # [(step, (reward, len))]
    # steps_m rounded to 2 decimals so generated Chart.js x-values match the
    # hand-rounded values already in index.html (0.5, 8.5, 50.0, ...) and
    # future --sync-index runs produce zero-noise diffs.
    return [{"steps": s, "steps_m": round(s / 1e6, 2), "reward": round(r, 2),
             "ep_len": round(l, 1) if l is not None else None}
            for s, (r, l) in pts]


def load_tb_scalars(run_dir: Path, tags, tb_subdir: str = "tb"):
    """Generic TB reader: {tag: [(step, value)]} merged across event files. Missing -> []."""
    out = {t: {} for t in tags}
    tb_root = run_dir / tb_subdir
    if not tb_root.exists():
        return {t: [] for t in tags}
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return {t: [] for t in tags}
    tb_dirs = [d for d in tb_root.iterdir() if d.is_dir()] or [tb_root]
    for d in tb_dirs:
        try:
            ea = event_accumulator.EventAccumulator(
                str(d), size_guidance={"scalars": 0})
            ea.Reload()
            available = set(ea.Tags().get("scalars", []))
            for t in tags:
                if t in available:
                    for s in ea.Scalars(t):
                        out[t][s.step] = float(s.value)  # later files win on overlap
        except Exception as e:
            print(f"[warn] TB read failed for {d}: {e}", file=sys.stderr)
    return {t: sorted(steps.items()) for t, steps in out.items()}


def load_version_train_metrics(key: str):
    """All PPO-loop scalars for one version: {tag: [(step, value)]}."""
    cfg = VERSIONS[key]
    return load_tb_scalars(cfg["run_dir"], TRAIN_TAGS, cfg["tb_subdir"])


def get_ckpt_status(key: str):
    """Checkpoint inventory for the Envs tab: counts + newest ckpt + best/final presence."""
    cfg = VERSIONS[key]
    ckpt_dir = cfg["run_dir"] / "ckpts"
    best = cfg["run_dir"] / "best" / "best_model.zip"
    final = cfg["run_dir"] / "final_model.zip"
    vecnorm = cfg["run_dir"] / "vecnormalize.pkl"
    ckpts = sorted(ckpt_dir.glob("ppo_*_steps.zip")) if ckpt_dir.exists() else []
    newest = None
    if ckpts:
        try:
            newest = max(ckpts, key=lambda p: p.stat().st_mtime).name
        except OSError:
            newest = ckpts[-1].name
    return {
        "n_ckpts": len(ckpts),
        "newest_ckpt": newest,
        "has_best": best.exists(),
        "has_final": final.exists(),
        "has_vecnormalize": vecnorm.exists(),
    }


def run_last_modified(run_dir: Path) -> str | None:
    """Newest mtime under run_dir (for 'last updated' + realtime staleness)."""
    if not run_dir.exists():
        return None
    newest = 0.0
    for p in run_dir.rglob("*"):
        try:
            if p.is_file():
                newest = max(newest, p.stat().st_mtime)
        except OSError:
            pass
    return datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M") if newest else None


def is_training_active(run_dir: Path, stale_sec: int = 600) -> bool:
    """Heuristic: any file under run_dir modified within stale_sec => likely training."""
    if not run_dir.exists():
        return False
    now = time.time()
    for p in run_dir.rglob("*.tfevents*"):
        try:
            if now - p.stat().st_mtime < stale_sec:
                return True
        except OSError:
            pass
    return False


# ---------------------------------------------------------------------------
# 3. Stats aggregation
# ---------------------------------------------------------------------------
def summarize_version(key: str, curve=None):
    cfg = VERSIONS[key]
    curve = curve if curve is not None else load_version_curve(key)
    if not curve:
        return {"key": key, "label": cfg["label"], "n_points": 0,
                "status": cfg["status"], "exists": cfg["run_dir"].exists(),
                "curve": []}
    rewards = [p["reward"] for p in curve]
    lens = [p["ep_len"] for p in curve if p["ep_len"] is not None]
    last = curve[-1]
    best_i = int(np.argmax(rewards))
    return {
        "key": key,
        "label": cfg["label"],
        "color": cfg["color"],
        "status": cfg["status"],
        "desc": cfg["desc"],
        "obs_dim": cfg["obs_dim"],
        "push_n": cfg["push_n"],
        "exists": True,
        "n_points": len(curve),
        "max_steps": last["steps"],
        "max_steps_m": last["steps_m"],
        "last_reward": last["reward"],
        "last_len": last["ep_len"],
        "best_reward": round(float(rewards[best_i]), 2),
        "best_reward_steps_m": curve[best_i]["steps_m"],
        "max_len": round(float(max(lens)), 1) if lens else None,
        "survival_rate": f"{last['ep_len']:.0f} / 1000" if last["ep_len"] else "n/a",
        "last_updated": run_last_modified(cfg["run_dir"]),
        "training_active": is_training_active(cfg["run_dir"]),
        "curve": curve,
    }


def summarize_all():
    return OrderedDict((k, summarize_version(k)) for k in VERSIONS)


def print_summary(summaries):
    print(f"\n{'ver':<5}{'label':<24}{'pts':>5}{'max_steps':>11}{'last_R':>10}{'best_R':>10}{'last_len':>10}  status")
    print("-" * 100)
    for k, s in summaries.items():
        if not s["n_points"]:
            print(f"{k:<5}{s['label']:<24}{'0':>5}{'--':>11}{'--':>10}{'--':>10}{'--':>10}  {s['status']} (no data)")
            continue
        live = " [LIVE]" if s["training_active"] else ""
        print(f"{k:<5}{s['label']:<24}{s['n_points']:>5}{s['max_steps_m']:>10.2f}M"
              f"{s['last_reward']:>10.1f}{s['best_reward']:>10.1f}"
              f"{s['last_len']:>10.1f}  {s['status']}{live}")
    total_max = max((s.get("max_steps_m") or 0 for s in summaries.values()), default=0)
    print(f"\nMax single-version steps: {total_max:.2f}M | Last updated: " +
          ", ".join(f"{k}={s.get('last_updated')}" for k, s in summaries.items()))
    print("Tip: full curves -> assets/progress_data.json ; website refresh -> --sync-index\n")


# ---------------------------------------------------------------------------
# 4. Blog-ready exports (JSON + CSV + Markdown)
# ---------------------------------------------------------------------------
def export_all(summaries, out_dir: Path = ASSETS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "progress_data.json"
    csv_path = out_dir / "progress_summary.csv"
    md_path = out_dir / "blog_stats.md"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "versions": {k: {kk: vv for kk, vv in s.items() if kk != "curve"}
                     for k, s in summaries.items()},
        "curves": {k: [{"x": p["steps_m"], "y": p["reward"], "len": p["ep_len"]}
                       for p in s.get("curve", [])]
                   for k, s in summaries.items()},
        # Chart.js-ready aliases matching index.html const names
        "chart_js": {f"{k}Data": [{"x": p["steps_m"], "y": p["reward"], "len": p["ep_len"]}
                                  for p in summaries[k].get("curve", [])]
                     for k in summaries},
        "push_bar": {k: summaries[k].get("push_n") for k in summaries},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["version", "label", "n_points", "max_steps_M", "last_reward",
                    "best_reward", "best_at_M", "last_len", "max_len",
                    "push_N", "obs_dim", "status", "last_updated"])
        for k, s in summaries.items():
            w.writerow([k, s.get("label"), s.get("n_points", 0),
                        s.get("max_steps_m", ""), s.get("last_reward", ""),
                        s.get("best_reward", ""), s.get("best_reward_steps_m", ""),
                        s.get("last_len", ""), s.get("max_len", ""),
                        s.get("push_n", ""), s.get("obs_dim", ""),
                        s.get("status", ""), s.get("last_updated", "")])

    lines = [f"# Humanoid Balance — Training Stats (auto-generated {payload['generated_at']})",
             "",
             "| Version | Steps (M) | Last R | Best R | Last len | Max len | Push (N) | Status |",
             "|---|---|---|---|---|---|---|---|"]
    for k, s in summaries.items():
        lines.append(f"| {k} {s.get('label','')} | {s.get('max_steps_m','--')} | "
                     f"{s.get('last_reward','--')} | {s.get('best_reward','--')} | "
                     f"{s.get('last_len','--')} | {s.get('max_len','--')} | "
                     f"{s.get('push_n','--')} | {s.get('status','')} |")
    lines += ["",
              "## Copy-paste blog numbers",
              f"- Best model: v3 origin-locked, best R ≈ {summaries['v3'].get('best_reward','?')} "
              f"@ {summaries['v3'].get('best_reward_steps_m','?')}M steps." if summaries.get("v3") else "- v3: n/a",
              f"- Max push: 300N curriculum (v4, experimental, last R ≈ {summaries['v4'].get('last_reward','?')})." if summaries.get("v4") else "",
              f"- Total curves JSON: `assets/progress_data.json` → paste `chart_js` straight into Chart.js.",
              ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[export] {json_path}\n[export] {csv_path}\n[export] {md_path}")
    return json_path, csv_path, md_path


# ---------------------------------------------------------------------------
# 5. index.html sync — overwrite embedded Chart.js data with latest curves
# ---------------------------------------------------------------------------
def _js_array(points):
    # [{"x":.., "y":.., "len":..}, ...] compact, matches existing index.html style
    items = ", ".join(
        f'{{\"x\": {p["steps_m"]}, \"y\": {p["reward"]}, \"len\": {p["ep_len"]}}}'
        for p in points)
    return f"[{items}]"


def sync_index_html(summaries, index_path: Path = INDEX_HTML, dry_run: bool = False):
    text = index_path.read_text(encoding="utf-8")
    orig = text
    report = []

    # 5a. Replace const v1Data..v4Data arrays
    for k in VERSIONS:
        curve = summaries[k].get("curve", [])
        if not curve:
            report.append(f"{k}: no data, skipped")
            continue
        new_arr = _js_array(curve)
        pat = re.compile(rf"const\s+{k}Data\s*=\s*\[.*?\];", re.DOTALL)
        if not pat.search(text):
            report.append(f"{k}: const {k}Data not found in HTML, skipped")
            continue
        text = pat.sub(f"const {k}Data = {new_arr};", text, count=1)
        report.append(f"{k}: {len(curve)} pts, up to {curve[-1]['steps_m']}M steps")

    # 5b. Refresh the 4 version-stats divs (in v1..v4 document order)
    stat_divs = list(re.finditer(
        r'<div class="version-stats">.*?</div>', text, re.DOTALL))
    if len(stat_divs) >= 4:
        new_blocks = []
        for k in ["v1", "v2", "v3", "v4"]:
            s = summaries[k]
            if not s["n_points"]:
                new_blocks.append(None)
                continue
            if k == "v4":
                new_blocks.append(
                    f'<div class="version-stats">Curriculum: 50N &rarr; 300N (3x freq) '
                    f'&middot; Mean Episode Duration: {s["last_len"]:.1f} steps '
                    f'&middot; Reward: ~{s["last_reward"]:,.0f}</div>')
            else:
                new_blocks.append(
                    f'<div class="version-stats">Observation: {s["obs_dim"]}-D '
                    f'&middot; Mean Episode Duration: {s["last_len"]:.1f} steps '
                    f'&middot; Reward: ~{s["last_reward"]:,.0f}</div>')
        # replace from last to first to keep indices valid
        for m, nb in zip(reversed(stat_divs), reversed(new_blocks)):
            if nb:
                text = text[:m.start()] + nb + text[m.end():]
        report.append("version-stats divs: refreshed")
    else:
        report.append("version-stats divs: not found (layout changed?), skipped")

    # 5c. Quick-spec total steps = max across versions
    max_m = max((s.get("max_steps_m") or 0 for s in summaries.values()), default=0)
    if max_m:
        pat = re.compile(
            r'(<div class="spec-label">Total Steps Trained</div>\s*'
            r'<div class="spec-val">)[^<]*(</div>)')
        if pat.search(text):
            text = pat.sub(rf"\g<1>{max_m:.1f}M+\g<2>", text, count=1)
            report.append(f"quick-spec total steps: {max_m:.1f}M+")

    # 5d. Chart note timestamp
    note_pat = re.compile(r"(Logged from <code>runs/humanoid_balance\*/eval/evaluations\.npz</code>)[^.]*\.")
    if note_pat.search(text):
        stamp = datetime.now().strftime("%Y-%m-%d")
        text = note_pat.sub(rf"\1 and TensorBoard event logs (refreshed {stamp}).", text, count=1)

    if text == orig:
        print("[sync] no changes (already up to date?)")
        return False
    if dry_run:
        print("[sync dry-run] would change:")
        print("\n".join("  - " + r for r in report))
        return True
    backup = index_path.with_suffix(".html.bak")
    shutil.copy(index_path, backup)
    index_path.write_text(text, encoding="utf-8")
    print(f"[sync] {index_path.name} updated (backup: {backup.name})")
    print("\n".join("  - " + r for r in report))
    return True


# ---------------------------------------------------------------------------
# 6. Realtime desktop UI (tkinter + matplotlib, auto-refresh)
# ---------------------------------------------------------------------------
def launch_ui(refresh_sec: int = 10):
    import tkinter as tk
    from tkinter import ttk, messagebox
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        has_mpl = True
    except ImportError:
        has_mpl = False
        plt = None

    root = tk.Tk()
    root.title("Humanoid Balance — Progress Dashboard (v1–v4)")
    root.geometry("1100x700")

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    ttk.Label(top, text="Humanoid Balance — v1–v4 Tracker",
              font=("Segoe UI", 13, "bold")).pack(side="left")
    status_var = tk.StringVar(value="loading…")
    ttk.Label(top, textvariable=status_var).pack(side="left", padx=12)

    btn_frame = ttk.Frame(top)
    btn_frame.pack(side="right")
    summaries: OrderedDict = OrderedDict()

    def refresh():
        nonlocal summaries
        try:
            summaries = summarize_all()
            render()
            live = [k for k, s in summaries.items() if s.get("training_active")]
            stamp = datetime.now().strftime("%H:%M:%S")
            status_var.set(f"Updated {stamp} · "
                           + (f"LIVE: {', '.join(live)}" if live else "no active training"))
        except Exception as e:
            status_var.set(f"refresh failed: {e}")
        root.after(refresh_sec * 1000, refresh)

    def do_export():
        try:
            export_all(summaries if summaries else summarize_all())
            messagebox.showinfo("Export", "Wrote assets/progress_data.json + .csv + blog_stats.md")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def do_sync():
        try:
            sync_index_html(summaries if summaries else summarize_all())
            messagebox.showinfo("Sync", "index.html refreshed from latest runs/")
        except Exception as e:
            messagebox.showerror("Sync failed", str(e))

    ttk.Button(btn_frame, text="Refresh now",
               command=lambda: root.after(0, refresh)).pack(side="left", padx=3)
    ttk.Button(btn_frame, text="Export blog data", command=do_export).pack(side="left", padx=3)
    ttk.Button(btn_frame, text="Sync index.html", command=do_sync).pack(side="left", padx=3)

    body = ttk.PanedWindow(root, orient="horizontal")
    body.pack(fill="both", expand=True, padx=8, pady=4)

    left = ttk.Frame(body, width=340)
    body.add(left, weight=1)
    txt = tk.Text(left, wrap="word", font=("Consolas", 9), width=42)
    txt.pack(fill="both", expand=True)

    right = ttk.Frame(body)
    body.add(right, weight=3)
    canvas_holder = ttk.Frame(right)
    canvas_holder.pack(fill="both", expand=True)
    fig = ax1 = ax2 = canvas = None
    if has_mpl:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
        fig.tight_layout(pad=2.5)
        canvas = FigureCanvasTkAgg(fig, master=canvas_holder)
        canvas.get_tk_widget().pack(fill="both", expand=True)
    else:
        ttk.Label(canvas_holder,
                  text="matplotlib not installed — text panel still live.\n"
                       "pip install matplotlib for charts.").pack(pady=40)

    def render():
        # text panel
        txt.delete("1.0", "end")
        for k, s in summaries.items():
            if not s["n_points"]:
                txt.insert("end", f"{k} {s['label']}: no data\n\n")
                continue
            flag = " ●LIVE" if s["training_active"] else ""
            txt.insert("end",
                       f"{k} — {s['label']}{flag}\n"
                       f"  pts={s['n_points']}  max={s['max_steps_m']}M\n"
                       f"  last R={s['last_reward']:,.1f}  len={s['last_len']}\n"
                       f"  best R={s['best_reward']:,.1f} @ {s['best_reward_steps_m']}M\n"
                       f"  push={s['push_n']}N  updated={s['last_updated']}\n\n")
        # charts
        if has_mpl:
            for ax in (ax1, ax2):
                ax.clear()
            for k, s in summaries.items():
                curve = s.get("curve", [])
                if not curve:
                    continue
                xs = [p["steps_m"] for p in curve]
                color = VERSIONS[k]["color"]
                ls = "--" if k == "v4" else "-"
                ax1.plot(xs, [p["reward"] for p in curve],
                         label=f"{k} {VERSIONS[k]['label']}", color=color, ls=ls, lw=2)
                ax2.plot(xs, [p["ep_len"] for p in curve if p["ep_len"] is not None],
                         color=color, ls=ls, lw=2)
            ax1.set_ylabel("Eval reward")
            ax1.legend(fontsize=8, loc="upper left")
            ax1.grid(alpha=0.3)
            ax2.set_ylabel("Episode length")
            ax2.set_xlabel("Env steps (M)")
            ax2.set_ylim(0, 1050)
            ax2.grid(alpha=0.3)
            fig.tight_layout()
            canvas.draw()

    root.after(100, refresh)
    root.mainloop()


def live_text_loop(refresh_sec: int = 15):
    print("Live watch — Ctrl+C to stop. (TB + npz polled every "
          f"{refresh_sec}s)\n")
    try:
        while True:
            print("\033c", end="")  # clear terminal
            print(f"=== Humanoid Balance live — {datetime.now():%Y-%m-%d %H:%M:%S} ===")
            print_summary(summarize_all())
            time.sleep(refresh_sec)
    except KeyboardInterrupt:
        print("\nstopped.")


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Unified v1–v4 progress tracker + blog exporter + index.html sync")
    ap.add_argument("--summary", action="store_true", help="print version table")
    ap.add_argument("--ui", action="store_true", help="open realtime dashboard UI")
    ap.add_argument("--live", action="store_true", help="text live-watch loop in terminal")
    ap.add_argument("--export", action="store_true", help="write JSON + CSV + MD to assets/")
    ap.add_argument("--sync-index", action="store_true", help="refresh index.html charts from runs/")
    ap.add_argument("--dry-run", action="store_true", help="with --sync-index: preview only")
    ap.add_argument("--refresh", type=int, default=10, help="UI/live refresh seconds (default 10)")
    ap.add_argument("--index-path", type=Path, default=INDEX_HTML)
    ap.add_argument("--out-dir", type=Path, default=ASSETS_DIR)
    args = ap.parse_args(argv)

    if not any([args.summary, args.ui, args.live, args.export, args.sync_index]):
        args.summary = True  # default action

    summaries = None
    if args.ui:
        launch_ui(refresh_sec=args.refresh)
        return 0
    if args.live:
        live_text_loop(refresh_sec=args.refresh)
        return 0

    summaries = summarize_all()
    if args.summary:
        print_summary(summaries)
    if args.export:
        export_all(summaries, out_dir=args.out_dir)
    if args.sync_index:
        sync_index_html(summaries, index_path=args.index_path, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
