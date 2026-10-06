#!/usr/bin/env python3
"""
Humanoid Balance — Streamlit Dashboard (browser UI)
===================================================
Complete training-loop view across v1..v4, side by side:

  Tab 1  Overview       eval reward + survival + comparison table + push bar
  Tab 2  Training Loop  all PPO internals (loss, KL, clip, entropy, lr, fps...)
  Tab 3  Clips          recorded GIFs side by side + launch live MuJoCo viewers
  Tab 4  Envs & System  parallel-env config + throughput + checkpoints

Run:
    streamlit run streamlit_app.py

Data backend is progress_dashboard.py (TB primary + evaluations.npz fallback).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from progress_dashboard import (
    ASSETS_DIR,
    CLIPS,
    INDEX_HTML,
    TRAIN_TAG_LABELS,
    VERSION_TRAIN_CONFIG,
    VERSIONS,
    WATCH_CMDS,
    export_all,
    get_ckpt_status,
    load_version_train_metrics,
    summarize_all,
    sync_index_html,
)

st.set_page_config(
    page_title="Humanoid Balance — v1–v4 Dashboard",
    layout="wide",
)

CACHE_TTL = 120  # seconds; TB re-read at most this often


# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading eval curves…")
def _summaries():
    return summarize_all()


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading PPO metrics…")
def _train_metrics(key: str):
    return load_version_train_metrics(key)


def _clear_cache():
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def eval_frame(summaries, keys, field="reward"):
    """Outer-join eval curves on steps_m so versions share one x-axis."""
    frames = []
    for k in keys:
        curve = summaries[k].get("curve", [])
        if not curve:
            continue
        df = pd.DataFrame(
            {"steps_m": [p["steps_m"] for p in curve],
             VERSIONS[k]["label"]: [p[field] for p in curve
                                    if p.get(field) is not None]})
        frames.append(df.set_index("steps_m"))
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return out.sort_index()


def metric_frame(all_metrics, keys, tag):
    """Outer-join one TB tag across versions on raw env step."""
    frames = []
    for k in keys:
        pts = all_metrics.get(k, {}).get(tag, [])
        if not pts:
            continue
        df = pd.DataFrame(pts, columns=["step", VERSIONS[k]["label"]])
        df["step_m"] = (df["step"] / 1e6).round(3)
        frames.append(df.set_index("step_m")[[VERSIONS[k]["label"]]])
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return out.sort_index()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
summaries = _summaries()

st.sidebar.title("Versions")
default_keys = [k for k in VERSIONS if summaries[k].get("n_points")]
keys = st.sidebar.multiselect(
    "Compare",
    options=list(VERSIONS),
    default=default_keys,
    format_func=lambda k: f"{k} — {VERSIONS[k]['label']}",
)
if not keys:
    st.warning("Select at least one version.")
    st.stop()

live = [k for k in keys if summaries[k].get("training_active")]
if live:
    st.sidebar.success("LIVE: " + ", ".join(live))
else:
    st.sidebar.caption("No active training detected.")

if st.sidebar.button("Refresh now"):
    _clear_cache()
    st.rerun()
st.sidebar.caption(f"Auto-refresh cache every {CACHE_TTL}s.")

with st.sidebar.expander("Blog / website"):
    if st.button("Export blog data"):
        paths = export_all(summaries)
        st.success("Wrote:\n" + "\n".join(f"`{p}`" for p in paths))
    if st.button("Sync index.html"):
        ok = sync_index_html(summaries, index_path=INDEX_HTML)
        st.success("index.html refreshed (backup .bak kept).") if ok \
            else st.info("No changes — already current.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Humanoid Balance — v1–v4 Training Dashboard")
st.caption("PPO standing-balance + 300N push recovery · 12 parallel envs · MuJoCo/Gymnasium")

tabs = st.tabs(["Overview", "Training Loop", "Clips", "Envs & System"])

# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    cols = st.columns(len(keys))
    for col, k in zip(cols, keys):
        s = summaries[k]
        with col:
            st.subheader(f"{k} {s.get('label','')}")
            if not s.get("n_points"):
                st.info("No data.")
                continue
            badge = " :green[● LIVE]" if s.get("training_active") else ""
            st.markdown(f"*{s.get('status','')}{badge}*")
            c1, c2 = st.columns(2)
            c1.metric("Last reward", f"{s['last_reward']:,.0f}")
            c2.metric("Best reward", f"{s['best_reward']:,.0f}",
                      f"@ {s['best_reward_steps_m']}M")
            c1.metric("Episode len", f"{s['last_len']:.0f} / 1000")
            c2.metric("Steps", f"{s['max_steps_m']:.1f}M")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Evaluation reward (all versions)**")
        df = eval_frame(summaries, keys, "reward")
        st.line_chart(df, height=320) if not df.empty else st.info("No curves.")
    with c2:
        st.markdown("**Episode survival duration**")
        df = eval_frame(summaries, keys, "ep_len")
        st.line_chart(df, height=320) if not df.empty else st.info("No curves.")

    st.markdown("**Side-by-side comparison** (Δ vs v1 last reward)")
    rows = []
    base = summaries["v1"].get("last_reward") if summaries.get("v1", {}).get("n_points") else None
    for k in keys:
        s = summaries[k]
        if not s.get("n_points"):
            continue
        delta = (f"{s['last_reward'] - base:+,.0f}"
                 if base and k != "v1" else "—")
        rows.append({"ver": k, "label": s["label"], "pts": s["n_points"],
                     "steps_M": s["max_steps_m"], "last_R": round(s["last_reward"], 1),
                     "best_R": round(s["best_reward"], 1),
                     "best_at_M": s["best_reward_steps_m"],
                     "last_len": s["last_len"], "Δ_vs_v1": delta,
                     "push_N": s.get("push_n"), "updated": s.get("last_updated"),
                     "live": bool(s.get("training_active"))})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("**Push resistance by version (N)**")
    push = pd.DataFrame(
        {"push_N": [summaries[k].get("push_n", 0) for k in keys]},
        index=[f"{k} {VERSIONS[k]['label']}" for k in keys])
    st.bar_chart(push, height=220)

# ---------------------------------------------------------------------------
# Tab 2 — Training Loop (all PPO internals)
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("Full PPO loop per version — loss terms, KL / clip health, entropy, LR, fps.")
    all_metrics = {k: _train_metrics(k) for k in keys}
    missing = [k for k in keys if not any(all_metrics[k].values())]
    if missing:
        st.info(f"No TB train metrics for: {', '.join(missing)} "
                "(v1 never wrote TB logs — eval curves only).")

    tag_choices = [t for t in TRAIN_TAG_LABELS if any(all_metrics[k].get(t) for k in keys)]
    sel = st.multiselect(
        "Metrics",
        options=tag_choices,
        default=[t for t in ["train/loss", "train/approx_kl",
                             "train/clip_fraction", "time/fps"] if t in tag_choices],
        format_func=lambda t: TRAIN_TAG_LABELS[t],
    )
    for tag in sel:
        st.markdown(f"**{TRAIN_TAG_LABELS[tag]}**  `{tag}`")
        df = metric_frame(all_metrics, keys, tag)
        if df.empty:
            st.caption("No data for this tag.")
        else:
            st.line_chart(df, height=240)

    st.markdown("**Last values (health check)** — high KL / clip_fraction ≈ policy churning")
    hrows = []
    for k in keys:
        row = {"ver": k}
        for tag in tag_choices:
            pts = all_metrics[k].get(tag, [])
            row[TRAIN_TAG_LABELS[tag]] = round(pts[-1][1], 5) if pts else None
        hrows.append(row)
    if hrows:
        st.dataframe(pd.DataFrame(hrows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 3 — Clips (recorded + live)
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("Recorded clips side by side — same fall vs. recovery story as the blog.")
    gif_keys = [k for k in keys if k in CLIPS]
    if gif_keys:
        cols = st.columns(len(gif_keys))
        for col, k in zip(cols, gif_keys):
            with col:
                st.markdown(f"**{k} — {VERSIONS[k]['label']}**")
                gif = CLIPS[k]["gif"]
                png = CLIPS[k]["png"]
                if gif.exists():
                    st.image(str(gif), caption=CLIPS[k]["caption"], use_container_width=True)
                elif png.exists():
                    st.image(str(png), caption=CLIPS[k]["caption"] + " (still — GIF missing)",
                             use_container_width=True)
                else:
                    st.warning("Clip missing.")
                st.caption(f"Push: {VERSIONS[k].get('push_n')}N · "
                           f"{summaries[k].get('status','')}")
    # step0 reference
    if "step0" in CLIPS and CLIPS["step0"]["gif"].exists():
        with st.expander("Reference: untrained free fall (step 0)"):
            st.image(str(CLIPS["step0"]["gif"]), caption=CLIPS["step0"]["caption"])

    st.divider()
    st.markdown("**Watch live** — launches the MuJoCo viewer for that version's newest checkpoint "
                "(separate window, same `--watch` you use in terminal).")
    wcols = st.columns(len([k for k in keys if k in WATCH_CMDS]))
    for col, k in zip(wcols, [k for k in keys if k in WATCH_CMDS]):
        with col:
            st.code(WATCH_CMDS[k], language="bash")
            if st.button(f"Launch {k} viewer", key=f"watch_{k}"):
                try:
                    subprocess.Popen([sys.executable, *WATCH_CMDS[k].split()[1:]],
                                     cwd=Path(__file__).resolve().parent)
                    st.success(f"{k} viewer launching…")
                except Exception as e:
                    st.error(f"Launch failed: {e}")
    st.caption("Tip: run training (`python humanoid_balance_v4.py`) in one terminal, "
               "`streamlit run streamlit_app.py` in another, viewer in a third.")

# ---------------------------------------------------------------------------
# Tab 4 — Envs & System
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("Parallel-env config + throughput + checkpoint inventory per version.")
    erows = []
    for k in keys:
        cfg = VERSION_TRAIN_CONFIG[k]
        ck = get_ckpt_status(k)
        s = summaries[k]
        m = _train_metrics(k)
        fps_pts = m.get("time/fps", [])
        erows.append({
            "ver": k,
            "script": cfg["script"] + (" (est.)" if cfg["estimated"] else ""),
            "envs": cfg["n_envs"],
            "vec": cfg["vec"],
            "target_M": cfg["total_steps"] / 1e6,
            "reached_M": s.get("max_steps_m"),
            "last_fps": round(fps_pts[-1][1], 0) if fps_pts else None,
            "ckpts": ck["n_ckpts"],
            "newest_ckpt": ck["newest_ckpt"],
            "best": ck["has_best"],
            "final": ck["has_final"],
            "vecnorm": ck["has_vecnormalize"],
            "updated": s.get("last_updated"),
            "live": bool(s.get("training_active")),
        })
    st.dataframe(pd.DataFrame(erows), use_container_width=True, hide_index=True)
    st.caption("All versions: SubprocVecEnv ×12, rollout 1024 steps, batch 512, "
               "5 epochs, γ 0.99, GAE 0.95, clip 0.2, entropy 1e-3, lr 1e-4. "
               "v1 values estimated (predates versioned scripts).")

    st.markdown("**Throughput (fps) — estimates wall-clock training speed**")
    fps_df = metric_frame({k: _train_metrics(k) for k in keys}, keys, "time/fps")
    st.line_chart(fps_df, height=240) if not fps_df.empty \
        else st.info("No fps data (v1 has no TB logs).")

    st.markdown("**Checkpoints on disk**")
    for k in keys:
        ck = get_ckpt_status(k)
        st.markdown(f"**{k}** — {ck['n_ckpts']} ckpts · "
                    f"best: {'yes' if ck['has_best'] else 'no'} · "
                    f"final: {'yes' if ck['has_final'] else 'no'} · "
                    f"VecNormalize: {'yes' if ck['has_vecnormalize'] else 'no'}"
                    + (f" · newest: `{ck['newest_ckpt']}`" if ck["newest_ckpt"] else ""))

st.divider()
st.caption(f"Source: `runs/` TB logs + `evaluations.npz` · blog data: `assets/progress_data.json` · "
           f"website sync: `--sync-index` in progress_dashboard.py")
