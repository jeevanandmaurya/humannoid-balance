# Humanoid Standing Balance & Push Recovery

Train a 17-DoF MuJoCo humanoid (`Gymnasium Humanoid-v5`) to stand still and recover from random horizontal pushes using PPO, iterated across v1 → v4.

Best model: **v3 origin-locked** — 1000/1000 step episodes, ~10k eval return, withstands 100N+ pushes. **v4** extends this to a 50N → 300N curriculum (experimental).

- Interactive write-up: `index.html`
- Live dashboard: `streamlit_app.py` + `progress_dashboard.py`
- Clips: `assets/media/versions/`

## Results

| Version | Idea | Obs | Push | Episode len | Status |
|---|---|---|---|---|---|
| v1 Baseline | locomotion wrapper, no smoothing | 350-D | 0N | ~64 / 1000 | collapsed |
| v2 Action Buffer | prev-action obs + smoothness penalty `-0.02·\|a_t − a_{t-1}\|` | 367-D | 50N | 1000 / 1000 | first survival |
| v3 Origin Lock (best) | + quadratic drift penalty `-5.0·\|p_xy − p_target\|²` | 367-D | 100N | 1000 / 1000 | best |
| v4 Curriculum (exp.) | standing gate + settled bonus, 50N→300N, 1×→3× frequency | 367-D | 300N | updating | experimental |

Reward (v3/v4): alive + upright + height + feet − root vel/ang-vel − ctrl/action − smoothness − drift + posture − joint-vel (+ settled bonus in v4). Termination: torso `z < 0.8m` or tilt `> 60°` after 100-step grace.

PPO (all versions): `SubprocVecEnv ×12`, rollout 1024, batch 512, 5 epochs, γ 0.99, GAE 0.95, clip 0.2, entropy 1e-3, lr 1e-4, `VecNormalize`, 50M steps target.

## Repo layout

```
humanoid_balance_v4.py       # current training + watch (auto-resume, standing gate)
humanoid_balance_v3.py       # v3 origin-locked training
humanoid_balance_gymnasium.py# v1/v2 training
balance.py                   # MJX + Brax PPO variant (GPU-parallel, experimental)
humanoid_balance_colab.py / humanoid_balance_pybullet.py  # alt backends
progress_dashboard.py        # TB + evaluations.npz tracker, blog export, index.html sync
streamlit_app.py             # browser dashboard (Overview / Training Loop / Clips / Envs)
index.html                   # project website + model card + Chart.js curves
assets/media/versions/       # step0, v1_final, v2_final, v3_best GIFs/PNGs
assets/progress_data.json / progress_summary.csv / blog_stats.md  # exported curves
runs/                        # checkpoints, VecNormalize, TB logs (gitignored, 868MB)
```

## Install

```bash
pip install gymnasium[mujoco] stable-baselines3 tensorboard torch
# dashboard (optional)
pip install streamlit pandas
# MJX/Brax variant only (optional)
pip install mujoco mujoco-mjx brax flax optax "jax[cuda12]"
```

## Train / watch

```bash
# v4 — trains, auto-resumes from newest runs/humanoid_balance_v4/ckpts/ppo_*_steps.zip
python humanoid_balance_v4.py

# watch newest checkpoint (continuous viewer), or single eval
python humanoid_balance_v4.py --watch
python humanoid_balance_v4.py --watch --checkpoint runs/humanoid_balance_v4/ckpts/ppo_1000000_steps.zip
python humanoid_balance_v4.py --eval

# older versions
python humanoid_balance_v3.py --watch
python humanoid_balance_gymnasium.py --watch

# MJX/Brax PPO variant
python balance.py --envs 4096 --steps 200000000
```

Watch keys: `Space` pause · `R` reset · `P` toggle pushes · `F` manual push · `A` toggle AI · `-/=` `0` push force.

## Dashboards

```bash
# terminal summary / live / export
python progress_dashboard.py --summary
python progress_dashboard.py --live
python progress_dashboard.py --export --sync-index --summary

# desktop UI (tkinter + matplotlib)
python progress_dashboard.py --ui

# browser UI
streamlit run streamlit_app.py

# raw TensorBoard
tensorboard --logdir runs/humanoid_balance_v4/tb
```

`progress_dashboard.py` reads `runs/*/tb` (primary) + `runs/*/eval/evaluations.npz` (fallback), exports `assets/progress_data.json`, and refreshes `index.html` charts with `--sync-index`.

## Notes

- `runs/` is gitignored; publish final models via releases, not git.
- v1 config predates versioned scripts (values in dashboard marked estimated).
- Hardware reference: RTX 3050 Laptop / Ryzen 5, CPU training.
