# MarchMadSim

Monte Carlo simulator for the NCAA Men's and Women's March Madness tournaments. Trains a calibrated probabilistic model from historical Kaggle data and runs thousands of bracket simulations to produce win probabilities, round-by-round paths, and Kaggle competition submissions.

> **Disclaimer:** For educational and entertainment purposes only. Not intended for betting or financial decisions.

---

## Features

- **Four Factors matchup model** — win probability is derived from pace-adjusted offensive and defensive efficiency (effective FG%, turnover rate, offensive rebounding, free throw rate), not just a single net rating number.
- **Heavy-tailed shock distribution** — Student-t noise calibrated against historical seed-matchup upset rates (1v16 ~1%, 5v12 ~35%, 8v9 ~49%) to reproduce real upset frequency.
- **Temperature scaling** — post-hoc calibration fitted via leave-one-year-out CV, so predicted probabilities match actual win rates across confidence bins.
- **Schedule features** — recent form (opponent-quality-weighted win rate over the last 10 games) and season trajectory derived from regular-season results.
- **Coach & neutral court features** (men's) — career NCAA tournament wins per coach, neutral-court win percentage, consensus ranking from Massey/Sagarin/AP.
- **Fatigue modeling** — teams accumulate fatigue in close games; decays between rounds so it doesn't compound unrealistically over six rounds.
- **First Four support** — bracket builder handles the play-in games and substitutes winners into the correct slots.
- **Two simulation modes** — Monte Carlo (10k full brackets → championship probability distribution) and round-by-round (1k sims per game → most likely path).
- **Kaggle submission** — generates `submission_2026.csv` with win probabilities for every possible team pair (men's + women's combined).
- **No KenPom required** — all data comes from the free Kaggle March Mania dataset.

---

## Project structure

```
MarchMadSim/
├── marchmadness/          # Core Python package
│   ├── data.py            # Feature engineering (Four Factors, schedule, Kaggle stats)
│   ├── model.py           # Win probability model + calibration
│   ├── simulator.py       # Monte Carlo + round-by-round simulation
│   ├── bracket.py         # Bracket / matchup generation incl. First Four
│   ├── training.py        # Parameter optimization + cross-validation
│   └── output.py          # Print helpers
├── M_train.ipynb          # Train men's model → data/trained_params.json
├── M_simulate_2026.ipynb  # Simulate 2026 men's bracket
├── W_train.ipynb          # Train women's model → data/W_trained_params.json
├── W_simulate_2026.ipynb  # Simulate 2026 women's bracket
├── data/
│   ├── kaggle/            # Kaggle data files (not committed — download below)
│   ├── trained_params.json
│   ├── feature_scaler.json
│   ├── W_trained_params.json
│   └── W_feature_scaler.json
└── requirements.txt
```

---

## Quick start

**1. Get the data**

Download the [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) dataset and place the CSV files in `data/kaggle/`.

Files needed:

| File | Used by |
|---|---|
| `MRegularSeasonDetailedResults.csv` | M train + simulate |
| `MNCAATourneyCompactResults.csv` | M train |
| `MNCAATourneySeeds.csv` | M train |
| `MTeams.csv` | M train + simulate |
| `MTeamSpellings.csv` | M simulate (optional Kaggle features) |
| `MTeamCoaches.csv` | M train (optional) |
| `MMasseyOrdinals.csv` | M train (optional) |
| `WRegularSeasonDetailedResults.csv` | W train + simulate |
| `WNCAATourneyCompactResults.csv` | W train |
| `WNCAATourneySeeds.csv` | W train |
| `WTeams.csv` | W train + simulate |
| `SampleSubmissionStage2.csv` | Kaggle submission |

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Train the model** (once per season)

Open and run `M_train.ipynb` for men's, `W_train.ipynb` for women's. This calibrates shock parameters against historical upset rates, trains feature weights via L-BFGS-B, runs leave-one-year-out cross-validation, fits temperature scaling, and saves `data/trained_params.json` + `data/feature_scaler.json`.

**4. Run the simulator**

Open `M_simulate_2026.ipynb` or `W_simulate_2026.ipynb`, update the bracket dict in Cell 4 if needed, and run all cells.

---

## How it works

### Win probability

Each game is evaluated with a pace-adjusted Four Factors matchup:

```
# Pace-adjusted expected scoring margin
game_pace  = (adj_t_a + adj_t_b) / 2
a_pts = adj_o_a * (adj_d_b / league_avg_o) * (game_pace / league_avg_t)
b_pts = adj_o_b * (adj_d_a / league_avg_o) * (game_pace / league_avg_t)
base_margin = (a_pts - b_pts) / margin_scale

# Secondary signals
recent_form_contrib = recent_form_weight * (form_a - form_b)
luck_contrib        = luck_weight        * (luck_a  - luck_b)
conf_wins_contrib   = conf_tourney_weight * (conf_wins_a - conf_wins_b)
seed_contrib        = seed_gap_prior(seed_a, seed_b, seed_prior_weight)

# Heavy-tailed shock (reproduces real upset rates)
shock = t.rvs(df=shock_df, scale=shock_scale)

# Logistic with team-specific uncertainty
mu = base_margin + recent_form + luck + conf_wins + seed + shock
p  = sigmoid(mu / sigma_eff)
```

`adj_o` and `adj_d` are derived from Kaggle regular-season box scores using iterative rating — they track points per 100 possessions adjusted for opponent strength, the same concept as KenPom's AdjO/AdjD.

### Training

The training pipeline (`M_train.ipynb`) runs a two-stage optimization:

1. **Shock calibration** — grid search over `(shock_df, shock_scale)` minimizing SSE against historical seed-matchup upset rates (simulated on equal-strength teams so only the shock distribution matters).
2. **Feature weight training** — L-BFGS-B over `(margin_scale, luck_weight, recent_form_weight, w_efg, w_to, w_orb, w_ftr, seed_prior_weight)` minimizing tournament log-loss on 2010–2025 games.
3. **Temperature scaling** — scalar `tau` fitted by leave-one-year-out CV, applied as `p_cal = sigmoid(logit(p) / tau)`.

### Bracket simulation

Two modes:

| Mode | Method | Output |
|---|---|---|
| Monte Carlo | 10,000 full bracket runs | Championship probability per team |
| Round-by-round | 1,000 sims per game, advance modal winner | Most likely bracket path with per-game confidence |

---

## Trained model performance (men's, 2010–2025)

| Metric | Value |
|---|---|
| CV log-loss | 0.522 ± 0.109 |
| CV accuracy | 75.9% ± 5.8% |
| Better-seed baseline | 70.3% |
| Higher AdjEM baseline | 74.7% |

---

## Contributing

Open a pull request or file an issue. The core model lives in `marchmadness/model.py` and `marchmadness/training.py`.

## License

MIT
