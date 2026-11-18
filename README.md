# SupeRR Selector, Season 2 Round 1

Who holds when the match tightens? This project rates T20 batters and bowlers ball by ball, 2022-26, using how much each ball
swings a match (its importance), and picks a top 10 of each from the players who have a real record under pressure.

- `superr/src/` is the modelling pipeline: win probability, per-ball value, ratings, validation and the final package.
- `run_all.py` rebuilds the package and verifies it against a frozen reference.
- `dashboard/` is the interactive web app (Vite, React, TypeScript) that shows the picks and the evidence.

## Final top 10s (v3: floor 50 and a pressure value above average, from `superr/reports/final_v3/`)

Ranked by cautious rating, in win-probability points per average ball. A player must have at least 50% of his rating from his
own matches, at least 5 matches in 2025-26, at least 50 high-pressure balls (importance 1.5 or more) in 2022-26, and a
pressure value above average (value on those high-pressure balls above zero). The last rule is post-hoc: it was decided on 18 November,
after the v2 list was reviewed against the problem statement, in the same way as the 50-ball floor. The ratings and the order are unchanged.
The v2 lists remain in `superr/reports/final_v2/`.

**Batters**

| Rank | Player | Cautious rating | In the top 10 in | Pressure value | Gap |
|---|---|---|---|---|---|
| 1 | Tim David | +0.106 | 93.4% of re-rankings | +0.334 | +0.019 |
| 2 | Suryakumar Yadav | +0.104 | 96.0% of re-rankings | +0.136 | -0.129 |
| 3 | Abhishek Sharma | +0.087 | 95.2% of re-rankings | +0.094 | -0.150 |
| 4 | Heinrich Klaasen | +0.080 | 78.0% of re-rankings | +0.392 | +0.217 |
| 5 | Shreyas Iyer | +0.073 | 76.6% of re-rankings | +0.673 | +0.617 |
| 6 | Jos Buttler | +0.058 | 61.0% of re-rankings | +0.377 | +0.302 |
| 7 | Tilak Varma | +0.058 | 52.2% of re-rankings | +0.080 | -0.030 |
| 8 | Rinku Singh | +0.051 | 44.0% of re-rankings | +0.559 | +0.592 |
| 9 | Reeza Hendricks | +0.050 | 29.8% of re-rankings | +0.437 | +0.346 |
| 10 | Glenn Maxwell | +0.049 | 55.2% of re-rankings | +0.750 | +0.709 |

Alternates: Nicholas Pooran (rank 11), Virat Kohli (rank 12).

**Bowlers**

| Rank | Player | Cautious rating | In the top 10 in | Pressure value | Gap |
|---|---|---|---|---|---|
| 1 | Jasprit Bumrah | +0.094 | 94.4% of re-rankings | +0.196 | -0.068 |
| 2 | Rashid Khan | +0.093 | 97.4% of re-rankings | +0.555 | +0.346 |
| 3 | Sunil Narine | +0.087 | 97.6% of re-rankings | +0.709 | +0.499 |
| 4 | Josh Hazlewood | +0.065 | 89.0% of re-rankings | +0.492 | +0.278 |
| 5 | Varun Chakravarthy | +0.062 | 85.0% of re-rankings | +0.274 | +0.087 |
| 6 | Kuldeep Yadav | +0.054 | 76.8% of re-rankings | +0.093 | -0.116 |
| 7 | Mohsin Khan | +0.045 | 48.6% of re-rankings | +0.592 | +0.412 |
| 8 | Wanindu Hasaranga | +0.040 | 63.2% of re-rankings | +0.222 | +0.075 |
| 9 | Axar Patel | +0.038 | 74.8% of re-rankings | +0.191 | +0.041 |
| 10 | Krunal Pandya | +0.026 | 42.0% of re-rankings | +0.348 | +0.174 |

Alternates: Mujeeb Ur Rahman (rank 11), Mustafizur Rahman (rank 12).

## Rerun the package

```
pip install -r requirements.txt
python run_all.py --floor 50            # reruns the final steps with the exposure floor; writes the *_v2 folders
python run_all.py --floor 50 --verify-only   # compares superr/reports/final_v2 with superr/reports/final_v2/reference
python run_all.py --floor 50 --pressure-positive   # v3: adds the pressure rule to the floor-50 pool; writes out/model_v3 and reports/final_v3
python run_all.py --floor 50 --pressure-positive --verify-only
```

`--floor 50` starts at `phase4f_final` and needs the upstream model outputs that a full run (`python run_all.py`) produces from the raw data.
`--verify-only` needs only the files in `superr/reports/final_v2/` (or `final_v3/` with `--pressure-positive`) and its `reference/` folder.
`--pressure-positive` reads the frozen v2 outputs only and never writes to the v1 or v2 folders.

## Build the dashboard

```
cd dashboard
npm install
npm run data     # builds public/data/*.json from the package and the ball data, then checks every headline number
npm run build    # static site in dist/
```

## Data

The raw data (the supplied data pack and Cricsheet files), the model outputs and the written report documents are not included in this repository.

The written report is prepared separately.
