# SupeRR Selector, Season 2 Round 1

Who holds when the match tightens? This project rates T20 batters and bowlers ball by ball, 2022-26, using how much each ball
swings a match (its importance), and picks a top 10 of each from the players who have a real record under pressure.

- `superr/src/` is the modelling pipeline: win probability, per-ball value, ratings, validation and the final package.
- `run_all.py` rebuilds the package and verifies it against a frozen reference.
- `dashboard/` is the interactive web app (Vite, React, TypeScript) that shows the picks and the evidence.

## Final top 10s (floor 50, from `superr/reports/final_v2/`)

Ranked by cautious rating, in win-probability points per average ball. A player must have at least 50% of his rating from his
own matches, at least 5 matches in 2025-26 and at least 50 high-pressure balls (importance 1.5 or more) in 2022-26.

**Batters**

| Rank | Player | Cautious rating | In the top 10 in |
|---|---|---|---|
| 1 | Tim David | +0.106 | 89.6% of re-rankings |
| 2 | Suryakumar Yadav | +0.104 | 93.8% of re-rankings |
| 3 | Abhishek Sharma | +0.087 | 91.8% of re-rankings |
| 4 | Heinrich Klaasen | +0.080 | 66.0% of re-rankings |
| 5 | Phil Salt | +0.078 | 81.4% of re-rankings |
| 6 | Shreyas Iyer | +0.073 | 65.4% of re-rankings |
| 7 | Jos Buttler | +0.058 | 45.0% of re-rankings |
| 8 | Tilak Varma | +0.058 | 36.8% of re-rankings |
| 9 | Shubman Gill | +0.054 | 29.0% of re-rankings |
| 10 | Rinku Singh | +0.051 | 34.2% of re-rankings |

Alternate: Glenn Maxwell (rank 12).

**Bowlers**

| Rank | Player | Cautious rating | In the top 10 in |
|---|---|---|---|
| 1 | Jasprit Bumrah | +0.094 | 92.8% of re-rankings |
| 2 | Rashid Khan | +0.093 | 96.8% of re-rankings |
| 3 | Sunil Narine | +0.087 | 96.4% of re-rankings |
| 4 | Josh Hazlewood | +0.065 | 86.0% of re-rankings |
| 5 | Varun Chakravarthy | +0.062 | 80.4% of re-rankings |
| 6 | Kuldeep Yadav | +0.054 | 72.8% of re-rankings |
| 7 | Mohsin Khan | +0.045 | 44.4% of re-rankings |
| 8 | Wanindu Hasaranga | +0.040 | 53.2% of re-rankings |
| 9 | Axar Patel | +0.038 | 67.2% of re-rankings |
| 10 | Krunal Pandya | +0.026 | 35.0% of re-rankings |

Alternate: Bhuvneshwar Kumar (rank 11).

## Rerun the package

```
pip install -r requirements.txt
python run_all.py --floor 50            # reruns the final steps with the exposure floor; writes the *_v2 folders
python run_all.py --floor 50 --verify-only   # compares superr/reports/final_v2 with superr/reports/final_v2/reference
```

`--floor 50` starts at `phase4f_final` and needs the upstream model outputs that a full run (`python run_all.py`) produces from the raw data.
`--verify-only` needs only the files in `superr/reports/final_v2/` and its `reference/` folder.

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
