# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/plm34554a/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/plm34554a/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for GTA VI's PS5 trophy earn rates on Exophase. Given every recorded read of the game's trophy page before a checkpoint, it predicts, for each of the 55 trophies, five quantiles (5/25/50/75/95%) of the earn rate one week after release, scored by pinball loss on the logit scale.

The model works per trophy in logit space against ln(1 + hours since release). It fits only the last 18 hours of reads — older reads describe a launch regime of early adopters and a barely-refreshed page that no longer drives the rates — with a weighted least-squares line whose weights grow with the players tracked and with recency. Each trophy's slope is shrunk toward a pooled slope for its tier (platinum/gold/silver/bronze) blended with the global pool, by an information weight based on the x-span covered and the fit's residual scatter, so noisy or flat trophies borrow the field's trajectory. The median is anchored at the trophy's recent level and carried forward along the shrunk slope with the move damped by 0.85. The spread combines a refresh floor, a sqrt-horizon random-walk term, a share of the trophy's residual scatter, and the slope's standard error projected over the extrapolation distance.

Because no real future data exists before the deadline, the model was tuned on synthetic launch-week worlds (`dev/backtest.py`): simulated player growth, per-trophy adoption curves, Exophase-style page refreshes and count-quantised printing, cut at the real checkpoint and target. Across train, held-out and shock-perturbed world sets it scores mean pinball losses of about 0.265 / 0.265 / 0.274, against about 0.319 / 0.342 / 0.326 for a logit-linear curve fit and 0.558 / 0.551 / 0.500 for last-rate persistence (the two house baselines' shapes).

Pure Python 3 standard library, deterministic, runs in well under a second per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

(with `input/` holding a case file such as the practice cases `p00.in` / `p01.in`; in the grader's sandbox the program reads `/input` and writes `/output` with no environment set.)

---
<sub>Entered on [make.fun](https://make.fun/?ref=9321azde&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
