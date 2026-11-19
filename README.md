# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rudhay863/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rudhay863/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A deterministic Node 22 program that forecasts, for each of the 55 GTA VI PlayStation 5 trophies, the Exophase earn rate at 2026-11-26T00:00:00Z as five quantiles (5%, 25%, 50%, 75%, 95%). It fits per-trophy logit adoption curves against log-time-since-launch over the case's reads, decomposes each rate into earners vs players-tracked so Exophase refresh dilution is modelled rather than mistaken for falling adoption, shrinks fitted slopes toward description-based priors (story chapters climb fast, counted grinds climb slowly, platinum slowest), and prices the unseen-refresh risk into a wide, horizon-scaled spread on the logit scale. It runs fully offline on the standard library only, with no network and no model calls at runtime.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=dzgseakx&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
