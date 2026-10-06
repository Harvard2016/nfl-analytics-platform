# Model card: game prediction, forecast protocol v3

**Model version:** `pregame-v2-frozen` (Elo-offset logistic and ridge point margin with the hyperparameters selected in v2 on 2012–2022). **Protocol:** `forecast-protocol-v3`.

| | |
|---|---|
| Target | Home team wins (ties excluded from scoring) |
| Cutoff | 24 hours before the scheduled kickoff, in UTC, converted from the published US-Eastern time |
| Inputs | Elo, rest, recent form, rolling offensive and defensive efficiency and rates, quarterback form for each team's previous primary passer |
| Prior-game rule | A finished game counts only if it kicked off at least 28 hours before the target kickoff. This is a proxy for "published"; it is not proof |
| Update policy | Refit on every decided game in the source snapshot before each run; hyperparameters never change |
| Not used | The target game's score, box score or starter; injuries; weather; betting lines; anything from the coverage or highlight systems |

## What every v3 record carries

Game, kickoff and cutoff in UTC and Eastern; the creation time read after fitting; timing class (`before_cutoff`, `late`, `after_kickoff`, `ineligible`); source snapshot id, time and content hash with per-file hashes; model bundle hash; pipeline hash; training cohort hash and through-date; full-precision probabilities for Elo, Elo-offset, margin and blend; feature values; quarterback assumption; log-odds terms. Outcomes are attached beside the record at publish time, never written into it.

Official scored record per game: the latest record created, and sourced, at or before the cutoff for the kickoff actually played. Rules are in every forecast file under `policy`.

## Evidence

- 2023–2025 (previously examined): Elo 0.6362, Elo-offset 0.6349, margin 0.6319 log loss; paired intervals against Elo include zero. **No established gain over Elo.**
- v3 development experiments (`reports/v3/pregame_experiments_v3.json`): opponent-adjusted ratings beat Elo on nested development seasons but are indistinguishable from the frozen v2 model; not promoted.
- Prospective 2026 record: `apps/web/public/demo/pregame/forecasts.json`. Too few games to separate models.

## Limits

Historical features come from today's revised nflverse files; nflverse EPA models were fitted with later data. The projected quarterback is the previous game's primary passer, not a confirmed starter. Legacy records from before protocol v3 hashed only the schedule file and are never scored. Not betting advice.

## Commands

`bin/pregame-lens snapshot` · `bin/pregame-lens forecast-v3` · `bin/pregame-lens publish-forecasts` · scheduler: `ops/com.gridironlens.forecast.plist`.
