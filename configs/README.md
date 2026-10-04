# Configurations

Settings live next to the code that uses them so a run record can quote them exactly:

| Module | Settings | Command |
|---|---|---|
| Coverage v1 benchmark | `src/gridiron_lens/coverage/train.py`, `features.py` | `bin/coverage-lens train` |
| Coverage experiment A (relational features, class weights) | `coverage/experiment_a.py` (`FOLDS`, `WEIGHTS`, `GBM`) | `bin/coverage-lens exp-a` |
| Coverage experiment B (temporal model) | `coverage/experiment_b.py` (`CONFIGS`, `BASE`, `SEEDS`) | `bin/coverage-lens exp-b` |
| Coverage v2 comparison | `coverage/benchmark_v2.py` | `bin/coverage-lens benchmark-v2` |
| Game predictor v2 | `pregame/model_v2.py` (`LAMBDAS`, `ALPHAS`, `GROUP_SETS`), `features_v2.py` | `bin/pregame-lens backtest`, `bin/pregame-lens forecast` |
| Highlights ranking | `highlights/models.py`, `pipeline.py` | `bin/highlights-lens run` |

Every run writes a record to `reports/v2/runs/<run_id>.json` with the settings it actually used.
