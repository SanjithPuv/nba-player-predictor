# NBA Next Game Predictor

A runnable Python research project forecasting **player points conditional on playing**. It separates structured data, numerical forecasting, and evidence-grounded explanations. No LLM key or GPU is needed for the core application.

**Read [the execution report](docs/EXECUTION_RESULTS.md) for measured results and remaining work.** Synthetic results are demonstrations only. Historical NBA backtests use conservative date proxies, not fully verified historical two-hour pregame snapshots. The schedule endpoint returned HTTP 403 during implementation; live prediction needs a permitted timestamped schedule feed or import.

## Setup and offline run

Tested with Python 3.14 on Apple Silicon. Use Python 3.14 for the supplied exact lock; the source targets 3.11+ but other dependency combinations were not tested. Run from this repository directory:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps .
nba-predictor --root data/synthetic --artifacts artifacts/synthetic --reports reports/synthetic demo
pytest -q
ruff check .
streamlit run app.py
```

The demo generates a reproducible artificial league, stores Parquet/DuckDB, builds strictly lagged features, trains/selects models, calibrates intervals on a separate period, tests later games, and writes a synthetic upcoming forecast with source evidence. Small static artificial examples are also checked into `fixtures/`; the larger demo is generated locally, not downloaded. Never compare its metrics with real NBA performance.

During development, use `PYTHONPATH=src python -m nba_predictor.cli ...` to run the latest source without reinstalling. This workspace's development environment is one directory above the repository (`../.venv`); a fresh clone should use the setup above.

## Real NBA workflow

Default seasons: 2021-22 through 2025-26, plus available 2026-27 regular-season games. Change `config/default.json` or supply `ingest --seasons`. The season list is a saved experiment choice, not automatically rolling.

```bash
nba-predictor --reports reports/real ingest
nba-predictor --reports reports/real features
nba-predictor --reports reports/real train
nba-predictor --reports reports/real calibrate
nba-predictor --reports reports/real backtest --experiments --thresholds 10.5 20.5 30.5
nba-predictor schedule
```

Successful player/team season requests are reused after failures; repeat ingestion to resume. For new current-season games, run `ingest --refresh` (specify only current season if desired, then re-ingest all seasons from cache to rebuild the full clean table). `--refresh` creates immutable new raw snapshots. Failed seasons are recorded and not silently fabricated. A partial ingestion builds only successful seasons, so review `reports/real/ingestion.json` before training. Requests have a configurable interval, timeout and bounded exponential retries. NBA data remains outside Git.

`nba-predictor schedule` was blocked here with HTTP 403. To use authorized schedule records:

```bash
nba-predictor import-schedule /absolute/path/to/schedule.json
```

Expected JSON is an array of records with `game_id`, `tipoff`, `home_team_id`, `away_team_id`, `source_url`, `published_at` (nullable), `retrieved_at`, `available_at`, and `timestamp_quality`. All times are ISO-8601 UTC. `available_at` must be at least `retrieved_at`. See the synthetic schema in `fixtures/synthetic_schedule.json`, which is artificial and unsuitable for real forecasts.

The interface forecasts at the two-hour pregame cutoff, once that time has occurred. It uses only schedule snapshots already observed by cutoff. It reports no forecast if those conditions are not met. A freshly retrieved schedule inside the two-hour window cannot reconstruct the earlier snapshot. Capture schedules ahead of time.

```bash
nba-predictor predict --player-id 2544 --threshold 24.5
```

To fetch a single date through the NBA stats endpoint when the full-season CDN feed is denied:

```bash
nba-predictor schedule --date 2026-11-06
nba-predictor predict --player-id 1628369 --cutoff-now
```

`--cutoff-now` explicitly uses the current observed pregame time (or `--as-of`) instead of the fixed two-hour cutoff. Its output is labeled `explicit_pregame_as_of`; future cutoffs, post-tipoff requests, late schedule snapshots and checkpoints created after cutoff are still rejected. Custom-cutoff forecasts reuse historical date-proxy calibration and have not been separately backtested. The two-hour default remains unchanged. This mode is useful when a schedule is first captured after the two-hour mark, or when requesting an earlier preview before that mark.

The player ID above is an input example, not a claimed upcoming prediction. Add `--game-id` to select an available scheduled game. Live team membership is inferred from the last available box score; verify trades. Historical experiments use `backtest`; live `predict` rejects historical cutoffs before actual model execution. The synthetic demo intentionally simulates its own fictional clock.

## Architecture and outputs

```text
nba_api P/T logs ──> immutable raw Parquet ──> clean Parquet + DuckDB
                                               │
                         cutoff-aware lagged features
                                               │
          training ──> selection validation ──> calibration ──> frozen test
                                               │
            selected mean model + quantile trees + conformal expansion
                                               │
      C++ or Python evidence retrieval ──> optional LLM explanation
                                               │
                                 CLI JSON / Streamlit
```

Forecast JSON contains expected and median points, 65/80/90% prediction intervals, threshold exceedance probability, estimated minutes, separate unknown availability probability, assumptions, current prediction timestamp, historical data cutoff, descriptive factors, citations and missing information. Intervals describe future game outcomes; they are not parameter confidence intervals. A 65% interval target is distinct from the goal of 65% of point forecasts landing within ±5 points. Both are measured, and failures are preserved.

The selected mean model competes against season/recent means, Ridge, boosted trees and a minutes/rate model with paired residual uncertainty. Separate quantile models supply conditional quantiles. A chronological conformal holdout expands intervals. Threshold probabilities use a documented approximate CDF; Brier score and reliability must be inspected separately. Independent mean/quantile models are not constrained to a single coherent distribution.

Reports include MAE, RMSE, within ±3/5/7, three coverages and widths, interval scores, Brier scores, date-block bootstrap intervals, outcome groups, predeclared feature ablations, earlier expanding-window backtests, exclusions and reliability plots. Player-level prediction Parquet files are saved beside evaluation reports but ignored by Git; only aggregate real-data metrics are intended for version control. Final test IDs are frozen before selection; do not tune after reading final results.

## Retrieval and optional LLM

### C++ evidence retrieval

The native C++17 engine queries the same DuckDB database and report Parquet file as
the Python backend. It applies player, team, opponent, season, and cutoff filters,
then ranks eligible reports in C++ using fixed token hashing and cosine similarity.
Results include source IDs, timestamps, and evidence text.

Build on macOS or Linux with a C++17 compiler, Make, curl, and tar. The dependency
step verifies the checksum of the [DuckDB 2.0.0 release](https://github.com/duckdb/duckdb/releases/tag/v2.0.0).
You can also point `DUCKDB_DIR` at an extracted distribution containing the
DuckDB C header and shared library.

```bash
make native-deps
make native
export PYTHONPATH="$PWD/src"
export NBA_RETRIEVAL_BINARY="$PWD/build/native/nba-retrieve"
make check-native
make demo
nba-predictor --root data/synthetic index fixtures/synthetic_reports.json
```

The environment variable selects C++ retrieval for the CLI and Streamlit app.
Python is the default backend. Model training and forecast scoring run in Python.
The native parity test compares source IDs, evidence values, and ranking scores
against Python on fixtures with accented names, exact cutoff boundaries, and
a report retrieved one nanosecond too late. The standalone command can query
the demo data directly:

```bash
build/native/nba-retrieve data/synthetic/nba.duckdb data/synthetic/text_index.parquet \
  10 1 2 2023-24 2023-10-30T21:00:00Z "minutes injury role" 5
```

### Indexing and explanation

```bash
nba-predictor --root data/synthetic index fixtures/synthetic_reports.json
nba-predictor injury 'https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-02-18_06_15AM.pdf' --published-at '2026-02-18T11:15:00Z'
```

The injury command archives an official PDF and extracts text. **Publication time is user-supplied and must be verified against the PDF.** It does not claim reliable injury entity parsing. Add stable player/team/opponent IDs and season to the extracted JSON before indexing it with `nba-predictor index file.json`. This implementation does not use injury documents as numerical features. It discloses that information as missing instead of inventing adjustments.

Historical text retrieval checks publication, actual retrieval and availability timestamps before ranking; unknown publication excludes text. Entity and season filtering happens before similarity search. A stateless hashing vectorizer avoids fitting future vocabulary or loading future embedding checkpoints. Numerical facts are SQL retrieved, with the same conservative retrospective historical availability caveat as numerical features. Results retain source IDs, URLs, timestamps and excerpts.

The default explanation is a deterministic evidence renderer, **not a completed LLM run**. The provider-independent `LLMProvider.generate(prompt)` interface supports an OpenAI-compatible local/hosted server (`--llm`) and local Transformers (`TransformersLLM`). For a local Ollama server, separately install/start Ollama and pull a licensed model, then export the variables from `.env.example`; the application does not automatically load `.env` files. `predict --llm` validates output and falls back safely on invalid responses. Its optional endpoint can be unavailable without blocking numerical forecasts.

LLM output must preserve the forecast dictionary exactly, match the schema, cite retrieved IDs, and use verbatim evidence excerpts for claims. Unknown facts remain missing. Retrieved text is untrusted; no tools or code execute from it. Fine-tuning targets this evidence-extraction behavior, not basketball forecasting. Base model and adapter timestamps must predate a genuinely prospective evaluation; an LLM used today cannot establish historical knowledge restrictions merely because its retrieval is filtered.

## Optional LoRA and comparison

See [the fine-tuning guide](docs/FINETUNING.md). Training code is implemented; **no LoRA training or base-versus-adapter experiment has been completed**. Commands are executable after installing optional dependencies and supplying valid disjoint data:

```bash
python -m pip install '.[llm]'
nba-predictor --root data/synthetic --artifacts artifacts/synthetic llm-examples --output data/llm/train.jsonl
nba-predictor finetune data/llm/train.jsonl --output artifacts/lora
nba-predictor llm-evaluate data/llm/evaluation_bundles.json --adapter artifacts/lora
```

Alternatively pass archived training bundles to `llm-examples path/to/bundles.json`. Evaluation bundles must have disjoint game IDs and later cutoffs. Do not publish weights or imply fine-tuning improved numerical accuracy. `config/lora.json` starts with Qwen2.5-0.5B-Instruct (Apache-2.0 as declared by its model card); pin the exact base commit for reproducible runs. Optional ML dependencies are separate from the core lock; resolve and save a hardware-specific environment for an actual training experiment.

## Repository guide

- `src/nba_predictor/`: providers, feature construction, modeling, calibration, evaluation, retrieval, explanation, prediction and CLI.
- `app.py`: player/game selection, forecasts, source evidence, history, freshness and evaluation.
- `config/`: exact experiment defaults and optional LoRA configuration.
- `fixtures/`: explicitly artificial logs/schedules/text.
- `tests/`: no-peeking, DNP, game/date splits, retrieval, quantiles, output preservation and tuning-set separation.
- `reports/`: measured aggregate results, ingestion status and clearly synthetic predictions.
- [Data dictionary](docs/DATA_DICTIONARY.md), [model card](docs/MODEL_CARD.md), [data rights](docs/DATA_RIGHTS.md), [plain-language interview guide](docs/INTERVIEW_GUIDE.md).

The source code uses the MIT license. Sports data and model weights have separate terms. No remote repository has been created or published.
