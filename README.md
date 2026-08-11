# Flight Delay Classification System

An end-to-end machine-learning project that predicts whether a scheduled flight will be delayed. It covers relational data preparation, reproducible model experiments, hyperparameter tuning, experiment tracking, and a containerized prediction service.

## Project highlights

- Normalizes raw flight records into a 3NF SQLite database and loads modeling data with SQL joins.
- Compares Ridge Classifier, Histogram Gradient Boosting, XGBoost, and LightGBM pipelines.
- Evaluates preprocessing with and without PCA and tunes models with Optuna.
- Tracks parameters, metrics, and serialized models in MLflow.
- Serves the selected model through FastAPI and provides a Streamlit prediction UI.
- Packages the API and UI as separate Docker services on a private Compose network.

## Results

The committed MLflow runs contain 16 controlled baseline/tuned experiments. The highest recorded held-out F1 score is:

| Metric | Value | Run |
| --- | ---: | --- |
| Test F1 | **0.481** | `xgboost_baseline_optuna` |

This result is reported from the checked-in MLflow metric files rather than reconstructed from the training data. The dataset contains 2,201 records, so the score should be interpreted as a small-dataset experiment, not a production benchmark.

## System design

```text
Raw flight data
      │
      ▼
3NF SQLite database ──► SQL join ──► Pandas
                                      │
                                      ▼
                         preprocessing + model search
                                      │
                            MLflow ◄───┤
                                      ▼
                              serialized champion
                                      │
                         FastAPI ◄─────┴─────► Streamlit
```

The model expects five numerical features (`schedtime`, `distance`, `dayweek`, `daymonth`, `flightnumber`) and four categorical features (`weather`, `carrier`, `origin`, `dest`). The saved schema provides the valid ranges and categories used by the UI.

## Run the prediction app

Prerequisite: Docker with the Compose plugin.

```bash
docker compose up --build
```

Then open:

- Streamlit UI: <http://localhost:8501>

The UI calls FastAPI over the private Compose network. The API is intentionally not published to the host; to open its `/docs` page directly, add a host mapping for port `8000` in `docker-compose.yml` or run the API outside Compose.

The API container loads `models/global_best_flightdelays_optuna.pkl` at startup. Compose waits for its health check before starting the UI.

Stop the stack with:

```bash
docker compose down
```

## API contract

`POST /predict` accepts a list of feature dictionaries:

```json
{
  "instances": [
    {
      "schedtime": 930,
      "distance": 214,
      "weather": 0,
      "dayweek": 3,
      "daymonth": 15,
      "flightnumber": 2385,
      "carrier": "DL",
      "origin": "DCA",
      "dest": "LGA"
    }
  ]
}
```

The response includes the binary prediction, a human-readable label, and delayed-flight probability when the selected estimator supports `predict_proba`.

## Reproduce the analysis

The notebooks are intended to be run in order:

| Notebook | Purpose |
| --- | --- |
| `01_data_and_sqlite.ipynb` | Raw-data inspection and relational database preparation |
| `02_baseline_experiments.ipynb` | Baseline experiment matrix with/without PCA |
| `03_optuna_tuning.ipynb` | Tuned model comparison and diagnostics |
| `04_feature_schema_export.ipynb` | Feature-schema export for the prediction UI |

Before running locally, replace the original `base_folder`/CSV path cells with this clone's absolute path. Training dependencies are represented by the imports in the notebooks and `flightdelays_pipeline.py`. Runtime dependencies for deployment are pinned separately in `api/requirements.txt` and `streamlit/requirements.txt`.

## Repository layout

```text
.
├── api/                         # FastAPI service and runtime pipeline
├── streamlit/                   # Interactive prediction UI
├── models/                      # Selected serialized pipelines
├── mlruns/                      # Committed MLflow experiment metadata/artifacts
├── xiaowei_data/                # Source data, SQLite DB, and UI schema
├── xiaowei_notebooks/           # Data, experiment, tuning, and schema notebooks
├── docker-compose.yml
├── flightdelays_pipeline.py     # Shared preprocessing/model builders
└── SummaryFlightDelayClassificationSystem.md
```

## Limitations

- The sample is small and covers a limited set of airports, carriers, schedules, and distances.
- Random stratified validation does not measure temporal or geographic drift.
- The committed pickle files require compatible Python/library versions; the API image pins those runtime versions.
- The current Compose file exposes only Streamlit to the host and is designed for local demonstration.
