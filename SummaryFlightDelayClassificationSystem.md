Flight Delay Classification System (End-to-End ML Project)

Python, SQL, scikit-learn, XGBoost, LightGBM, MLflow, Optuna, FastAPI, Streamlit, Docker

Designed and implemented an end-to-end flight delay classification system, covering data modeling, feature engineering, model training, evaluation, and deployment.

Built a 3NF normalized relational database (SQLite) from raw flight records and wrote SQL JOIN queries to load data into Pandas for analysis.

Developed scikit-learn pipelines integrating imputation, scaling, one-hot encoding, PCA, and multiple classifiers (Ridge, Gradient Boosting, XGBoost, LightGBM).

Conducted 16 controlled experiments (with/without PCA, with/without hyperparameter tuning) using 3-fold stratified cross-validation, optimizing F1-score.

Applied Optuna (TPE sampler) for hyperparameter optimization and tracked experiments, metrics, and best models using MLflow.

Performed EDA and diagnostics, including correlation analysis and confusion matrix evaluation, to assess feature relevance and model performance.

Deployed the best-performing model via a FastAPI REST service and built an interactive Streamlit web app for real-time prediction.

Containerized the entire system using Docker and docker-compose for reproducible local and cloud deployment.