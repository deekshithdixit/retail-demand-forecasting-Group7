# Retail Demand Forecasting — Group 7

## 1. Project Overview

Retail businesses need accurate demand forecasts to plan inventory, promotions, staffing, and store operations.

This project develops an end-to-end retail demand forecasting solution using the Rossmann Store Sales dataset. The solution combines DataOps, machine learning, MLOps, API deployment, containerization, and monitoring.

The pipeline is designed to:

- Ingest retail sales data
- Validate data quality
- Perform feature engineering
- Train a demand forecasting model
- Track experiments using MLflow
- Expose predictions through a FastAPI service
- Containerize the prediction service using Docker
- Expose application metrics for Prometheus monitoring
- Orchestrate the workflow using Apache Airflow

---

## 2. Business Objective

The primary objective is to estimate future store-level sales demand.

Accurate demand forecasting can support:

- Inventory planning
- Promotion planning
- Store operations
- Staffing decisions
- Resource allocation
- Reduction of overstocking and stockouts

### Target Variable

The prediction target is:

`Sales`

The `Customers` column is intentionally excluded from model features because future customer counts would not be available when making real-world future sales predictions.

---

## 3. Dataset

The project uses the Rossmann Store Sales dataset.

The dataset contains:

- `train.csv` — historical store sales
- `test.csv` — future prediction records
- `store.csv` — store-level metadata

### Original Dataset Dimensions

| Dataset | Rows | Columns |
|---|---:|---:|
| Train | 1,017,209 | 9 |
| Test | 41,088 | 8 |
| Store | 1,115 | 10 |

Important information includes:

- Store
- Date
- Day of Week
- Sales
- Open/Closed status
- Promotions
- State Holiday
- School Holiday
- Store Type
- Assortment
- Competition information
- Promo2 information

Raw datasets are not stored in GitHub because they are excluded through `.gitignore`.

---

## 4. Solution Architecture

```text
Rossmann Dataset
       |
       v
Data Ingestion
       |
       v
Data Validation
       |
       v
Feature Engineering
       |
       v
Model Training
       |
       +------> MLflow Experiment Tracking
       |
       v
Model Validation
       |
       v
Saved Model Artifact
       |
       v
FastAPI Prediction API
       |
       +------> Prometheus Metrics
       |
       v
Docker Container
Apache Airflow is designed to orchestrate the pipeline tasks.

5. DataOps Pipeline
5.1 Data Ingestion

The ingestion module loads:

Training data
Test data
Store metadata

The pipeline uses a centralized configuration module so that file paths remain consistent across the project.

5.2 Data Validation

The validation pipeline checks:

Required columns
Duplicate records
Negative sales values
Valid store IDs
Valid dates
Missing values

The validation stage prevents invalid data from progressing through the pipeline.

5.3 Feature Engineering

The following time-based features are created:

Year
Month
Day
Week of Year
Weekend indicator

Store metadata is joined with the sales data.

Historical demand features are also created:

Sales Lag 1
Sales Lag 7
Sales Lag 14
Rolling Mean 7
Rolling Mean 14

Lag and rolling features are calculated using previous observations to avoid target leakage.

6. Machine Learning
6.1 Model

The project uses:

HistGradientBoostingRegressor

The model is trained using a temporal validation strategy rather than a random split because retail demand is time-dependent.

6.2 Model Features

The model uses:

Store
Day of Week
Open
Promo
State Holiday
School Holiday
Year
Month
Day
Week of Year
Weekend indicator
Store Type
Assortment
Competition Distance
Competition Open Since Month
Competition Open Since Year
Promo2
Promo2 Since Week
Promo2 Since Year
Promo Interval
Sales Lag 1
Sales Lag 7
Sales Lag 14
Rolling Mean 7
Rolling Mean 14

Categorical features are encoded using OneHotEncoder.

The Customers column is excluded from the predictive feature set to avoid using information that would not be available for future demand prediction.

7. Model Performance

The current validation results are:

Metric	Value
MAE	542.58
RMSE	818.16
MAE

Mean Absolute Error represents the average absolute difference between predicted and actual sales.

RMSE

Root Mean Squared Error gives greater weight to larger prediction errors.

8. MLflow

MLflow is used for experiment tracking.

The project records:

Model parameters
Validation metrics
Trained model artifact
MLflow Experiment

Retail_Demand_Forecasting_Group7

The trained model artifact is stored locally as:

models/retail_demand_model.pkl

The MLflow training run recorded the current model results:

MAE  = 542.58
RMSE = 818.16
9. Prediction API

The trained model is exposed through a FastAPI application.

API Endpoints
Endpoint	Purpose
/	API information
/health	Health check
/predict	Generate sales prediction
/metrics	Prometheus metrics
Example Prediction

The API has been tested successfully with a prediction response similar to:

{
  "predicted_sales": 5656.67
}

The API also records:

Prediction request count
Prediction latency
Prediction error count
10. Docker

The prediction API is containerized using Docker.

Build the Docker Image
docker build -t retail-demand-api .
Run the Container
docker run --rm -p 8001:8000 retail-demand-api

The API can then be accessed at:

http://localhost:8001

The Docker container has been tested successfully with:

/health
/docs
/metrics
11. Monitoring

The FastAPI application exposes Prometheus-compatible metrics through:

/metrics

Tracked application metrics include:

retail_prediction_requests_total
retail_prediction_latency_seconds
retail_prediction_errors_total

These metrics provide monitoring information about API usage, prediction latency, and errors.

The local API metrics endpoint has been verified successfully.

12. Apache Airflow

The project includes an Airflow DAG:

airflow/dags/retail_demand_pipeline.py

The DAG is designed to orchestrate:

Data Ingestion
      |
      v
Data Validation
      |
      v
Feature Engineering
      |
      v
Model Training + MLflow
      |
      v
Model Artifact Validation

The DAG uses PythonOperator tasks and dependency-based execution.

Airflow Tasks
data_ingestion
data_validation
feature_engineering
model_training_mlflow
model_artifact_validation

The DAG syntax has been validated locally.

The university Airflow environment currently requires the project files to be available in its mounted Docker environment before this DAG can be executed there.

13. Project Structure
retail-demand-forecasting-Group7/
│
├── airflow/
│   └── dags/
│       └── retail_demand_pipeline.py
│
├── api/
│   └── predict.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── retail_demand_model.pkl
│
├── src/
│   ├── config.py
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── train_mlflow.py
│
├── monitoring/
│
├── tests/
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
14. Running the Project
Create the Virtual Environment
python -m venv .venv
Install Dependencies
pip install -r requirements.txt
Run Data Ingestion
python src/data_ingestion.py
Run Data Validation
python src/data_validation.py
Run Feature Engineering
python src/feature_engineering.py
Train the Model
python src/train.py
Train with MLflow Tracking
python src/train_mlflow.py
Run the FastAPI Application
uvicorn api.predict:app --host 0.0.0.0 --port 8000
15. Technologies Used
Python
Pandas
NumPy
Scikit-learn
MLflow
FastAPI
Uvicorn
Docker
Apache Airflow
Prometheus
Git
GitHub
16. MLOps Workflow
Data
  |
  v
Validation
  |
  v
Feature Engineering
  |
  v
Model Training
  |
  v
MLflow Tracking
  |
  v
Model Artifact
  |
  v
FastAPI
  |
  v
Docker
  |
  v
Prometheus Monitoring

The project follows a modular structure so that ingestion, validation, feature engineering, training, inference, and monitoring can be maintained independently.

17. Reproducibility

The project uses Git and GitHub for version control.

Initial Commit
35fe99a
Initial retail demand forecasting pipeline

Raw and processed datasets are excluded from version control through .gitignore.

The project repository is available on GitHub:

https://github.com/deekshithdixit/retail-demand-forecasting-Group7

18. Team
Group 7 — Retail Demand Forecasting

The project follows the DataOps and MLOps capstone requirements for the GUVI × Jain University program.

The team project focuses on building an end-to-end retail demand forecasting pipeline covering data ingestion, data quality, feature engineering, machine learning, experiment tracking, deployment, orchestration, and monitoring.

19. Current Project Status
Completed
Data ingestion
Data validation
Feature engineering
Demand forecasting model
Temporal model validation
Model evaluation
MLflow experiment tracking
Model artifact generation
FastAPI inference API
Docker containerization
Prometheus-compatible API metrics
Airflow DAG development
Git version control
GitHub repository
Remaining Integration Work
Execute the Group 7 DAG successfully in the assigned Airflow environment
Capture Airflow DAG execution evidence
Complete Prometheus/Grafana monitoring evidence
Finalize project documentation
Prepare final presentation
Prepare individual contribution and viva documentation