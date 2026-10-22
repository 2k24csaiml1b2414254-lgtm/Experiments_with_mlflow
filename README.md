# MLflow Experiment Tracking — Hands-on Practice

A hands-on repository exploring **MLflow experiment tracking, model logging, autologging, remote tracking with DagsHub, and hyperparameter tuning** using Scikit-learn.

This project demonstrates how machine-learning experiments can be recorded, evaluated, and compared using MLflow. It includes practical examples with the Wine Classification and Breast Cancer datasets.

## Overview

During model development, different experiments may use different hyperparameters, datasets, and evaluation methods. Keeping track of these experiments makes it easier to compare results and reproduce previous work.

This repository explores MLflow's tracking capabilities through four Python scripts, progressing from basic manual logging to remote tracking and hyperparameter optimization.

## Features

- **Experiment Tracking:** Log model parameters, evaluation metrics, tags, and artifacts.
- **Model Logging:** Save trained Scikit-learn models as MLflow model artifacts.
- **Confusion Matrix:** Generate and log a confusion matrix for the Wine Classification experiment.
- **Remote Tracking:** Configure DagsHub as a remote MLflow tracking destination.
- **Autologging:** Explore automatic logging with `mlflow.autolog()`.
- **Hyperparameter Tuning:** Use `GridSearchCV` to evaluate Random Forest configurations.
- **Nested Runs:** Record individual hyperparameter candidates as nested MLflow runs.
- **Dataset Tracking:** Log training and testing dataset inputs with MLflow.
- **Experiment Visualization:** Explore runs and their recorded information through the MLflow UI.

## Tech Stack

- Python
- MLflow
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- DagsHub
- SQLite / MLflow local tracking
- Git and GitHub

## Repository Structure

```text
Experiments_with_mlflow/
│
├── src/
│   ├── file1.py
│   ├── file2_remote_setup.py
│   ├── file3_autolog.py
│   └── file4_hypertuning.py
│
├── mlruns/                       # Local experiment data, if generated
├── mlartifacts/                  # Local artifacts, if generated
├── mlflow.db                     # Local tracking database, if used
│
├── Confusion matrix.png
├── MLflow_Hands_On_Revision_Notes.txt
├── ml_flow_log.txt
├── mlflow_autolog.txt
├── mlflow_server_architec.jpg
├── mlflow_working.jpeg
├── .gitignore
├── LICENSE
└── README.md
```

The MLflow data directories and database shown above are generated tracking data. Their exact role depends on the tracking backend used for each experiment.

## Getting Started

### 1. Clone the repository

```powershell
git clone https://github.com/2k24csaiml1b2414254-lgtm/Experiments_with_mlflow.git

cd Experiments_with_mlflow
```

### 2. Check Python

```powershell
python --version
```

### 3. Install the dependencies

The repository currently uses the following packages:

```powershell
python -m pip install mlflow scikit-learn matplotlib seaborn pandas dagshub
```

Check the installed MLflow version and package dependencies:

```powershell
python -m mlflow --version
python -m pip check
```

**Note:** The commands above install the required libraries, but they do not pin exact versions. For more reproducible results, a dependency file with tested package versions can be added later.

## Experiments

### 1. Manual Experiment Tracking — `file1.py`

Uses the Scikit-learn Wine dataset and a Random Forest classifier.

The script demonstrates:

- Loading the dataset and splitting it into training and testing sets.
- Training a Random Forest model.
- Evaluating classification accuracy.
- Logging `max_depth` and `n_estimators` as parameters.
- Logging accuracy as a metric.
- Generating and logging a confusion matrix.
- Recording run tags.
- Logging the trained model and source script.

Run:

```powershell
python src/file1.py
```

This script is configured to send tracking requests to `http://127.0.0.1:5000`.

### 2. Remote Experiment Tracking — `file2_remote_setup.py`

Demonstrates configuring DagsHub for remote MLflow tracking.

The script:

- Initializes the DagsHub repository integration.
- Configures a remote MLflow tracking URI.
- Trains a Random Forest classifier on the Wine dataset.
- Logs parameters, accuracy, a confusion matrix artifact, source code, tags, and the trained model.

Run:

```powershell
python src/file2_remote_setup.py
```

Before running, configure DagsHub authentication and repository write access as required by your environment.

**Security:** Never commit access tokens, passwords, or other credentials to GitHub. Use an appropriate environment variable or authentication mechanism.

### 3. Automatic Logging — `file3_autolog.py`

Demonstrates MLflow autologging with Scikit-learn.

The script enables:

```python
mlflow.autolog()
```

It then trains a Random Forest classifier and generates a confusion matrix.

The example explores automatic logging of supported model parameters, metrics, and model-related information, alongside manually recorded tags and source code.

Run:

```powershell
python src/file3_autolog.py
```

The script is configured to use `http://127.0.0.1:5000` for tracking.

### 4. Hyperparameter Tuning — `file4_hypertuning.py`

Uses the Breast Cancer Wisconsin dataset and `GridSearchCV` to evaluate Random Forest configurations.

The parameter grid is:

```python
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [None, 10, 20, 30]
}
```

This produces **12 hyperparameter combinations**. With five-fold cross-validation, GridSearchCV performs 60 candidate-fold model fits.

The script demonstrates:

- Stratified train/test splitting.
- Five-fold cross-validation.
- Best-parameter and best-estimator selection.
- Nested MLflow runs for candidate configurations.
- Logging best cross-validation accuracy and held-out test accuracy separately.
- Logging training and testing dataset inputs.
- Logging the source script, model, and experiment tags.

Run:

```powershell
python src/file4_hypertuning.py
```

The script does not explicitly set a tracking URI, so its tracking destination depends on the active MLflow configuration and version. Configure the UI to use the same tracking backend when inspecting these runs.

## Running the MLflow UI

To start the local MLflow UI:

```powershell
python -m mlflow ui --port 5000
```

Open the following address in your browser:

**http://127.0.0.1:5000**

Keep the terminal running while using the UI. Stop the server with `Ctrl + C`.

For scripts that use `http://127.0.0.1:5000`, ensure that the local tracking server is available at that address.

If using SQLite as the backend, you can start the UI with:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

The script and UI must use the same tracking backend to display the expected runs. Local file-store tracking, SQLite tracking, and DagsHub remote tracking may contain separate experiment histories.

## What I Learned

Through these hands-on exercises, I explored:

- The difference between experiments and runs.
- Logging model parameters, metrics, tags, artifacts, and models.
- Using MLflow's tracking UI.
- Comparing manual logging with autologging.
- Configuring remote tracking through DagsHub.
- Applying GridSearchCV for hyperparameter optimization.
- Recording nested runs and dataset inputs.
- Distinguishing cross-validation performance from held-out test performance.
- Troubleshooting tracking configuration and model-serialization issues.

## Future Improvements

- Add a tested `requirements.txt` or another reproducible dependency specification.
- Document verified experiment results and add screenshots of the MLflow UI.
- Compare candidate models using additional classification metrics such as precision, recall, and F1-score.
- Improve consistency between tracking URI configuration and UI startup instructions.
- Add clearer experiment names and run names for easier comparison.
- Explore model versioning, production monitoring, and model/data drift detection.

## Author

**Himesh Mishra**

B.Tech — Computer Science Engineering (AI/ML)

Interested in Applied AI, Machine Learning, and MLOps.

---

This repository is a learning-focused demonstration of MLflow tracking and experimentation. Results and tracking behavior may vary with package versions and backend configuration.
