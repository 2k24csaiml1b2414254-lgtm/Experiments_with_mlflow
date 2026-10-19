from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import pandas as pd
import mlflow
import mlflow.sklearn
import os

# Disable skops serialization if supported by this MLflow version.
os.environ["MLFLOW_SKLEARN_SUPPORT_SKOPS"] = "false"

# 1. Load the dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# 4. Define the hyperparameter grid
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [None, 10, 20, 30]
}

# 5. Configure GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="accuracy"
)

# 6. Set up MLflow experiment
mlflow.set_experiment("breast-cancer-rf-hp")

with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning"):

    # Run hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Get the best model and CV results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # Evaluate on held-out test data
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    # 7. Log parameters
    mlflow.log_params(best_params)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring", "accuracy")
    mlflow.log_param("candidate_combinations", len(grid_search.cv_results_["params"]))

    # 8. Log metrics
    mlflow.log_metric("best_cv_accuracy", best_cv_score)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # 9. Log training and testing datasets as MLflow inputs
    train_df = X_train.copy()
    train_df["target"] = y_train

    test_df = X_test.copy()
    test_df["target"] = y_test

    train_dataset = mlflow.data.from_pandas(
        train_df,
        name="breast_cancer_training"
    )
    test_dataset = mlflow.data.from_pandas(
        test_df,
        name="breast_cancer_testing"
    )

    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(test_dataset, context="testing")

    # 10. Log the source code
    mlflow.log_artifact(__file__)

    # 11. Log the best model
    # The environment variable above is intended to disable skops;
    # if this MLflow version still uses skops, see the note below.
    mlflow.sklearn.log_model(
    sk_model=best_model,
    name="random_forest",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
  )

    # 12. Set tags
    mlflow.set_tag("author", "Himesh Mishra")
    mlflow.set_tag("model_type", "RandomForestClassifier")
    mlflow.set_tag("dataset", "Breast Cancer Wisconsin")

    # 13. Display results
    print("\nBest Parameters:", best_params)
    print("Best Cross-Validation Accuracy:", best_cv_score)
    print("Test Accuracy:", test_accuracy)

    print("\nMLflow run completed.")
