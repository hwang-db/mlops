# Databricks notebook source
# MAGIC %md ######Use online downloaded diabetes dataset which contains features from diabetic patients such as their bmi, age, blood pressure and glucose levels to predict the diabetes disease progression in patients.

# COMMAND ----------

import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# COMMAND ----------

# MAGIC %md Read data

# COMMAND ----------

def read_data():
    db = load_diabetes()
    X = db.data
    y = db.target
    return X, y

# COMMAND ----------

X, y = read_data()

# COMMAND ----------

print("X: ", X[0])
print("y: ", y[0])

# COMMAND ----------

# MAGIC %md Train MLflow model

# COMMAND ----------

client = MlflowClient()
experiment_name = "/Users/hao.wang@databricks.com/mlflow_aml_demo/" + "experiment_%s" % int(time.time())
experiment_id = client.create_experiment(experiment_name)
print(experiment_id)

# COMMAND ----------

def mlflow_run(params, run_name, experiment_id):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        # Create our model type instance and some random fake regression data
        sk_learn_rfr = RandomForestRegressor(**params)

        # typicaly here you would train and evaluation your model
        sk_learn_rfr.fit(X_train, y_train)
        y_pred =  sk_learn_rfr.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Log params and metrics using the MLflow APIs
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=sk_learn_rfr,
            artifact_path="sklearn-model")

# COMMAND ----------

def train(experiment_id):
    max_depth = 0
    for i in range (1, 5):
        n_estimators = 10 * i
        max_depth = max_depth + 2
        params = {"n_estimators": n_estimators, "max_depth": max_depth}
        mlflow_run(params, "MLflow run %s" % i, experiment_id)

# COMMAND ----------

train(experiment_id)

# COMMAND ----------

# MAGIC %md Search best MLflow model

# COMMAND ----------

def search(client, experiment_id):
    run = client.search_runs(
                    experiment_ids=experiment_id,
                    filter_string="",
                    order_by=["metrics.rmse ASC"]
                    )[0]
    return run.info.run_id

# COMMAND ----------

best_run_id = search(client, experiment_id)

# COMMAND ----------

print(best_run_id)

# COMMAND ----------

# MAGIC %md Register MLflow model

# COMMAND ----------

artifact_path = "sklearn-model"

def register(client, run_id, artifact_path, model_name):
    # register
    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path="sklearn-model")
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    # wait until ready
    model_name, model_version = model_details.name, model_details.version
    status = ModelVersionStatus.PENDING_REGISTRATION
    while status != ModelVersionStatus.READY:
        model_version_details = client.get_model_version(
            name=model_name,
            version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
    
    # transit to production
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage='Production',
    )

# COMMAND ----------

model_name = "ml_simple_demo_model"
register(client, best_run_id, artifact_path, model_name)

# COMMAND ----------


