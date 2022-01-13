# Databricks notebook source
# MAGIC %pip install azureml-core
# MAGIC %pip install azure-cli-core

# COMMAND ----------

import time

import mlflow
from mlflow.tracking.client import MlflowClient

from azureml.core import Workspace, Environment
from azureml.core.compute import ComputeTarget
from azureml.core.webservice import AksWebservice
from azureml.core.model import Model, InferenceConfig

# COMMAND ----------

import mlflow
import mlflow.azureml
import azureml.mlflow
import azureml.core

from azureml.core import Workspace

subscription_id = '3f2e4d32-8e8d-46d6-82bc-5bb8d962328b'
# Azure Machine Learning resource group NOT the managed resource group
resource_group = 'hwang-demo' 
#Azure Machine Learning workspace name, NOT Azure Databricks workspace
workspace_name = 'amltestworkspace'  
# Instantiate Azure Machine Learning workspace
ws = Workspace.get(name=workspace_name,
                   subscription_id=subscription_id,
                   resource_group=resource_group)
# aml uri
uri = ws.get_mlflow_tracking_uri()
#Set MLflow experiment. 
#experimentName = "/Users/{user_name}/{experiment_folder}/{experiment_name}"
#mlflow.set_experiment(experimentName)
#mlflow.set_tracking_uri(uri)

# COMMAND ----------

# MAGIC %md Create/load AzureML workspace

# COMMAND ----------

#ws = Workspace(subscription_id = subscription_id, 
#            resource_group = resource_group, 
#            workspace_name = workspace_name)

requirements_script = "dbfs:/tmp/env.yml"
dbutils.fs.put(requirements_script,""" 
name: test-env
dependencies:
- python=3.8.10
- pip
- pip:
  - azureml-defaults
  - sklearn
  - pandas
  - numpy
  - pickle5
  - mlflow""", overwrite = True)

env = Environment.from_conda_specification("deploy_env", "/dbfs/tmp/env.yml")
env.register(ws)

# COMMAND ----------

# MAGIC %md Load MLflow model into AML

# COMMAND ----------

ts = int(time.time())

client = MlflowClient()

model_name="ml_simple_demo_model"
model_production_uri = "models:/{model_name}/1".format(model_name=model_name)

my_model = mlflow.pyfunc.load_model(model_production_uri)

temp_model_dir = "/dbfs/tmp/%s/" % ts
temp_model_path = temp_model_dir + "model.pkl"

mlflow.sklearn.save_model(my_model, temp_model_dir)

model = Model.register(
  model_path = temp_model_path,
  model_name = "ml_simple_demo_model",
  workspace = ws)

# COMMAND ----------

# MAGIC %md Create prediction script

# COMMAND ----------

entry_script_name = "dbfs:/tmp/shenlin/model_score.py"
dbutils.fs.put(entry_script_name, """
import json
import pandas as pd
import os
import pickle5 as pickle
import mlflow

def init():
  global model
  model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
  model = pickle.load(open(model_path,"rb"))
  print("model is loaded")
  
def run(data):
  try:
    #json_data = json.loads(data) # this needs to be records orient
    pd_data = pd.read_json(data, orient = "records")
  
    predictions = model.predict(pd_data)
    return json.dumps(predictions.tolist())
    
  except Exception as e:
    result = str(e)
    return json.dumps({"error": result})
  
""", overwrite = True)


# COMMAND ----------

# MAGIC %md Attach existing AKS cluster

# COMMAND ----------

aks_target = ComputeTarget(workspace = ws, name = compute_name)

# COMMAND ----------

# MAGIC %md Deploy model

# COMMAND ----------

model_inference_config = InferenceConfig(
  environment = env,
  source_directory = "/dbfs/tmp/shenlin/",
  entry_script = "model_score.py"
)

deployment_config = AksWebservice.deploy_configuration(cpu_cores = 2, memory_gb = 1, enable_app_insights = True)

service = Model.deploy(
  workspace = ws,
  name = endpoint_name,
  models = [model],
  inference_config = model_inference_config,
  deployment_config = deployment_config,
  deployment_target = aks_target,
  overwrite = True
)

service.wait_for_deployment(show_output = True)

# COMMAND ----------

# [{"0":-0.0273097857,"1":0.0506801187,"2":-0.0234509473,"3":-0.0159992226,"4":0.0135665216,"5":0.0127778034,"6":0.0265502726,"7":-0.002592262,"8":-0.0109044358,"9":-0.0217882321}]

# COMMAND ----------


