# Databricks notebook source
!pip install sentence-transformers
!pip install --upgrade mlflow
!pip install databricks.feature_engineering

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------

from sentence_transformers import SentenceTransformer
from mlflow.tracking import MlflowClient

import datetime
import mlflow
import mlflow.pyfunc
import numpy as np
import logging

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from configs.config import MODEL_PATH

# COMMAND ----------

class getSentenceEmbeddings(mlflow.pyfunc.PythonModel):
    def __init__(self):
        logging.basicConfig()
        self.logger = logging.getLogger(__name__)

    def load_context(self, context):
        self.model = SentenceTransformer(MODEL_PATH, device='cpu')
        
    def __generate_embeddings(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings[0].tolist()

    def predict(self, context, model_input):
        text = model_input['text'].values[0]
        embeddings = self.__generate_embeddings([text])
        return {'text': text, 'embeddings': embeddings}

# COMMAND ----------

wrapped_model = getSentenceEmbeddings()

# COMMAND ----------

# Define the input and output schemas
input_schema = Schema([ColSpec("string", 'text')])
output_schema = Schema([ColSpec("float", 'embeddings')])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# COMMAND ----------

input_example = {
    'text': 'ทุเรียนทอด'
}

# COMMAND ----------

with mlflow.start_run():
    mlflow.pyfunc.log_model("model", 
                            python_model=wrapped_model, 
                            input_example=input_example, 
                            signature=signature,
                            pip_requirements=['sentence_transformers']
                            )

# COMMAND ----------

# Load the model from the run
run_id = mlflow.last_active_run().info.run_id
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

# COMMAND ----------

# MAGIC %%time
# MAGIC model_output = loaded_model.predict(input_example)
