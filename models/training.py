# Databricks notebook source
!pip install sentence-transformers==3.0.1
!pip install -U accelerate

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import pandas as pd
import numpy as np
import os, time
import logging
import traceback
import mlflow

from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset, Dataset
from sentence_transformers.losses import TripletLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
    SentenceTransformerTrainingArguments,
)
# from configs.config import (
#     BATCH_SIZE, NUM_EPOCHS, OUTPUT_DIR
# )

# Set timezone
os.environ['TZ'] = 'Asia/Bangkok'
time.tzset()

# COMMAND ----------

# Main training data
df_query = spark.sql("""
                     SELECT
                        a.search_term AS anchor,
                        b.product_name_th AS positive,
                        b.product_id
                     FROM playground_prod.gd_events.mart_search_term_results a
                     INNER JOIN prod.sv_aurora_major_merchant_vw.product_a_d b
                        ON STRING(a.menu_id) = STRING(b.product_id)
                     """).toPandas()

# COMMAND ----------

# Corpus
df_corpus = spark.sql("""
                      SELECT DISTINCT 
                      product_id, merchant_id, product_name_en, product_name_th, product_description_th, product_description_en, product_photo
                      FROM prod.sv_aurora_major_merchant_vw.product_a_d
                      WHERE product_photo IS NOT NULL AND product_price > 0
                      """).toPandas()

# COMMAND ----------

df_train.columns = ['anchor', 'product_ids', 'count', 'positives']

# COMMAND ----------

df_train = df_train[(df_train['count'] > 1) & (df_train['anchor'] != ' ')]

# COMMAND ----------

df_train.reset_index(drop=True, inplace=True)

# COMMAND ----------

while df_query[df_query['negatives'] == df_query['product_id']].shape[0] > 0:
    df_query['negatives'] = np.random.choice(np.array(list(corpus)), size = df_query.shape[0], replace=False)

# COMMAND ----------

train_dataset = Dataset.from_pandas(
    spark.table('playground_prod.ml_semantic_search.training_data').toPandas())

# COMMAND ----------

model = SentenceTransformer(MODEL_PATH)

# COMMAND ----------

train_loss = TripletLoss(model)

# COMMAND ----------

args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCH,
    per_device_train_batch_size=BATCH_SIZE,
    warmup_ratio=0.1,
    fp16=True, 
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATE,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,

    run_name="paraphrases-multi"
)

# COMMAND ----------

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
)

# COMMAND ----------

trainer.train()

# COMMAND ----------

trainer.save_pretrained(output_dir+'/final')
