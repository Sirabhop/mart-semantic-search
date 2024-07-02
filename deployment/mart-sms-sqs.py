# Databricks notebook source
!pip install sentence-transformers
!pip install --upgrade mlflow
!pip install databricks.feature_engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import ast
import boto3
import os, time

from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

from configs.config import (MODEL_PATH, AWS_REGION, 
                            AWS_ACCESS_KEY, AWS_SECRET_KEY, 
                            AWS_VPC_ENDPOINT, 
                            QUEUE_URL, QUEUE_RBH, 
                            LOG_TABLE_NAME,
                            QUEUE_MAX, MAX_SEC_WAITTIME)

from src.sqs import sqs_connector

# Set timezone
os.environ['TZ'] = 'Asia/Bangkok'
time.tzset()

# COMMAND ----------

model = SentenceTransformer(MODEL_PATH,
                            device='cpu')

# COMMAND ----------

sqs = sqs_connector(AWS_REGION, AWS_ACCESS_KEY, AWS_SECRET_KEY,
                    AWS_VPC_ENDPOINT, QUEUE_URL, QUEUE_RBH, LOG_TABLE_NAME)

# COMMAND ----------

print('QUEUE_MAX=', QUEUE_MAX, 'MAX_SEC_WAITTIME=',MAX_SEC_WAITTIME)

# COMMAND ----------

while True:
    response = sqs.get_msg(QUEUE_MAX, MAX_SEC_WAITTIME)
 
    if 'Messages' in response:
 
        # Start processing
        # sqs.log('get_msg', 'Received message', len(response['Messages']))
        
        for task in response['Messages']: # 10 as max queue
            task_body = json.loads(task['Body'])
            msg_id = task.get('MessageId', '')
            
            queue_data = []
            product_ids = []  # List to store product IDs
 
            try:
                for product in task_body.get('data', []):
                    product_id = product.get('product_id')
                    product_ids.append(product_id)
 
                    product_name_th = product.get('product_name_th', '')
                    product_name_en = product.get('product_name_en', '')
                    embeddings_th = model.encode([product_name_th])[0].tolist() if product_name_th else None
                    embeddings_en = model.encode([product_name_en])[0].tolist() if product_name_en else None
 
                    queue_data.append({
                        'product_id': product_id,
                        'product_name_th': embeddings_th,
                        'product_name_en': embeddings_en
                    })
 
            except:
                continue
            
            num_products = len(product_ids)
            
            # sqs.log('embedding', f"Processed {num_products} products: {', '.join(map(str, product_ids))}", num_products)
            
            send_msg_data = {
                'type': 'PRODUCT_NAME_TRANSFORMED',
                'data': queue_data
            }
            sqs.send_msg(send_msg_data, msg_id)
 
            # sqs.log('send_msg', f"Sent {len(send_msg_data['data'])} queue", len(send_msg_data['data']))
            
    else:
        continue
