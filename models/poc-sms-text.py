# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

import json
import pandas as pd
import os, time

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import InformationRetrievalEvaluator


from src.preprocessing import clean_sentence, create_prompt

# Set timezone
os.environ['TZ'] = 'Asia/Bangkok'
time.tzset()

# COMMAND ----------

# Corpus
df_corpus = spark.sql("""
                      SELECT product_id, merchant_id, product_name_en, product_name_th, product_description_th, product_description_en, product_photo
                      FROM prod.sv_aurora_major_merchant_vw.product_a_d
                      WHERE product_photo IS NOT NULL AND product_price > 0
                      """).toPandas()

# COMMAND ----------

df_corpus['product_corpus'] = df_corpus.apply(lambda row: create_prompt(row['product_name_en'],
                                                                        row['product_name_th'],
                                                                        row['product_description_en'],
                                                                        row['product_description_th']
                                                                        ), axis=1)

# COMMAND ----------

df_corpus['product_corpus'] = df_corpus['product_corpus'].apply(lambda x: clean_sentence(x))

# COMMAND ----------

corpus = dict(zip(df_corpus['product_id'].astype(str), df_corpus['product_name_th'])) #(cid -> product_name)

# COMMAND ----------

# Query and Evaluation
df_query = spark.sql("""
                     SELECT
                        a.search_term,
                        b.product_id, 
                        b.product_name_en, 
                        b.product_name_th,
                        a.shop_name
                     FROM playground_prod.gd_events.mart_search_term_results a
                     INNER JOIN prod.sv_aurora_major_merchant_vw.product_a_d b
                        ON a.menu_id = b.product_id
                     """).toPandas()

df_query['c_product_name_th'] = df_query['product_name_th'].apply(lambda x: clean_sentence(x) if x else None)
df_query['c_product_name_en'] = df_query['product_name_en'].apply(lambda x: clean_sentence(x) if x else None)

df_query = df_query[~df_query['product_name_th'].isnull()].reset_index(drop=True)
df_query = df_query.drop_duplicates().reset_index()

df_query.product_id = df_query.product_id.astype(int).astype(str)

# COMMAND ----------

df_query_1 = pd.read_excel('./data/test_on_pt_04062024.xlsx', sheet_name = 'ref_2023_result')
df_query_2 = pd.read_excel('./data/test_on_pt_04062024.xlsx', sheet_name = 'test_on_pt').rename(columns={'KEYWORD':'keyword'})

# COMMAND ----------

df_query = pd.concat([df_query_1[['keyword']], df_query_2[['keyword']]], axis=0, ignore_index=True)

# COMMAND ----------

len(df_query.keyword.unique())

# COMMAND ----------

def check_a_in_b(search, result):
    if search in result:
        return 1
    else:
        return 0
def check_a_eq_b(search, result):
    if search == result:
        return 1
    else:
        return 0

# COMMAND ----------

df_query['implication_flag'] = df_query.apply(lambda row: check_a_in_b(row['search_term'], row['product_name_th']), axis=1)
df_query['search_shop_flag'] = df_query.apply(lambda row: check_a_eq_b(row['search_term'], row['shop_name']), axis=1)

# COMMAND ----------

df_query['search_shop_flag'].sum()

# COMMAND ----------

# Use group by query should account this nuance in differences
df_groupby = df_query.groupby('search_term').agg({'product_id':[set, 'count']})
df_groupby = df_groupby.reset_index()

df_groupby.columns = ['search_term', 'product_ids', 'count']

df_eval = df_groupby[df_groupby['count'] >= 10].reset_index(drop=True).reset_index()

df_eval['index'] = df_eval['index'].astype(str)
print(f'Total corpus to embed = {df_eval.shape[0]}.')

# COMMAND ----------

relevant_product_ids = dict(zip(df_eval['index'], df_eval['product_ids'])) #(qid -> product_name)
queries = dict(zip(df_eval['index'], df_eval['search_term'])) #(qid -> search_ter,)

# COMMAND ----------

list_models = [
    'sentence-transformers/all-MiniLM-L6-v2', 
    'sentence-transformers/LaBSE', 
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'jinaai/jina-embeddings-v2-small-en', 
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'sentence-transformers/distiluse-base-multilingual-cased-v2', #512
    'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking', #768
    'sentence-transformers/use-cmlm-multilingual', #768 USE+LaBSE
    'sentence-transformers/stsb-xlm-r-multilingual', #768 
   'sentence-transformers/quora-distilbert-multilingual', #768
    'sentence-transformers/paraphrase-xlm-r-multilingual-v1', #768

    'sentence-transformers/msmarco-MiniLM-L-6-v3',
    'sentence-transformers/msmarco-MiniLM-L-12-v3',
    'sentence-transformers/msmarco-distilbert-base-v4',
    'sentence-transformers/msmarco-roberta-base-v3',
    'sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned'
]

# COMMAND ----------

for model_name in list_models[-1]:

    model_gpu = SentenceTransformer(model_name, device='cuda')
    ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_product_ids,
            name=model_name.split('/')[1],
            batch_size = 512,
            show_progress_bar = False
        )
    
    results = ir_evaluator(model_gpu, output_path='/Workspace/Shared/ds-projects/mart-search/poc/eval_10/name_desc')

    print('complete', model_name)

# COMMAND ----------

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cuda')

# COMMAND ----------

#sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
c_multilingual_minilm = model.encode(list(corpus.values()), batch_size=1024, convert_to_tensor=True, show_progress_bar=True)

# COMMAND ----------

# Encode from sentence-transformers/paraphrase-xlm-r-multilingual-v1
tensor_corpus = model_gpu.encode(list(corpus.values()), batch_size=512, convert_to_tensor=True, show_progress_bar=True)

# COMMAND ----------

def get_search_result(model, ts_corpus, query, k):
    def get_product(idx):
        ckeys = list(corpus.keys())
        return corpus[ckeys[idx]]

    ts_query = model.encode(query, batch_size=512, convert_to_tensor=True)
    
    res_queries = util.semantic_search(
                     query_embeddings=ts_query,
                     corpus_embeddings=ts_corpus,
                     top_k=k,
                     query_chunk_size=512,
                     ) # -> corpus idx

    # for query in res_queries:
    #     for sim_product in query:
    #         sim_product['product_name'] = get_product(sim_product['corpus_id'])
    
    return res_queries

# COMMAND ----------

list_dict = get_search_result(model, c_multilingual_minilm, df_query.keyword.values, 10)

# COMMAND ----------

pd.DataFrame(list_dict[0])

# COMMAND ----------

from tqdm import tqdm

# COMMAND ----------

df_res = pd.DataFrame()
for i, q_res in tqdm(enumerate(list_dict), total=len(list_dict)):
    df_res = pd.concat([df_res, pd.DataFrame(q_res)], axis=0)

# COMMAND ----------

df_corpus = df_corpus.reset_index()

# COMMAND ----------

df_res = df_res.merge(df_corpus[['index', 'product_name_th', 'merchant_id']], left_on='corpus_id', right_on='index', how='left')

# COMMAND ----------

# df_imp = df_query[df_query['implication_flag']==1]
df_imp = df_query

# COMMAND ----------

df_imp = df_imp.reset_index()

# COMMAND ----------

df_imp

# COMMAND ----------

sequence = []
for num in range(0, int(df_res.shape[0]/10)):
    sequence.extend([num] * 10)  # Repeat the number 10 times

# COMMAND ----------

df_res['index'] = sequence

# COMMAND ----------

df_res

# COMMAND ----------

# df_eval_imp = df_res.merge(df_imp[['index', 'search_term', 'product_name_th', 'shop_name']], on='index', how='left')
df_eval_imp = df_res.merge(df_imp[['index', 'keyword']], on='index', how='left')

# COMMAND ----------

df_eval_imp = df_eval_imp.groupby(['index', 'keyword']).agg(list)

# COMMAND ----------

# df_eval_imp = df_eval_imp.reset_index()
df_eval_imp.columns = ['index', 'keyword', 'corpus_ids', 'scores', 'product_name_ths', 'merchant_ids']

# COMMAND ----------

df_eval_imp

# COMMAND ----------

df_eval_imp = df_eval_imp.merge(df_imp[['index', 'search_term', 'product_name_th', 'shop_name']], on='index', how='left')

# COMMAND ----------

df_eval_imp.to_excel('./poc_sms_requested_result.xlsx', index=False)

# COMMAND ----------

ds_eval_imp = spark.createDataFrame(df_eval_imp)

# COMMAND ----------

ds_eval_imp.write.mode('overWrite').saveAsTable('playground_prod.sirabhop_eda.poc_desc_sms_result')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM playground_prod.sirabhop_eda.poc_sms_result

# COMMAND ----------

model_name = 'sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned'
model_gpu = SentenceTransformer(model_name, device='cuda')
ir_evaluator = InformationRetrievalEvaluator(
        queries=corpus,
        corpus=queries,
        relevant_docs=relevant_product_ids,
        name=model_name.split('/')[1],
        batch_size = 512,
        show_progress_bar = False
    )

# COMMAND ----------


results = ir_evaluator(model_gpu, output_path='/Workspace/Shared/ds-projects/mart-search/poc/eval_10/name_desc')
