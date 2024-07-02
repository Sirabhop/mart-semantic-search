# Databricks notebook source
import pandas as pd
import os

def read_and_label_files(directory_path):
    
    all_data = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):  # Adjust file extension as needed
            file_path = os.path.join(directory_path, filename)
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="ISO-8859-1") # Handle encoding issues
            df["file_label"] = filename
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# Example usage
directory = "/Workspace/Shared/ds-projects/mart-search/poc/eval_10/"
result_10_2 = read_and_label_files(directory)

# COMMAND ----------

result_10_2['input_type'] = 'name_desc'

# COMMAND ----------

df_result = pd.concat([result_10, result_10_2], axis=0).sort_values('cos_sim-Accuracy@10', ascending=False).reset_index(drop=True).reset_index()

# COMMAND ----------

df_result.file_label[0]

# COMMAND ----------

df_result[df_result['file_label'].str.find('MiniLM') != -1]

# COMMAND ----------

df_result.to_excel('./main_res.xlsx', index=False, sheet_name='model')

# COMMAND ----------


