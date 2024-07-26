import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def transform_data_to_minipile_format(data):
    transformed_data = []
    for i in range(125000):
        combined_text = f"Source: {data[str(i)]['source']}\nRationale: {data[str(i)]['rationale']}\nTarget: {data[str(i)]['target']}"
        transformed_data.append({"text": combined_text})
    return transformed_data

def save_to_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            json_record = json.dumps(item)
            f.write(json_record + '\n')

def save_to_parquet(data, file_path):
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)

data_path = 'ENTER PATH HERE'
f = open(data_path)
data = json.load(f)

transformed_data = transform_data_to_minipile_format(data)

jsonl_file_path = 'transformed_CoT.jsonl'
save_to_jsonl(transformed_data, jsonl_file_path)

parquet_file_path = 'transformed_CoT.parquet'
save_to_parquet(transformed_data, parquet_file_path)
