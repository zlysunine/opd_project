"""Lossless training metadata transport, including empty option dictionaries."""
import json
from pathlib import Path


def decode_extra_info(value):
    """Read JSON metadata without Arrow struct coercion; retain legacy dicts."""
    if isinstance(value,str):
        value=json.loads(value)
        if not isinstance(value,dict):
            raise ValueError('Serialized extra_info must contain a JSON object')
    return value


def export_verl_training(rows, path):
    import pandas as pd

    data = []
    for row in rows:
        metadata = {**row, 'index':row['problem_id']}
        data.append(dict(prompt=row['messages'], data_source=row['benchmark'],
                         reward_model=dict(style='rule',ground_truth=row['ground_truth']),
                         extra_info=json.dumps(metadata,ensure_ascii=False,allow_nan=False)))
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(data).to_parquet(path,index=False)
