# automl/metrics_utils.py
import pandas as pd
from typing import Dict, Any

def parse_compare_df(compare_df: pd.DataFrame) -> Dict[str, Any]:
    if compare_df is None or compare_df.empty:
        return {}
    top_row = compare_df.iloc[0].to_dict()
    metrics = {}
    for col in ['Accuracy', 'AUC', 'Recall', 'Precision', 'F1']:
        if col in top_row:
            metrics[col.lower()] = top_row[col]
    metrics['full_table'] = compare_df.to_dict(orient='records')
    return metrics
