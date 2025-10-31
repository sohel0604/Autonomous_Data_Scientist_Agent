# automl/eda.py
import pandas as pd

def generate_basic_eda(df: pd.DataFrame, target: str = None) -> dict:
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    if target and target in df.columns:
        summary["target_balance"] = df[target].value_counts().to_dict()
    return summary
