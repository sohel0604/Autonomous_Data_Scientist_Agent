# automl/automl_runner.py
import os
import pandas as pd
from pycaret.classification import setup, compare_models, pull, save_model, finalize_model
from typing import Dict, Any

def run_automl(df: pd.DataFrame, target: str, session_id: int = 123) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame columns: {list(df.columns)}")

    # updated setup for PyCaret >=3.x (do NOT use stale 'silent' kwarg)
    s = setup(data=df, target=target, session_id=session_id, verbose=False, log_experiment=False)

    # for speed on small machines, use turbo and small folds; you can change later
    best = compare_models(turbo=True, fold=3)
    best_final = finalize_model(best)

    compare_df = pull()

    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    # save_model will create a .pkl file inside models_dir
    save_name = os.path.join(models_dir, "best_automl_model")
    save_model(best_final, save_name)

    # return path to saved pickle (PyCaret adds .pkl)
    model_path = save_name + ".pkl"
    return {
        "model": best_final,
        "model_path": model_path,
        "compare_df": compare_df,
        "setup_info": s
    }
