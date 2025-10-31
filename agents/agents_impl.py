# agents/agents_impl.py
from typing import Dict, Any
import pandas as pd
from automl.automl_runner import run_automl
from automl.eda import generate_basic_eda
from llm.hf_client import generate_report_local

class DataLoaderAgent:
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if 'df' in context:
            context['dataframe'] = context['df']
        elif 'file_path' in context:
            context['dataframe'] = pd.read_csv(context['file_path'])
        else:
            raise ValueError("No 'df' or 'file_path' provided to DataLoaderAgent")
        return context

class EDAAgent:
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context['dataframe']
        target = context.get('target')
        eda_summary = generate_basic_eda(df, target=target)
        context['eda'] = eda_summary
        return context

class ModelTrainerAgent:
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context['dataframe']
        target = context.get('target')
        result = run_automl(df, target=target)
        context['automl'] = {
            "model_path": result['model_path'],
            "compare_df": result['compare_df']
        }
        return context

class ReportGeneratorAgent:
    def __init__(self):
        pass

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        compare_df = context['automl']['compare_df']
        eda = context.get('eda', {})

        prompt = f"""
You are an expert data scientist. The user ran AutoML on a classification dataset.
Provide a plain-English explanation that covers:
1) What the EDA indicates (data shape, class balance, major missing values)
2) Which model PyCaret selected as best (include top metrics)
3) Why this model might have performed best (consider tree-based vs linear, class balance, feature types).
4) Actionable next steps (feature engineering, data augmentation, threshold tuning).

Here is the EDA summary: {eda}

Here is the PyCaret compare table (top 5): {compare_df.head(5) if hasattr(compare_df, 'head') else compare_df}
Write the explanation in 5-8 short paragraphs.
"""
        # generate a report (local)
        explanation = generate_report_local(prompt, eda=eda, compare_df=compare_df)
        context['report'] = explanation
        return context
