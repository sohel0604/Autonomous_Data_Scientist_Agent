# llm/hf_client.py
import os
from typing import Optional
from transformers import pipeline, Pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import threading

# global cached pipeline and lock
_PIPELINE: Optional[Pipeline] = None
_PIPELINE_LOCK = threading.Lock()

def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        # choice of local model: google/flan-t5-base is instruction-tuned and works well locally
        model_name = os.getenv("LOCAL_LL_MODEL", "google/flan-t5-base")
        try:
            # load seq2seq pipeline (text2text-generation)
            _PIPELINE = pipeline("text2text-generation", model=model_name, tokenizer=model_name)
            return _PIPELINE
        except Exception as e:
            # if loading fails (out of memory or missing weights), set pipeline to None
            _PIPELINE = None
            print(f"[llm/hf_client] Could not load local model '{model_name}': {e}")
            return None

def generate_report_local(prompt: str, eda: dict = None, compare_df=None) -> str:
    """
    Generate a report using a local model if available.
    If model is not available or OOM, return a deterministic textual fallback.
    """
    pipe = _get_pipeline()
    # If pipeline available, try to generate
    if pipe:
        try:
            # keep prompt size reasonable
            q = prompt if len(prompt) < 4000 else prompt[:4000]
            out = pipe(q, max_length=512, do_sample=False)
            # Transformers returns list of dicts
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                text = out[0].get("generated_text") or out[0].get("translation_text") or str(out[0])
                return text
            return str(out)
        except Exception as e:
            # if model generation failed, continue to fallback text
            print(f"[llm/hf_client] Generation error: {e}")

    # fallback deterministic explanatory text (use EDA & compare_df to craft a human-like explanation)
    try:
        parts = []
        # EDA part
        if eda:
            shape = eda.get("shape")
            parts.append(f"The dataset has {shape[0]} rows and {shape[1]} columns.")
            tb = eda.get("target_balance")
            if tb:
                parts.append(
                    "Target class distribution: "
                    + ", ".join([f"{k}: {v}" for k, v in tb.items()])
                )
            miss = eda.get("missing_values", {})
            top_miss = {
                k: v for k, v in sorted(miss.items(), key=lambda x: -x[1])[:5] if v > 0
            }
            if top_miss:
                parts.append(
                    "Columns with most missing values: "
                    + ", ".join([f"{k}({v})" for k, v in top_miss.items()])
                )
        # model part
        if compare_df is not None:
            try:
                top = compare_df.head(1).to_dict(orient="records")[0]
                model_name = top.get("Model") or list(top.values())[0]
                parts.append(
                    f"PyCaret selected {model_name} as the best model based on cross-validation metrics."
                )
            except Exception:
                parts.append(
                    "PyCaret compared multiple models and selected the best-performing model."
                )
        # actionable steps
        parts.append(
            "Next steps: try feature engineering (interactions, binning), handle missing values carefully, "
            "and tune the best model's hyperparameters."
        )
        return "\n\n".join(parts)
    except Exception as e:
        return (
            "The pipeline completed but the local LLM was not available. "
            "Please install a compatible model or allow the fallback message."
        )
