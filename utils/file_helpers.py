# utils/file_helpers.py
import pandas as pd
from io import BytesIO, StringIO

def uploaded_file_to_df(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue()
    # try several common encodings
    for enc in ("utf-8", "utf-16", "latin1"):
        try:
            return pd.read_csv(BytesIO(content), encoding=enc)
        except Exception:
            try:
                return pd.read_csv(StringIO(content.decode(enc)))
            except Exception:
                continue
    # fallback to excel
    try:
        return pd.read_excel(BytesIO(content))
    except Exception as e:
        raise ValueError(f"Could not read uploaded file as CSV or Excel: {e}")
