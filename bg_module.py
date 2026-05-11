import pandas as pd
import numpy as np

def load_bg(bg_excel, pred_date=None) -> pd.DataFrame:
    df = pd.read_excel(bg_excel, sheet_name=0, engine="openpyxl")
    val_col = next(
        (c for c in df.columns if "validity" in str(c).lower() and "date" in str(c).lower()),
        None
    )
    if val_col is not None and pred_date is not None:
        df[val_col] = pd.to_datetime(df[val_col], errors="coerce").dt.date
        df = df[df[val_col].isna() | (df[val_col] >= pred_date)].copy()
    for c in ["Value", "Utilized", "Balance", "Br Secured Limit", "Br Unsec Limit"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

    if "Balance" in df.columns:
        df["available_credit"] = df["Balance"].fillna(0.0)
    else:
        sec   = df.get("Br Secured Limit", 0.0)
        unsec = df.get("Br Unsec Limit", 0.0)
        util  = df.get("Utilized", 0.0)
        df["available_credit"] = (
            sec.fillna(0.0) + unsec.fillna(0.0) - util.fillna(0.0)
        )

    out = df[["Cust. Code", "Cust. Name", "Buss.Area", "available_credit"]].copy()

    _n = pd.to_numeric(out["Cust. Code"], errors="coerce")
    out["Cust. Code"] = _n.apply(lambda x: str(int(x)) if pd.notna(x) else "").str.strip()
    out["Cust. Name"] = out["Cust. Name"].astype(str).str.strip()

    out = out.groupby("Cust. Code", as_index=False).agg(
        {"Cust. Name": "first", "available_credit": "sum"}
    )

    return out
