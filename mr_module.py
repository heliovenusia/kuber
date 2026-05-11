import pandas as pd

_DATE_COLS = ["MR. Date", "MR Date"]
_AMT_COLS  = ["Cheque Amount", "Amount", "MR Amount", "Payment Amount"]
_CUST_COLS = ["Customer Code", "Customer", "Cust. Code"]
_TIME_COLS = ["MR. Time", "MR Time", "Time"]

def _pick(cols, candidates):
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None

def load_mr(mr_excel, pred_date=None) -> pd.DataFrame:
    df = pd.read_excel(mr_excel, sheet_name=0, engine="openpyxl")
    df.columns = df.columns.str.strip()
    cols = list(df.columns)

    mr_no_col = _pick(cols, ["MR. Number", "MR Number", "MR No.", "MR No"])
    if mr_no_col:
        df = df[df[mr_no_col].notna()].copy()

    date_col = _pick(cols, _DATE_COLS)
    amt_col  = _pick(cols, _AMT_COLS)
    cust_col = _pick(cols, _CUST_COLS)
    time_col = _pick(cols, _TIME_COLS)
    type_col = next((c for c in cols if c.strip().lower() == "type of customer"), None)

    df["_date"]      = pd.to_datetime(df[date_col], errors="coerce").dt.date if date_col else None
    df["_amount"]    = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0) if amt_col else 0.0
    df["_cust"]      = df[cust_col].astype(str).str.strip() if cust_col else "(unknown)"
    df["_cust_type"] = df[type_col].astype(str).str.strip().str.lower() if type_col else ""
    
    if time_col:
        df["_time"] = pd.to_datetime(df[time_col].astype(str).str.strip(), format="%H:%M:%S", errors="coerce")
        if df["_time"].isna().all():
            df["_time"] = pd.to_datetime(df[time_col].astype(str).str.strip(), errors="coerce")
    else:
        df["_time"] = pd.NaT

    if pred_date is not None and "_date" in df.columns:
        df = df[df["_date"] == pred_date].copy()

    return df

def mr_collected_by_type(df: pd.DataFrame, d):
    if df is None or df.empty or "_date" not in df.columns:
        return 0.0, 0.0
    day = df[df["_date"] == d]
    rwy_mask = day["_cust_type"].str.contains("railway", na=False)
    rwy = float(day[rwy_mask]["_amount"].sum())
    non = float(day[~rwy_mask]["_amount"].sum())
    return rwy, non

def mr_collected_for_date(df: pd.DataFrame, d) -> float:
    if df is None or df.empty or "_date" not in df.columns:
        return 0.0
    return float(df[df["_date"] == d]["_amount"].sum())

def mr_snapshot_time(df: pd.DataFrame, d) -> str:
    if df is None or df.empty or "_date" not in df.columns or "_time" not in df.columns:
        return ""
    rows = df[df["_date"] == d]["_time"].dropna()
    if rows.empty:
        return ""
    latest = rows.max()
    try:
        return latest.strftime("%I:%M %p")
    except Exception:
        return str(latest)
