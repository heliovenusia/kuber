import pandas as pd

_CUST_COLS = ["Customer", "Sold to party", "Bill-To", "Customer Code"]
_TERM_COLS = ["Payment Terms", "Payment Term", "Pymt Terms", "Pay Terms"]

def _pick(cols, candidates):
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None

def _classify(t: str) -> str:
    u = str(t).upper()
    if "ZSCR" in u:
        return "CREDIT"
    if "ZCAS" in u:
        return "CASH"
    return "OTHER"

def load_ro(ro_excel) -> pd.DataFrame:
    df = pd.read_excel(ro_excel, sheet_name=0, engine="openpyxl")
    df.columns = df.columns.str.strip()
    cols = list(df.columns)

    rel_col = _pick(cols, ["Release Ord", "Release Order", "RO Number"])
    if rel_col:
        df = df[df[rel_col].notna()].copy()

    term_col = _pick(list(df.columns), _TERM_COLS)
    df["_term"] = df[term_col].astype(str).str.strip().str.upper() if term_col else ""

    for c in ["RO Value", "Paymnt Amt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df

def ro_payments(df: pd.DataFrame):
    empty = pd.DataFrame(columns=["customer_code", "amount", "payment_type"])
    if df is None or df.empty:
        return 0.0, 0.0, 0.0, empty

    if "RO Value" not in df.columns or "Paymnt Amt" not in df.columns:
        return 0.0, 0.0, 0.0, empty

    x = df[df["Paymnt Amt"] == 0].copy()
    if x.empty:
        return 0.0, 0.0, 0.0, empty

    x["payment_type"] = x["_term"].apply(_classify) if "_term" in x.columns else "OTHER"

    cust_col = _pick(list(x.columns), _CUST_COLS)
    if cust_col is None:
        total = float(x["RO Value"].sum())
        return total, 0.0, 0.0, pd.DataFrame({"customer_code": ["(unknown)"], "amount": [total], "payment_type": ["OTHER"]})

    x[cust_col] = x[cust_col].astype(str).str.strip()
    g = (
        x.groupby([cust_col, "payment_type"], as_index=False)["RO Value"]
        .sum()
        .rename(columns={cust_col: "customer_code", "RO Value": "amount"})
        .sort_values("amount", ascending=False)
        .reset_index(drop=True)
    )
    total  = float(g["amount"].sum())
    cash   = float(g.loc[g["payment_type"] == "CASH",   "amount"].sum())
    credit = float(g.loc[g["payment_type"] == "CREDIT", "amount"].sum())
    return total, cash, credit, g
