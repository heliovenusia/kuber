import pandas as pd

_RATE_COLS = ["SO Rate", "Rate", "Net Price", "Unit Price"]
_QTY_COLS  = ["Order Qty", "Quantity", "Qty", "Sales Qty"]
_AMT_COLS  = ["MR/CAM Amount", "Amount", "Net Amount", "Value"]
_TERM_COLS = ["Payment Term", "Pymt Terms", "Pay Terms", "Payment Terms"]
_CUST_COLS = ["Customer", "Sold to party", "Bill-To", "Customer Code"]

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

def load_sales_order(so_excel) -> pd.DataFrame:
    df = pd.read_excel(so_excel, sheet_name=0, engine="openpyxl")
    df.columns = df.columns.str.strip()
    so_no_col = _pick(list(df.columns), ["SO No.", "SO No", "Sales Order", "Order No", "Order Number"])
    if so_no_col:
        df = df[df[so_no_col].notna()].copy()
    cols = list(df.columns)

    rate_col = _pick(cols, _RATE_COLS)
    qty_col  = _pick(cols, _QTY_COLS)
    amt_col  = _pick(cols, _AMT_COLS)
    term_col = _pick(cols, _TERM_COLS)
    cust_col = _pick(cols, _CUST_COLS)

    rate = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0) if rate_col else 0.0
    qty  = pd.to_numeric(df[qty_col],  errors="coerce").fillna(0.0) if qty_col  else 0.0
    computed = rate * qty * 1.18
    if computed.sum() == 0 and amt_col:
        df["_amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
    else:
        df["_amount"] = computed
    df["_mr_cam_amt"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0) if amt_col else 0.0
    df["_term"]       = df[term_col].astype(str).str.strip().str.upper() if term_col else ""
    df["_cust"]       = df[cust_col].astype(str).str.strip()             if cust_col else "(unknown)"
    return df

def sales_order_payments(df: pd.DataFrame):
    empty = pd.DataFrame(columns=["customer_code", "amount", "payment_type"])
    if df is None or df.empty:
        return 0.0, 0.0, 0.0, empty
    x = df[df["_mr_cam_amt"] == 0].copy()
    if x.empty:
        return 0.0, 0.0, 0.0, empty
    x["payment_type"] = x["_term"].apply(_classify)
    g = (
        x.groupby(["_cust", "payment_type"], as_index=False)["_amount"]
        .sum()
        .rename(columns={"_cust": "customer_code", "_amount": "amount"})
        .sort_values("amount", ascending=False)
        .reset_index(drop=True)
    )
    total  = float(g["amount"].sum())
    cash   = float(g.loc[g["payment_type"] == "CASH",   "amount"].sum())
    credit = float(g.loc[g["payment_type"] == "CREDIT", "amount"].sum())
    return total, cash, credit, g
