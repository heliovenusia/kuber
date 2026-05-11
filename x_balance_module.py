import pandas as pd


def load_x_balance(uploaded_or_path) -> pd.DataFrame:
    df = pd.read_excel(uploaded_or_path, engine="openpyxl")
    df.columns = df.columns.str.strip()
    
    cols = list(df.columns)

    cust_col = (
        next((c for c in cols if "cust" in c.lower() and "code" in c.lower()), None)
        or next((c for c in cols if "cust" in c.lower()), None)

        or next((c for c in cols if c.lower() == "account"), None)
        or next((c for c in cols if "account" in c.lower()), None)
        or cols[0]
    )
    bal_col = (
        next((c for c in cols if "balance" in c.lower() or c.lower().endswith(" x")), None)
        or next((c for c in cols if "amount in local" in c.lower()), None)
        or next((c for c in cols if "bal" in c.lower()), None)

        or next((c for c in cols if "amount" in c.lower()), None)
        or cols[1]
    )

    gl_col = next(
        (c for c in cols if "special" in c.lower() and ("g/l" in c.lower() or "gl" in c.lower())),
        None
    )
    if gl_col:
        df = df[df[gl_col].astype(str).str.strip().str.upper() == "X"].copy()

    out = df[[cust_col, bal_col]].copy()
    out.columns = ["customer_code", "balance_x"]

    _n = pd.to_numeric(out["customer_code"], errors="coerce")
    out["customer_code"] = _n.apply(lambda x: str(int(x)) if pd.notna(x) else "").str.strip()


    out["balance_x"] = pd.to_numeric(out["balance_x"], errors="coerce").fillna(0.0)




    out = out.groupby("customer_code", as_index=False)["balance_x"].sum()

    out["balance_x"] = out["balance_x"].clip(lower=0.0)

    return out
