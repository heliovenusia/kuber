import re, math
from datetime import date, timedelta
from calendar import monthrange
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

try:
    from holiday_module import (
        hol_proximity_flags as _hol_proximity_flags,
        is_default_closed    as _is_default_closed,
        is_bank_holiday      as _is_bank_holiday,
    )
    _HOL_IMPORTED = True
except ImportError:
    _HOL_IMPORTED        = False
    _is_default_closed   = None
    _is_bank_holiday     = None

DEFAULT_MAX_AGE = 365



def _norm_col(c):
    c = str(c).strip().lower()
    c = re.sub(r"\s+", " ", c).replace("\n", " ")
    return c

def prep_df(df):
    df = df.dropna(how="all").copy()
    df.columns = [_norm_col(c) for c in df.columns]
    return df

def detect_columns(df):
    
    cols = set(df.columns)
    def f(cands):
        for c in cands:
            if c in cols:
                return c
        return None
    return {
        "invoice_no":       f(["invoice no.", "invoice no", "invoice number", "inv no", "inv no.", "invoice"]),
        "invoice_date":     f(["invoice date", "inv date", "date of invoice"]),
        "cam_date":         f(["cam date", "credit acceptance memo date", "cam", "credit memo date", "ca date"]),
        "due_date":         f(["due date", "duedate"]),
        "invoice_amount":   f(["invoice amount", "inv amount", "invoice amt", "amount"]),
        "due_amount":       f(["due amount", "due amt", "outstanding amount", "balance amount", "balance amt"]),
        
        "mr_date":          f(["cheque dt.", "cheque dt", "cheque date", "mr date", "payment date"]),
        "mr_amnt_breakup":  f(["mr amnt. breakup", "mr amnt breakup", "mr amount breakup",
                                "mr amnt. break up", "mr amnt break up"]),
        "customer":         f(["customer name", "customer", "party name", "customer name/ code"]),
        "customer_code":    f(["customer code", "party code", "cust code"]),
        "ifc":              f(["ifc", "ifc days", "interest free credit days"]),
        "ibc":              f(["ibc", "ibc days", "interest bearing credit days"]),
        "business_area":    f(["buss. area", "business area", "bus area", "buss area", "ba", "buss.area"]),
        "invoice_type":     f(["invoice type", "inv type", "type of invoice", "inv. type"]),
        "product":          f(["product", "product name", "item", "item name"]),
        "product_category": f(["product category", "product cat", "category", "prod category"]),
        "int_terms":        f(["int terms", "interest terms", "int. terms"]),
        "payment_term":     f(["payment term", "payment terms", "pay term", "pay terms"]),
    }

def basic_clean(df, m):
    df = df.copy()
    for k in ["invoice_date", "cam_date", "due_date", "mr_date"]:
        c = m.get(k)
        if c and c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for k in ["invoice_amount", "due_amount", "ifc", "ibc"]:
        c = m.get(k)
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    
    for k in ["customer", "customer_code", "business_area", "invoice_type",
              "product", "product_category", "int_terms", "payment_term"]:
        c = m.get(k)
        if c and c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()
    return df




def read_workbook(uploaded):
    xls = pd.ExcelFile(uploaded, engine="openpyxl")
    frames = []
    
    for n in xls.sheet_names:
        frames.append(pd.read_excel(xls, sheet_name=n, engine="openpyxl"))
    
    return pd.concat(frames, ignore_index=True)



def _mode(series):
    s = series.astype(str).fillna("").str.strip()
    s = s[s != ""]
    if s.empty:
        return ""
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]

def _split_slash(val):
    
    if pd.isna(val) or str(val).strip() in ("", "nan"):
        return []
    return [x.strip() for x in re.split(r"[/,;|]+", str(val)) if x.strip()]



def build_invoices_and_receipts(flat_df, m):
    inv_no    = m["invoice_no"]
    inv_dt    = m["invoice_date"]
    inv_amt   = m["invoice_amount"]
    due_amt   = m.get("due_amount")
    cust      = m.get("customer")
    cust_code = m.get("customer_code")
    cam_dt    = m.get("cam_date")
    ifc       = m.get("ifc")
    ibc       = m.get("ibc")
    ba        = m.get("business_area")
   
    it        = m.get("invoice_type")
    prod      = m.get("product")
    prod_cat  = m.get("product_category")
    int_terms = m.get("int_terms")
    pay_term  = m.get("payment_term")
    mr_dt_col   = m.get("mr_date")
    mr_amnt_col = m.get("mr_amnt_breakup")

    df = flat_df.dropna(subset=[inv_no, inv_dt, inv_amt]).copy()

    df["invoice_id"]       = df[inv_no].astype(str).str.strip()
    df["invoice_date"]     = pd.to_datetime(df[inv_dt], errors="coerce").dt.date
    df["invoice_amount"]   = pd.to_numeric(df[inv_amt], errors="coerce").fillna(0.0).clip(0, 1e12)
    df["due_amount"]       = (pd.to_numeric(df[due_amt], errors="coerce").fillna(0.0).clip(0, 1e12)
                              if (due_amt and due_amt in df.columns) else 0.0)
    df["customer"]         = df[cust].astype(str).str.strip()       if cust      else ""
    if cust_code:
        _n = pd.to_numeric(df[cust_code], errors="coerce")
        df["customer_code"] = _n.apply(lambda x: str(int(x)) if pd.notna(x) else "").str.strip()
    else:
        df["customer_code"] = ""
    df["cam_date"]         = (pd.to_datetime(df[cam_dt], errors="coerce").dt.date
                              if (cam_dt and cam_dt in df.columns) else None)
    df["ifc_days"]         = (pd.to_numeric(df[ifc], errors="coerce")
                              if (ifc and ifc in df.columns) else np.nan)
    df["ibc_days"]         = (pd.to_numeric(df[ibc], errors="coerce")
                              if (ibc and ibc in df.columns) else np.nan)
   
   
    df["business_area"]    = df[ba].astype(str).str.strip()          if (ba        and ba        in df.columns) else ""
    df["invoice_type"]     = df[it].astype(str).str.strip()          if (it        and it        in df.columns) else ""
    df["product"]          = df[prod].astype(str).str.strip()        if (prod      and prod      in df.columns) else ""
    df["product_category"] = df[prod_cat].astype(str).str.strip()    if (prod_cat  and prod_cat  in df.columns) else ""
    df["int_terms"]        = df[int_terms].astype(str).str.strip()   if (int_terms and int_terms in df.columns) else ""
    df["payment_term"]     = df[pay_term].astype(str).str.strip()    if (pay_term  and pay_term  in df.columns) else ""
    df["_mr_date_raw"]     = (df[mr_dt_col].astype(str)
                              if (mr_dt_col and mr_dt_col in df.columns) else "")
    df["_mr_amnt_raw"]     = (df[mr_amnt_col].astype(str)
                              if (mr_amnt_col and mr_amnt_col in df.columns) else "")

    df = df.dropna(subset=["invoice_date"])

    agg = {
        "invoice_date":     "min",
        "invoice_amount":   "max",
        "due_amount":       "min",    
        "customer":         "first",
        "customer_code":    "first",
        "business_area":    "first",  
        
        
        "invoice_type":     "first",
        "cam_date":         "first",
        "ifc_days":         "first",
        "ibc_days":         "first",
        "product":          "first",
        "product_category": "first",
        "int_terms":        "first",
        "payment_term":     "first",
    }
    inv = df.groupby("invoice_id", as_index=False).agg(agg)
    inv = inv.dropna(subset=["invoice_date"])
    inv["invoice_amount"] = inv["invoice_amount"].fillna(0.0).clip(0, 1e12)
    inv["due_amount"]     = inv["due_amount"].fillna(0.0).clip(0, 1e12)

   
    cam_ts   = pd.to_datetime(inv["cam_date"], errors="coerce")
    ifc_d    = pd.to_numeric(inv["ifc_days"],  errors="coerce").fillna(0).astype(int)
    ibc_d    = pd.to_numeric(inv["ibc_days"],  errors="coerce").fillna(0).astype(int)
    computed = cam_ts + pd.to_timedelta(ifc_d + ibc_d, unit="D")
    inv["due_date"] = computed.dt.date.where(cam_ts.notna(), other=None)

    rec_cols = ["customer_code", "customer", "mr_date", "paid"]

    mask = df["_mr_date_raw"].str.strip().ne("") & df["_mr_date_raw"].str.strip().ne("nan")
    
    rdf  = df.loc[mask, ["customer_code", "customer", "_mr_date_raw", "_mr_amnt_raw"]]

    if rdf.empty:
        rec = pd.DataFrame(columns=rec_cols)
    else:
        cc_arr  = rdf["customer_code"].to_numpy()
        cn_arr  = rdf["customer"].to_numpy()
        dt_arr  = rdf["_mr_date_raw"].to_numpy()
        amt_arr = rdf["_mr_amnt_raw"].to_numpy()

        raw_rows: list = []          
        for cc, cn, d_raw, a_raw in zip(cc_arr, cn_arr, dt_arr, amt_arr):
            dates = [x.strip() for x in str(d_raw).split("/") if x.strip()]
            amnts = [x.strip() for x in str(a_raw).split("/") if x.strip()]
            for j, dt in enumerate(dates):
                if dt == "00.00.0000":
                    continue
                amt_str = amnts[j] if j < len(amnts) else ""
                try:
                    amt = float(amt_str)
                except (ValueError, TypeError):
                    continue
                if amt > 0:
                    raw_rows.append((cc, cn, dt, amt))

        
        if not raw_rows:
            rec = pd.DataFrame(columns=rec_cols)
        else:
            tmp = pd.DataFrame(raw_rows, columns=["customer_code", "customer", "_dt_str", "paid"])
            tmp["mr_date"] = pd.to_datetime(
                tmp["_dt_str"], format="%d.%m.%Y", errors="coerce"
            ).dt.date
            rec = (tmp.dropna(subset=["mr_date"])
                      .drop(columns=["_dt_str"])
                      [rec_cols]
                      .drop_duplicates(subset=["customer_code", "mr_date", "paid"])
                      .reset_index(drop=True))

    cutoff_2yr = date.today() - timedelta(days=730)
    
    
    
    rec = rec[rec["mr_date"] >= cutoff_2yr].reset_index(drop=True)

    return inv, rec



def _bank_closed_flags(d):
    dow = d.weekday()
    dom = d.day
    wom = (dom - 1) // 7 + 1
    is_sun     = 1.0 if dow == 6 else 0.0
    
    
    is_sat     = dow == 5
    is_2nd_sat = 1.0 if (is_sat and wom == 2) else 0.0
    is_4th_sat = 1.0 if (is_sat and wom == 4) else 0.0
    is_bank_closed = 1.0 if (is_sun or is_2nd_sat or is_4th_sat) else 0.0
    return is_sun, is_2nd_sat, is_4th_sat, is_bank_closed

def _days_to_quarter_end(d):
    q_end_m  = 3 * ((d.month - 1) // 3 + 1)
    last_day = monthrange(d.year, q_end_m)[1]
    return float((date(d.year, q_end_m, last_day) - d).days)

def _days_to_fy_end(d):
    fy_end = date(d.year, 3, 31) if d.month <= 3 else date(d.year + 1, 3, 31)
    return float((fy_end - d).days)

def _calendar_feats(d):
    dow      = d.weekday()
    dom      = d.day
    last_dom = monthrange(d.year, d.month)[1]
    
    d2me     = float(last_dom - dom)
    is_sun, is_2nd_sat, is_4th_sat, is_bank_closed = _bank_closed_flags(d)
    m = d.month
    return {
        "day_of_week":         float(dow),
        "day_of_month":        float(dom),
        "week_of_year":        float(d.isocalendar().week),
        "days_to_month_end":   d2me,
        "is_month_end_3":      1.0 if d2me <= 2 else 0.0,
        "is_weekend":          1.0 if dow >= 5 else 0.0,
        "is_sunday":           float(is_sun),
        
        "is_2nd_sat":          float(is_2nd_sat),
        "is_4th_sat":          float(is_4th_sat),
        "is_bank_closed":      float(is_bank_closed),
        "days_to_quarter_end": _days_to_quarter_end(d),
        "days_to_fy_end":      _days_to_fy_end(d),
        "month_sin":           math.sin(2.0 * math.pi * m / 12.0),
        "month_cos":           math.cos(2.0 * math.pi * m / 12.0),
    }


def _days_since_last_open(pred_date, holidays=None):
    d = pred_date - timedelta(days=1)
    for gap in range(1, 15):
        _, _, _, is_closed = _bank_closed_flags(d)
        is_hol = False
        
        if _HOL_IMPORTED and holidays and _is_bank_holiday is not None:
            try:
                is_hol = _is_bank_holiday(d, holidays)
            except Exception:
                pass
        if is_closed < 0.5 and not is_hol:
            return float(gap)
        d -= timedelta(days=1)
    return 14.0




def _outstanding_as_of(inv, as_of, paid_to_date=None):
    
    _empty = pd.DataFrame(columns=["customer_code", "customer", "outstanding",
                                   "business_area", "invoice_type",
                                   "product", "product_category", "int_terms", "payment_term"])
    inv0 = inv[inv["invoice_date"] <= as_of].copy()
    if inv0.empty:
        return _empty

    if paid_to_date is not None and not paid_to_date.empty:
        
        
        
        total_invoiced = inv0.groupby("customer_code")["invoice_amount"].sum()
        out_amt = (total_invoiced
                   - paid_to_date.reindex(total_invoiced.index).fillna(0)
                   ).clip(lower=0)
    else:
        inv0 = inv0[inv0["due_amount"] > 1e-6]
        if inv0.empty:
            return _empty
        out_amt = inv0.groupby("customer_code")["due_amount"].sum()

    out_amt = out_amt[out_amt > 1e-6]
    if out_amt.empty:
        return _empty

    valid = out_amt.index
    
    
    
    
    inv0  = inv0[inv0["customer_code"].isin(valid)]
    cat_cols = [c for c in ["customer", "business_area", "invoice_type",
                             "product", "product_category", "int_terms", "payment_term"]
                if c in inv0.columns]
    agg_dict = {c: "first" for c in cat_cols}
    meta     = inv0.groupby("customer_code").agg(**{c: (c, fn) for c, fn in agg_dict.items()})
    result   = pd.DataFrame({"outstanding": out_amt}).join(meta).reset_index()
    return result[result["outstanding"] > 1e-6].copy()


def _open_invoices_as_of(inv, as_of):
    inv0 = inv[inv["invoice_date"] <= as_of].copy()
    if inv0.empty:
        return pd.DataFrame()
    
    inv0 = inv0[inv0["due_amount"] > 1e-6].copy()
    inv0["open_amt"] = inv0["due_amount"]
    return inv0.copy()




def _due_date_pressure(open_inv, pred_date):
    pred_ts    = pd.Timestamp(pred_date)
    due_ts     = pd.to_datetime(open_inv["due_date"],  errors="coerce")
    cam_ts     = pd.to_datetime(open_inv["cam_date"],  errors="coerce")
    ifc_num    = pd.to_numeric(open_inv["ifc_days"],   errors="coerce").fillna(0)
    ifc_end_ts = cam_ts + pd.to_timedelta(ifc_num, unit="D")

    open_inv = open_inv.copy()
    open_inv["days_to_due"]     = (due_ts - pred_ts).dt.days.fillna(999).astype(float)
    
    
    
    open_inv["days_past_due"]   = np.maximum(0.0, (pred_ts - due_ts).dt.days.fillna(0).astype(float))
    open_inv["is_overdue"]      = np.where(due_ts.notna() & (pred_ts > due_ts), 1.0, 0.0)
    open_inv["in_ifc_window"]   = np.where(ifc_end_ts.notna() & (pred_ts <= ifc_end_ts), 1.0, 0.0)
    open_inv["in_ibc_window"]   = np.where(
        ifc_end_ts.notna() & due_ts.notna() & (pred_ts > ifc_end_ts) & (pred_ts <= due_ts), 1.0, 0.0)
    open_inv["overdue_amt"]     = np.where(open_inv["is_overdue"] > 0.5, open_inv["open_amt"], 0.0)
    open_inv["ibc_amt"]         = np.where(open_inv["in_ibc_window"] > 0.5, open_inv["open_amt"], 0.0)
    open_inv["days_to_ifc_end"] = (ifc_end_ts - pred_ts).dt.days.fillna(999).astype(float)

    def _wt(g, col):
        w  = g["open_amt"].values.astype(float)
        v  = g[col].values.astype(float)
        ws = w.sum()
        return float(np.nansum(v * w) / ws) if ws > 1e-9 else 0.0

    agg = open_inv.groupby("customer_code").apply(lambda g: pd.Series({
        "wavg_days_to_due":    _wt(g, "days_to_due"),
        "min_days_to_due":     float(g["days_to_due"].min()),
        
        "max_days_past_due":   float(g["days_past_due"].max()),
        "overdue_outstanding": float(g["overdue_amt"].sum()),
        "overdue_share":       float(g["overdue_amt"].sum() / (g["open_amt"].sum() + 1e-9)),
        "share_in_ibc":        float(g["ibc_amt"].sum()     / (g["open_amt"].sum() + 1e-9)),
        "any_overdue":         float((g["is_overdue"] > 0.5).any()),
        "any_in_ibc":          float((g["in_ibc_window"] > 0.5).any()),
        "share_in_ifc":        float(g.loc[g["in_ifc_window"] > 0.5, "open_amt"].sum() /
                                     (g["open_amt"].sum() + 1e-9)),
        "min_days_to_ifc_end": float(g["days_to_ifc_end"].min()),
        "n_invoices_due_7d":   float((g["days_to_due"] <= 7).sum()),
        "n_invoices_due_30d":  float((g["days_to_due"] <= 30).sum()),
        "amt_due_7d":          float(g.loc[g["days_to_due"] <= 7,  "open_amt"].sum()),
        "amt_due_30d":         float(g.loc[g["days_to_due"] <= 30, "open_amt"].sum()),
    }), include_groups=False).reset_index()
    return agg



def build_customer_features(inv, rec, as_of, pred_date, holidays=None):
    rec_asof_all = rec[rec["mr_date"] <= as_of] if not rec.empty else rec
    if not rec_asof_all.empty and not inv.empty:
        inv_start    = inv["invoice_date"].dropna().min()
        
        rec_for_recon = rec_asof_all[rec_asof_all["mr_date"] >= inv_start]
        paid_to_date  = (rec_for_recon.groupby("customer_code")["paid"].sum()
                         if not rec_for_recon.empty else pd.Series(dtype=float))
    else:
        paid_to_date = pd.Series(dtype=float)

    out = _outstanding_as_of(inv, as_of, paid_to_date)
    if out.empty:
        return pd.DataFrame()

    open_inv = _open_invoices_as_of(inv, as_of)
    as_of_ts = pd.Timestamp(as_of)

    inv_date_ts     = pd.to_datetime(open_inv["invoice_date"], errors="coerce")
    open_inv        = open_inv.copy()
    open_inv["age"] = (as_of_ts - inv_date_ts).dt.days.clip(0, DEFAULT_MAX_AGE).fillna(0).astype(float)

    age_agg = open_inv.groupby("customer_code").apply(lambda g: pd.Series({
        "open_invoices": float(g["invoice_id"].nunique()),
        "wavg_age":      float(np.sum(g["age"] * g["open_amt"]) / (g["open_amt"].sum() + 1e-9)),
        "max_age":       float(g["age"].max()),
    }), include_groups=False).reset_index()

    base = out.merge(age_agg, on="customer_code", how="left")

    due_feats = _due_date_pressure(open_inv, pred_date)
    base = base.merge(due_feats, on="customer_code", how="left")
    for c in ["wavg_days_to_due", "min_days_to_due", "max_days_past_due",
              "overdue_outstanding", "overdue_share", "share_in_ibc",
              
              "any_overdue", "any_in_ibc", "share_in_ifc",
              "min_days_to_ifc_end", "n_invoices_due_7d", "n_invoices_due_30d",
              "amt_due_7d", "amt_due_30d"]:
        if c in base.columns:
            base[c] = base[c].fillna(0.0)

    
    rec_asof = rec[rec["mr_date"] <= as_of] if not rec.empty else rec
    if not rec_asof.empty:
        cust_total_paid = rec_asof.groupby("customer_code")["paid"].sum()
        cust_n_days     = rec_asof.groupby("customer_code")["mr_date"].nunique()
        last_pay_ts     = pd.to_datetime(rec_asof.groupby("customer_code")["mr_date"].max())
        days_since      = (as_of_ts - last_pay_ts).dt.days.astype(float)
        inv_amt_to_date = inv[inv["invoice_date"] <= as_of].groupby("customer_code")["invoice_amount"].sum()

        
        base["cust_total_paid"]      = base["customer_code"].map(cust_total_paid).fillna(0.0)
        base["cust_n_pay_days"]      = base["customer_code"].map(cust_n_days).fillna(0.0)
        base["cust_days_since_pay"]  = base["customer_code"].map(days_since).fillna(999.0)
        base["cust_avg_pay_per_day"] = base["cust_total_paid"] / (base["cust_n_pay_days"] + 1e-9)
        base["cust_pay_rate"]        = (base["cust_total_paid"] /
                                        (base["customer_code"].map(inv_amt_to_date).fillna(1.0) + 1e-9)).clip(0.0, 2.0)
    else:
        for c in ["cust_total_paid", "cust_n_pay_days", "cust_avg_pay_per_day", "cust_pay_rate"]:
            base[c] = 0.0
        base["cust_days_since_pay"] = 999.0

    
    rec_30d = rec_asof[rec_asof["mr_date"] >= (as_of - timedelta(days=29))] if not rec_asof.empty else rec_asof
    rec_7d  = rec_asof[rec_asof["mr_date"] >= (as_of - timedelta(days=6))]  if not rec_asof.empty else rec_asof

    def _roll(r, col):
        if r.empty:
            return pd.DataFrame(columns=["customer_code", col])
        return r.groupby("customer_code", as_index=False)["paid"].sum().rename(columns={"paid": col})

    base = base.merge(_roll(rec_30d, "paid_30d"), on="customer_code", how="left")
    
    
    base = base.merge(_roll(rec_7d,  "paid_7d"),  on="customer_code", how="left")
    base["paid_30d"]     = base["paid_30d"].fillna(0.0)
    base["paid_7d"]      = base["paid_7d"].fillna(0.0)
    base["velocity_30d"] = base["paid_30d"] / (base["outstanding"] + 1e-9)
    base["velocity_7d"]  = base["paid_7d"]  / (base["outstanding"] + 1e-9)

    
    for k, v in _calendar_feats(pred_date).items():
        base[k] = v

    base["days_since_last_open"] = _days_since_last_open(pred_date, holidays)

    
    if holidays is not None and _HOL_IMPORTED:
        
        
        for k, v in _hol_proximity_flags(pred_date, holidays).items():
            base[k] = v

    return base




def _preprocess(X):
    cat = [c for c in ["customer_code", "customer", "business_area", "invoice_type",
                        "product", "product_category", "int_terms", "payment_term"]
           if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ],
        remainder="drop"
    )

def _fit_clf(clf_rows):
    if not clf_rows:
        return None
    clf_df = pd.concat(clf_rows, ignore_index=True)
    if clf_df["y_clf"].sum() < 2:
        return None
    feat_cols = [c for c in clf_df.columns if c not in ("y_clf", "y_amt")]
    X = clf_df[feat_cols]
    y = clf_df["y_clf"].values
   
    pre = _preprocess(X)
    clf = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=500,
        min_samples_leaf=5, random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    return pipe

_RNG = np.random.default_rng(42)

def _build_clf_rows(inv, rec, pay_days, holidays=None, neg_ratio=3):
    paid_set = set(zip(rec["customer_code"], rec["mr_date"]))
    clf_rows = []
    for d in pay_days:
        as_of    = d - timedelta(days=1)
        rec_asof = rec[rec["mr_date"] < d]
        Xd = build_customer_features(inv, rec_asof, as_of, d, holidays)
        if Xd.empty:
            continue
        Xd = Xd.copy()
        Xd["y_clf"] = Xd["customer_code"].apply(lambda c: int((c, d) in paid_set))
        day_amts = rec[rec["mr_date"] == d].groupby("customer_code")["paid"].sum()
        Xd["y_amt"] = Xd["customer_code"].map(day_amts).fillna(0.0)

        pos = Xd[Xd["y_clf"] == 1]
        neg = Xd[Xd["y_clf"] == 0]

        if len(pos) == 0:
            max_neg_zero = 50
            if len(neg) > max_neg_zero:
                neg = neg.sample(n=max_neg_zero, random_state=int(_RNG.integers(1e6)))
            if not neg.empty:
                clf_rows.append(neg)
            continue

        max_neg = neg_ratio * len(pos)
        if len(neg) > max_neg:
            neg = neg.sample(n=max_neg, random_state=int(_RNG.integers(1e6)))

        clf_rows.append(pd.concat([pos, neg], ignore_index=True))
    return clf_rows


def _fit_reg(clf_rows):
    if not clf_rows:
        return None
    all_df = pd.concat(clf_rows, ignore_index=True)
    pos_df = all_df[all_df["y_amt"] > 0].copy()
    if len(pos_df) < 5:
        return None
    feat_cols = [c for c in pos_df.columns if c not in ("y_clf", "y_amt")]
    pre = _preprocess(pos_df[feat_cols])
    reg = HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=300,
        min_samples_leaf=3, random_state=42)
    pipe = Pipeline([("pre", pre), ("reg", reg)])
    pipe.fit(pos_df[feat_cols], pos_df["y_amt"].values)
    return pipe

def train_customer_models(inv, rec, holidays=None):
    if inv.empty or rec.empty:
        return None
    all_pay_dates = sorted(rec["mr_date"].unique())
    if len(all_pay_dates) < 2:
        return None
    start, end = all_pay_dates[0], all_pay_dates[-1]
    all_days: list = []
    d = start
    while d <= end:
        all_days.append(d)
        d += timedelta(days=1)
    clf_rows = _build_clf_rows(inv, rec, all_days, holidays)
    pipe_clf = _fit_clf(clf_rows)
    pipe_reg = _fit_reg(clf_rows)
    return pipe_clf, pipe_reg

def predict_customer_payments(pipe_clf, inv, rec, pred_date, holidays=None, pipe_reg=None):
    as_of = pred_date - timedelta(days=1)
    X = build_customer_features(inv, rec, as_of, pred_date, holidays)
    if X.empty or pipe_clf is None:
        return pd.DataFrame(columns=["customer_code", "customer", "outstanding", "p_pay", "exp_pay"])
    X = X[pd.to_numeric(X["outstanding"], errors="coerce").fillna(0.0) > 0].copy()
    if X.empty:
        return pd.DataFrame(columns=["customer_code", "customer", "outstanding", "p_pay", "exp_pay"])
    feat_cols = [c for c in X.columns if c not in ("y_clf", "y_amt")]
    prob = pipe_clf.predict_proba(X[feat_cols])[:, 1]
    out  = X[["customer_code", "customer", "outstanding"]].copy()
    out["p_pay"] = prob

    if pipe_reg is not None:
        base_amt = np.maximum(pipe_reg.predict(X[feat_cols]), 0.0)
    else:
        if not rec.empty:
            daily_totals   = rec.groupby(["customer_code", "mr_date"])["paid"].sum()
            cust_hist_mean = daily_totals.groupby("customer_code").mean()
            global_mean    = float(daily_totals.mean())
        else:
            cust_hist_mean = pd.Series(dtype=float)
            global_mean    = 0.0
        outstanding_arr = pd.to_numeric(out["outstanding"], errors="coerce").fillna(0.0).values
        hist_mu = np.array([
            cust_hist_mean.get(cc, global_mean if global_mean > 0 else 0.0)
            for cc in out["customer_code"]
        ])
        base_amt = np.maximum(hist_mu, outstanding_arr * 0.05)

    out["exp_pay"] = prob * base_amt

    return out.sort_values("p_pay", ascending=False).reset_index(drop=True)
