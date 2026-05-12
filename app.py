import os, json
from datetime import date, datetime
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import joblib
import math

from rr_module import read_rr_master_excel, train_rr_model_from_master, predict_rr_cashflow_from_master, derive_cash_type
from credit_module import prep_df, detect_columns, read_workbook, build_invoices_and_receipts, train_customer_models, predict_customer_payments
from bg_module import load_bg
from x_balance_module import load_x_balance

from so_module import load_sales_order, sales_order_payments
from ro_module import load_ro, ro_payments
from mr_module import load_mr, mr_collected_for_date, mr_snapshot_time, mr_collected_by_type
from holiday_module import load_holiday_master, build_global_holiday_set, is_default_closed, BRANCH_STATE_MAP, build_holiday_set, build_national_holiday_set

st.set_page_config(page_title="SAIL Kuber", layout="wide")

LOG_PATH = "results/predictions_log.xlsx"
CRORE               = 10_000_000.0
CREDIT_TUNE_WORKING = 2.2
CREDIT_TUNE_HOLIDAY = 0.4

def _fmt_cr(v: float) -> str:
    return f"{float(v) / CRORE:,.2f} Cr"

def _to_cr_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    for c in df.columns:
        if "₹" in c and "Cr" not in c:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) / CRORE
            if "(₹)" in c:
                rename[c] = c.replace("(₹)", "(₹ Cr)")
            else:
                rename[c] = c.replace("₹", "₹ Cr")
    return df.rename(columns=rename)

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _save_prediction_log(R, manual_cr: float):
    row = {
        "date":                   str(R["date"]),
        "saved_at":               datetime.now().isoformat(timespec="seconds"),
        "manual_prediction_cr":   float(manual_cr),
        "model_total_cr":         float(R["total"])      / CRORE,
        "rr_total_cr":            float(R["rr_total"])   / CRORE,
        "credit_total_cr":        float(R["credit_total"])/ CRORE,
        "railway_total_cr":       float(R["railway_total"])/ CRORE,
        "so_total_cr":            float(R["so_total"])   / CRORE,
        "ro_file_total_cr":       float(R["ro_file_total"])/ CRORE,
        "mr_collected_cr":        float(R.get("mr_collected", 0.0)) / CRORE,
        "yet_to_collect_cr":      float(R.get("yet_to_collect", 0.0)) / CRORE,
        "mr_as_of_time":          R.get("mr_as_of", ""),
    }
    _ensure_dir("results")
    if os.path.exists(LOG_PATH):
        existing = pd.read_excel(LOG_PATH, engine="openpyxl")
    else:
        existing = pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_excel(LOG_PATH, index=False, engine="openpyxl")

def _latest_model_path(folder, prefix, exts):
    if not os.path.isdir(folder):
        return None
    files = []
    for f in os.listdir(folder):
        if not f.startswith(prefix):
            continue
        if any(f.lower().endswith(e) for e in exts):
            files.append(f)
    if not files:
        return None
    files.sort()
    return os.path.join(folder, files[-1])



def _save_credit_model(pipe_clf, pipe_reg, meta):
    _ensure_dir("models/credit")
    tag = _ts()
    model_path = f"models/credit/credit_model_{tag}.joblib"
    meta_path = f"models/credit/credit_model_{tag}.json"
    joblib.dump({"clf": pipe_clf, "reg": pipe_reg}, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    return model_path

def _load_latest_credit_model():
    p = _latest_model_path("models/credit", "credit_model_", [".joblib"])
    if not p:
        return None
    return joblib.load(p)

def _save_rr_model(model_data, meta):
    _ensure_dir("models/rr")
    tag = _ts()
    model_path = f"models/rr/rr_model_{tag}.joblib"
    meta_path = f"models/rr/rr_model_{tag}.json"
    joblib.dump(model_data, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    return model_path

def _load_latest_rr_model():
    p = _latest_model_path("models/rr", "rr_model_", [".joblib"])
    if not p:
        return None
    return joblib.load(p)

def _safe_df(df):
    
    x = df.copy()
    for c in x.columns:
        if np.issubdtype(x[c].dtype, np.number):
            x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        elif x[c].dtype == object or str(x[c].dtype) == "object":
            
            x[c] = x[c].apply(lambda v:
                v.isoformat() if hasattr(v, "isoformat")
                else ("" if v is None or (isinstance(v, float) and math.isnan(v))
                      else str(v))
            )

    for c in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = x[c].dt.strftime("%Y-%m-%d").fillna("")
        elif str(x[c].dtype).startswith("datetime"):
            x[c] = x[c].astype(str)
    return x

def _donut(df, value_col, label_col, title):
    df = _safe_df(df)
    base = alt.Chart(df).encode(theta=alt.Theta(field=value_col, type="quantitative"), color=alt.Color(field=label_col, type="nominal"),
                                tooltip=[label_col, alt.Tooltip(value_col, format=",.0f")])
    st.altair_chart(base.mark_arc(innerRadius=60, outerRadius=110).properties(title=title), use_container_width=True)

def _waterfall(df_steps, title):
    
    df_steps = _safe_df(df_steps.copy())
    df_steps["end"] = df_steps["amount"].cumsum()
    df_steps["start"] = df_steps["end"] - df_steps["amount"]
    bars = alt.Chart(df_steps).mark_bar().encode(x=alt.X("step:N", sort=list(df_steps["step"])), y=alt.Y("start:Q"), y2=alt.Y2("end:Q"),
                                                 tooltip=["step:N", alt.Tooltip("amount:Q", format=",.0f")])
    st.altair_chart(bars.properties(title=title), use_container_width=True)

def _progress_bar(collected: float, total: float):
    if total <= 0:
        return
    pct = min(100.0, collected / total * 100.0)
    rem = 100.0 - pct
    st.markdown(f"""
<div style="background:#2d2d2d;border-radius:8px;height:28px;overflow:hidden;margin:6px 0 2px 0;display:flex;">
  <div style="width:{pct:.1f}%;background:#00c853;height:100%;"></div>
  <div style="width:{rem:.1f}%;background:#f9a825;height:100%;"></div>
</div>
<p style="margin:2px 0 10px 0;font-size:0.85em;color:#aaa;">
  <span style="color:#00c853;">&#9646; Collected {pct:.1f}%</span>
  &nbsp;&nbsp;
  <span style="color:#f9a825;">&#9646; Yet to Collect {rem:.1f}%</span>
</p>
""", unsafe_allow_html=True)


def _rr_heatmap(rr_daily: pd.DataFrame, start_date):
    df = rr_daily.head(28).copy()
    df["date"]     = pd.to_datetime(df["date"])
    start_ts       = pd.Timestamp(start_date)
    df["week"]     = ((df["date"] - start_ts).dt.days // 7 + 1).astype(str).radd("Week ")
    DOW            = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["dow"]      = df["date"].dt.strftime("%a")
    df["label"]    = df["date"].dt.strftime("%d %b")
    df["amount_cr"] = df["rr_final_cash"] / CRORE
    df["is_today"] = df["date"] == start_ts
    df["opacity"]  = df["day_type"].apply(lambda t: 1.0 if t == "WORKING" else 0.35)
    base = alt.Chart(df).encode(
        x=alt.X("dow:O", sort=DOW, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("week:O", title=None, sort=["Week 1", "Week 2", "Week 3", "Week 4"]),
        tooltip=["label:N", alt.Tooltip("amount_cr:Q", format=".2f", title="₹ Cr"), "day_type:N"]
    )
    cells = base.mark_rect().encode(
        color=alt.Color("amount_cr:Q", scale=alt.Scale(scheme="blues"), title="₹ Cr"),
        opacity=alt.Opacity("opacity:Q", scale=alt.Scale(domain=[0, 1]), legend=None)
    )
    today_outline = base.transform_filter(
        alt.datum.is_today == True
    ).mark_rect(filled=False, stroke="#ff4b4b", strokeWidth=3)
    st.altair_chart((cells + today_outline).properties(
        title="RR Surrender - 4-Week Horizon", height=160
    ), use_container_width=True)

_RAILWAY_EXACT = {"accounts manager", "stores dept"}

def _is_railway_customer(name: str) -> bool:
    n = str(name).lower().strip()
    return n in _RAILWAY_EXACT or "railway" in n

def _validate_columns(df: pd.DataFrame, required: list, file_label: str) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"**{file_label}**: Missing required column(s): {', '.join(missing)}. Check the uploaded file.")
        return False
    return True

def _load_holidays(hol_up):
    if hol_up is None:
        return set()
    try:
        master_df = load_holiday_master(hol_up)
        return build_global_holiday_set(master_df)
    except Exception:
        return set()

def _load_national_holidays(hol_up) -> set:
    if hol_up is None:
        return set()
    try:
        master_df = load_holiday_master(hol_up)
        return build_national_holiday_set(master_df)
    except Exception:
        return set()

def _load_branch_holidays(hol_up, inv: pd.DataFrame) -> set:
    if hol_up is None:
        return set()
    try:
        master_df = load_holiday_master(hol_up)
        branches = inv["business_area"].dropna().unique() if "business_area" in inv.columns else []
        hols = set()
        for br in branches:
            br_str = str(br).strip()
            if br_str in BRANCH_STATE_MAP:
                hols |= build_holiday_set(br_str, master_df)
        return hols if hols else build_global_holiday_set(master_df)
    except Exception:
        return set()

_logo_l, _tab_mid, _logo_r = st.columns([1, 8, 1])
with _logo_l:
    st.image("logo1.jpg")
with _logo_r:
    st.image("logo2.jpg")
with _tab_mid:
    tabs = st.tabs(["Summary View", "RR Surrender", "Credit Customers", "Sales Order + Release Order", "Training", "File Inputs"])
    tab_summary, tab_rr, tab_credit, tab_so_ro, tab_train, tab_files = tabs

    if "railway_override_cr" not in st.session_state:
        st.session_state.railway_override_cr = 0.0
    if "railway_override_active" not in st.session_state:
        st.session_state.railway_override_active = False
    if "results" not in st.session_state:
        st.session_state.results = None
    if "awaiting_manual_input" not in st.session_state:
        st.session_state.awaiting_manual_input = False
    
    with tab_files:
        st.subheader("Upload Input Files")
        rr_master_up = st.file_uploader("RR Pending Report (ZFIRH008A)", type=["xlsx"], key="rr_master")
        credit_up = st.file_uploader("Credit Sales Report - last 4 months (ZFIRH013)", type=["xlsx"], key="credit")
        so_up = st.file_uploader("Sales Order Report - all SOs linked/unlinked (ZSDR014)", type=["xlsx"], key="so")
        ro_up = st.file_uploader("Release Order Report (ZSDR026)", type=["xlsx"], key="ro_file")
        bg_up = st.file_uploader("BG Utilisation Report (ZFIRH075N)", type=["xlsx"], key="bg")
        x_balance_up = st.file_uploader("Ledger Balance in X (FBL5N)", type=["xlsx"], key="x_balance")
        mr_up = st.file_uploader("MR Report for the day (ZFIRH076)", type=["xlsx", "xls", "XLSX"], key="mr")
        hol_up = st.file_uploader("Holiday Master", type=["xlsx", "csv"], key="hol")

    with tab_train:
        st.subheader("Training")
        colA, colB = st.columns(2)
        with colA:
            train_rr = st.button("Train RR Model", type="primary", use_container_width=True)
            if train_rr:
                if rr_master_up is None:
                    st.error("Upload RR Master Excel in File Inputs.")
                else:
                    with st.spinner("Training RR model…"):
                        holidays      = _load_holidays(hol_up)
                        hol_master_df = load_holiday_master(hol_up) if hol_up is not None else None
                        rr_master_df  = read_rr_master_excel(rr_master_up)
                        if not _validate_columns(rr_master_df, ["retirement_date","invoice_date","invoice_amount","customer_code"], "RR Master Excel"):
                            st.stop()
                        model_data    = train_rr_model_from_master(
                                            rr_master_df,
                                            holidays=holidays,
                                            holiday_master_df=hol_master_df,
                                        )
                        meta = {"trained_at":   datetime.now().isoformat(timespec="seconds"),
                                "rows_total":   int(len(rr_master_df)),
                                "rows_history": int(rr_master_df["retirement_date"].notna().sum()),
                                "rows_pending": int(rr_master_df["retirement_date"].isna().sum()),
                                "holidays_used": len(holidays)}
                        p = _save_rr_model(model_data, meta)
                    st.success(f"Saved: {p}")
        with colB:
            train_credit = st.button("Train Credit Model", type="primary", use_container_width=True)
            if train_credit:
                if credit_up is None:
                    st.error("Upload Credit Sales Workbook in File Inputs.")
                else:
                    with st.spinner("Training Credit model…"):
                        raw_flat = read_workbook(credit_up)
                        raw_flat = prep_df(raw_flat)
                        m        = detect_columns(raw_flat)
                        _missing_credit = [k for k in ["invoice_date","customer_code"] if m.get(k) is None]
                        if _missing_credit:
                            st.error(f"**Credit Workbook**: Could not detect column(s): {', '.join(_missing_credit)}. Check the file structure.")
                            st.stop()
                        inv, rec = build_invoices_and_receipts(raw_flat, m)
                        result   = train_customer_models(inv, rec, holidays=_load_branch_holidays(hol_up, inv))
                    if result is None or result[0] is None:
                        st.error("Credit model training failed - insufficient data.")
                    else:
                        pipe_clf, pipe_reg = result
                        meta = {"trained_at":   datetime.now().isoformat(timespec="seconds"),
                                "invoice_rows": int(len(inv)),
                                "receipt_rows": int(len(rec)),
                                "has_reg":      pipe_reg is not None}
                        p = _save_credit_model(pipe_clf, pipe_reg, meta)
                        st.success(f"Saved: {p}")
    
    with tab_summary:
        st.subheader("Summary View")
        pred_date = st.date_input("Prediction date", value=date.today())
        run = st.button("Predict", type="primary")
        if run:
            st.session_state.railway_override_cr = 0.0
            st.session_state.railway_override_active = False
            rr_model = _load_latest_rr_model()
            credit_art = _load_latest_credit_model()
            
            
            if rr_model is None:
                st.error("RR model not found. Train RR model first.")
                st.stop()
            if credit_art is None:
                st.error("Credit model not found. Train Credit model first.")
                st.stop()
            
            
            if rr_master_up is None or credit_up is None:
                st.error("Upload RR Master and Credit Sales Workbook files.")
                st.stop()




            D = pred_date
            rr_master_df = read_rr_master_excel(rr_master_up)
            if not _validate_columns(rr_master_df, ["retirement_date","invoice_date","invoice_amount","customer_code"], "RR Master Excel"):
                st.stop()
            holidays = _load_holidays(hol_up)
            rr_daily, rr_cust, rr_inv_details = predict_rr_cashflow_from_master(rr_master_df, rr_model, start_date=D, horizon_days=31, holidays=holidays)
            rr_total = float(rr_daily.loc[rr_daily["date"] == pd.Timestamp(D), "rr_final_cash"].sum()) if rr_daily is not None else 0.0
            rr_today_by_code = pd.DataFrame(columns=["Customer Code","RR Surrender _Rest of Day (₹)"])
            if rr_cust is not None and not rr_cust.empty:
                t = rr_cust[rr_cust["date"] == pd.Timestamp(D)].copy()
                t["customer_code"] = t["customer_code"].astype(str).str.strip()
                rr_today_by_code = t.groupby("customer_code", as_index=False)["rr_final_cash"].sum().rename(columns={"customer_code":"Customer Code","rr_final_cash":"RR Surrender _Rest of Day (₹)"})
            if bg_up is not None:
                bg_df = load_bg(bg_up, pred_date=D).rename(columns={"Cust. Code":"Customer Code","available_credit":"Available Credit Limit (₹)"})
                bg_df["Customer Code"] = bg_df["Customer Code"].astype(str).str.strip()
                if bg_df.empty:
                    st.error("**BG Utilisation file**: No valid rows loaded. Check that customer code and balance columns are present.")
                    st.stop()
            else:
                bg_df = pd.DataFrame(columns=["Customer Code","Available Credit Limit (₹)"])
            bg_codes = set(bg_df["Customer Code"].astype(str).unique())
            raw_flat = read_workbook(credit_up)
            raw_flat = prep_df(raw_flat)
            m        = detect_columns(raw_flat)
            _missing_credit = [k for k in ["invoice_date","customer_code"] if m.get(k) is None]
            if _missing_credit:
                st.error(f"**Credit Workbook**: Could not detect column(s): {', '.join(_missing_credit)}. Check the file structure.")
                st.stop()
            inv, rec = build_invoices_and_receipts(raw_flat, m)
            credit_holidays = _load_branch_holidays(hol_up, inv)
            base = predict_customer_payments(credit_art["clf"], inv, rec, D, holidays=credit_holidays, pipe_reg=credit_art.get("reg"))
            _national_hols = _load_national_holidays(hol_up)
            _factor = CREDIT_TUNE_HOLIDAY if (is_default_closed(D) or D in _national_hols) else CREDIT_TUNE_WORKING
            base["exp_pay"] = base["exp_pay"] * _factor
            base = base.rename(columns={"customer_code":"Customer Code","customer":"Customer","outstanding":"Total Outstanding as of Yesterday (₹)","p_pay":"Probability of Payment","exp_pay":"Predicted Collection _Rest of Day (₹)"})
            base["Customer Code"] = base["Customer Code"].astype(str).str.strip()
            so_total, so_cash_total, so_credit_total, so_by_code = 0.0, 0.0, 0.0, pd.DataFrame(columns=["customer_code","amount","payment_type"])
            if so_up is not None:
                so_df = load_sales_order(so_up)
                if "_amount" not in so_df.columns:
                    st.error("**Sales Order file**: Could not compute order amounts _check for 'SO Rate', 'Order Qty', or 'MR/CAM Amount' columns.")
                    st.stop()
                so_total, so_cash_total, so_credit_total, so_by_code = sales_order_payments(so_df)
            ro_file_total, ro_cash_total, ro_credit_total, ro_by_code = 0.0, 0.0, 0.0, pd.DataFrame(columns=["customer_code","amount","payment_type"])




            if ro_up is not None:
                ro_df = load_ro(ro_up)
                for _req_col in ["RO Value", "Paymnt Amt"]:
                    if _req_col not in ro_df.columns:
                        st.error(f"**Release Order file**: Missing required column '{_req_col}'.")
                        st.stop()
                ro_file_total, ro_cash_total, ro_credit_total, ro_by_code = ro_payments(ro_df)
            so_by_code_detail = so_by_code.rename(columns={"customer_code":"Customer Code","amount":"S.O _Rest of Day (₹)","payment_type":"S.O Payment Type"})
            so_agg = so_by_code.groupby("customer_code", as_index=False)["amount"].sum().rename(columns={"customer_code":"Customer Code","amount":"S.O _Rest of Day (₹)"})
            so_credit_agg = so_by_code[so_by_code["payment_type"]=="CREDIT"].groupby("customer_code",as_index=False)["amount"].sum().rename(columns={"customer_code":"Customer Code","amount":"_so_credit"})
            so_cash_agg   = so_by_code[so_by_code["payment_type"]=="CASH"].groupby("customer_code",as_index=False)["amount"].sum().rename(columns={"customer_code":"Customer Code","amount":"_so_cash"})
            ro_by_code_detail = ro_by_code.rename(columns={"customer_code":"Customer Code","amount":"R.O _Rest of Day (₹)","payment_type":"R.O Payment Type"})
            ro_agg = ro_by_code.groupby("customer_code", as_index=False)["amount"].sum().rename(columns={"customer_code":"Customer Code","amount":"R.O _Rest of Day (₹)"})
            _ro_has_type = not ro_by_code.empty and "payment_type" in ro_by_code.columns



            ro_credit_agg = ro_by_code[ro_by_code["payment_type"]=="CREDIT"].groupby("customer_code",as_index=False)["amount"].sum().rename(columns={"customer_code":"Customer Code","amount":"_ro_credit"}) if _ro_has_type else pd.DataFrame(columns=["Customer Code","_ro_credit"])
            ro_cash_agg   = ro_by_code[ro_by_code["payment_type"]=="CASH"].groupby("customer_code",as_index=False)["amount"].sum().rename(columns={"customer_code":"Customer Code","amount":"_ro_cash"}) if _ro_has_type else pd.DataFrame(columns=["Customer Code","_ro_cash"])
            _rr_today = (rr_inv_details[
                pd.to_datetime(rr_inv_details["pred_retirement_date"]).dt.date == D
            ].copy() if not rr_inv_details.empty else
                pd.DataFrame(columns=["pred_retirement_date","customer_code","invoice_amount","sale_type"]))
            if not _rr_today.empty:
                _n = pd.to_numeric(_rr_today["customer_code"], errors="coerce")
                _rr_today["customer_code"] = (
                    _n.apply(lambda x: str(int(x)) if pd.notna(x) else "")
                      .where(lambda s: s != "", _rr_today["customer_code"].astype(str).str.strip())
                )
                def _rr_classify(row):
                    if derive_cash_type(row["sale_type"]) == "CREDIT":
                        return "CREDIT"
                    return "CREDIT" if row["customer_code"] in bg_codes else "CASH"
                _rr_today["_rr_type"] = _rr_today.apply(_rr_classify, axis=1)
                rr_credit_agg = (_rr_today[_rr_today["_rr_type"]=="CREDIT"]
                    .groupby("customer_code", as_index=False)["invoice_amount"].sum()
                    .rename(columns={"customer_code":"Customer Code","invoice_amount":"_rr_credit"}))
                rr_cash_agg = (_rr_today[_rr_today["_rr_type"]=="CASH"]
                    .groupby("customer_code", as_index=False)["invoice_amount"].sum()
                    .rename(columns={"customer_code":"Customer Code","invoice_amount":"_rr_cash"}))
            else:
                rr_credit_agg = pd.DataFrame(columns=["Customer Code","_rr_credit"])
                rr_cash_agg   = pd.DataFrame(columns=["Customer Code","_rr_cash"])
            so_ro_by_code = so_agg.merge(ro_agg, on="Customer Code", how="outer")
            if so_ro_by_code.empty:
                so_ro_by_code = pd.DataFrame(columns=["Customer Code","S.O _Rest of Day (₹)","R.O _Rest of Day (₹)"])
            so_ro_by_code["S.O _Rest of Day (₹)"] = pd.to_numeric(so_ro_by_code.get("S.O _Rest of Day (₹)", 0.0), errors="coerce").fillna(0.0)
            so_ro_by_code["R.O _Rest of Day (₹)"] = pd.to_numeric(so_ro_by_code.get("R.O _Rest of Day (₹)", 0.0), errors="coerce").fillna(0.0)
            so_ro_by_code["S.O + R.O _Rest of Day (₹)"] = so_ro_by_code["S.O _Rest of Day (₹)"] + so_ro_by_code["R.O _Rest of Day (₹)"]
            x_balance_df = pd.DataFrame(columns=["Customer Code","Balance in X (₹)"])



            if x_balance_up is not None:
                xb = load_x_balance(x_balance_up).rename(columns={"customer_code":"Customer Code","balance_x":"Balance in X (₹)"})
                x_balance_df = xb
            merged = base.merge(bg_df[["Customer Code","Available Credit Limit (₹)"]], on="Customer Code", how="left")
            merged["Available Credit Limit (₹)"] = pd.to_numeric(merged["Available Credit Limit (₹)"], errors="coerce").fillna(0.0)
            merged = merged.merge(so_ro_by_code[["Customer Code","S.O _Rest of Day (₹)","R.O _Rest of Day (₹)","S.O + R.O _Rest of Day (₹)"]], on="Customer Code", how="left")
            for c in ["S.O _Rest of Day (₹)","R.O _Rest of Day (₹)","S.O + R.O _Rest of Day (₹)"]:
                merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)
            merged = merged.merge(x_balance_df, on="Customer Code", how="left")
            merged["Balance in X (₹)"] = pd.to_numeric(merged.get("Balance in X (₹)", 0.0), errors="coerce").fillna(0.0)
            merged = merged.merge(so_credit_agg, on="Customer Code", how="left")
            merged["_so_credit"] = pd.to_numeric(merged["_so_credit"], errors="coerce").fillna(0.0)
            merged = merged.merge(so_cash_agg, on="Customer Code", how="left")
            merged["_so_cash"] = pd.to_numeric(merged["_so_cash"], errors="coerce").fillna(0.0)
            merged = merged.merge(ro_credit_agg, on="Customer Code", how="left")
            merged["_ro_credit"] = pd.to_numeric(merged["_ro_credit"], errors="coerce").fillna(0.0)
            merged = merged.merge(ro_cash_agg, on="Customer Code", how="left")
            merged["_ro_cash"] = pd.to_numeric(merged["_ro_cash"], errors="coerce").fillna(0.0)
            merged = merged.merge(rr_credit_agg, on="Customer Code", how="left")
            merged["_rr_credit"] = pd.to_numeric(merged["_rr_credit"], errors="coerce").fillna(0.0)
            merged = merged.merge(rr_cash_agg, on="Customer Code", how="left")
            merged["_rr_cash"] = pd.to_numeric(merged["_rr_cash"], errors="coerce").fillna(0.0)
            _rwy_flag = merged["Customer"].apply(_is_railway_customer)
            railway_customers = merged[_rwy_flag].reset_index(drop=True)
            merged = merged[~_rwy_flag].reset_index(drop=True)
            _A_credit = merged["_so_credit"] + merged["_ro_credit"] + merged["_rr_credit"]
            _A_cash   = merged["_so_cash"]   + merged["_ro_cash"]   + merged["_rr_cash"]
            _D = merged["Balance in X (₹)"]
            _E = merged["Available Credit Limit (₹)"]
            _F = merged["Predicted Collection _Rest of Day (₹)"]
            _has_bg      = _E > 0
            _final_bg    = np.maximum(0.0, _A_cash + np.maximum(_F - _D, _A_credit - _D - _E - _F))
            _final_no_bg = np.maximum(0.0, _F + _A_credit + _A_cash - _D)
            merged["Final Expected Collection _Rest of Day (₹)"] = np.where(_has_bg, _final_bg, _final_no_bg)
            _rA_credit = railway_customers["_so_credit"] + railway_customers["_ro_credit"] + railway_customers["_rr_credit"]
            _rA_cash   = railway_customers["_so_cash"]   + railway_customers["_ro_cash"]   + railway_customers["_rr_cash"]
            _rD = railway_customers["Balance in X (₹)"]
            railway_customers["Final Expected Collection _Rest of Day (₹)"] = (
                np.maximum(0.0, _rA_credit + _rA_cash - _rD)
            )
            _DISPLAY_THRESHOLD_HIGH = 0.60
            _DISPLAY_THRESHOLD_WATCH = 0.40



            credit_total = float(merged["Final Expected Collection _Rest of Day (₹)"].sum()) if not merged.empty else 0.0
            railway_predicted_total = (
                float(railway_customers["Final Expected Collection _Rest of Day (₹)"].sum())
                if not railway_customers.empty else 0.0
            )
            ro_manual_total = st.session_state.railway_override_cr * CRORE
            railway_total = (
                ro_manual_total
                if st.session_state.railway_override_active
                else railway_predicted_total
            )
            _all_known_codes = (
                set(merged["Customer Code"].astype(str))
                | set(railway_customers["Customer Code"].astype(str))
            )
            
            _xbal_map = (
                x_balance_df.set_index("Customer Code")["Balance in X (₹)"].to_dict()
                if not x_balance_df.empty else {}
            )
            def _xnet(df, code_col, amount_col):
                """Deduct each customer's X balance from their amount, floor at 0, sum."""
                if df.empty:
                    return df.assign(**{amount_col: 0.0}), 0.0
                df = df.copy()
                xb = df[code_col].astype(str).map(_xbal_map).fillna(0.0)
                df[amount_col] = (
                    pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
                    .sub(xb).clip(lower=0.0)
                )
                return df, float(df[amount_col].sum())

            
            _orphan_rr_credit_df = (
                rr_credit_agg[~rr_credit_agg["Customer Code"].isin(_all_known_codes)].copy()
                if not rr_credit_agg.empty
                else pd.DataFrame(columns=["Customer Code", "_rr_credit"])
            )
            _, _rr_credit_orphan_total = _xnet(_orphan_rr_credit_df, "Customer Code", "_rr_credit")

            
            _orphan_rr_cash_df = (
                rr_cash_agg[~rr_cash_agg["Customer Code"].isin(_all_known_codes)].copy()
                if not rr_cash_agg.empty
                else pd.DataFrame(columns=["Customer Code", "_rr_cash"])
            )
            _orphan_rr_cash_df, _rr_cash_display = _xnet(_orphan_rr_cash_df, "Customer Code", "_rr_cash")

            
            if not so_agg.empty:
                _orphan_so = so_agg[~so_agg["Customer Code"].isin(_all_known_codes)].copy()
                _known_so_sum = float(so_agg[so_agg["Customer Code"].isin(_all_known_codes)]["S.O _Rest of Day (₹)"].sum())
                _, _orphan_so_net = _xnet(_orphan_so, "Customer Code", "S.O _Rest of Day (₹)")
                so_total = _known_so_sum + _orphan_so_net
           

            
            if not ro_agg.empty:
                _orphan_ro = ro_agg[~ro_agg["Customer Code"].isin(_all_known_codes)].copy()
                _known_ro_sum = float(ro_agg[ro_agg["Customer Code"].isin(_all_known_codes)]["R.O _Rest of Day (₹)"].sum())
                _, _orphan_ro_net = _xnet(_orphan_ro, "Customer Code", "R.O _Rest of Day (₹)")
                ro_file_total = _known_ro_sum + _orphan_ro_net
            

            credit_total_final = credit_total + _rr_credit_orphan_total
            
            rr_today_by_code = (
                _orphan_rr_cash_df.rename(columns={"_rr_cash": "RR Surrender _Rest of Day (₹)"})
                [["Customer Code", "RR Surrender _Rest of Day (₹)"]]
                .reset_index(drop=True)
            ) if not _orphan_rr_cash_df.empty else pd.DataFrame(
                columns=["Customer Code", "RR Surrender _Rest of Day (₹)"])
            mr_df = load_mr(mr_up, pred_date=D) if mr_up is not None else None
            mr_collected = mr_collected_for_date(mr_df, D)
            mr_as_of     = mr_snapshot_time(mr_df, D)
            mr_railway_collected, mr_nonrailway_collected = mr_collected_by_type(mr_df, D) if mr_df is not None else (0.0, 0.0)
            model_prediction = _rr_cash_display + credit_total_final + railway_total + so_total + ro_file_total
            if mr_up is not None:
                yet_to_collect = model_prediction
                total = mr_collected + model_prediction
            else:
                yet_to_collect = model_prediction
                total = model_prediction
            _drop_internal = ["_rr_credit","_rr_cash","_so_credit","_so_cash","_ro_credit","_ro_cash","S.O _Rest of Day (₹)","R.O _Rest of Day (₹)","S.O + R.O _Rest of Day (₹)","Balance in X (₹)"]
            st.session_state.results = {
                "date": D,
                "rr_total": _rr_cash_display,
                "rr_daily": rr_daily,
                "rr_today_by_code": rr_today_by_code,
                "credit_total": credit_total_final,
                "credit_customers": merged.drop(columns=[c for c in _drop_internal if c in merged.columns], errors="ignore").sort_values("Final Expected Collection _Rest of Day (₹)", ascending=False).reset_index(drop=True),
                "railway_customers": railway_customers.drop(columns=[c for c in _drop_internal if c in railway_customers.columns], errors="ignore").sort_values("Final Expected Collection _Rest of Day (₹)", ascending=False).reset_index(drop=True),
                "railway_predicted_total": railway_predicted_total,
                "railway_total": railway_total,
                "so_total": so_total,
                "so_cash_total": so_cash_total,
                "so_credit_total": so_credit_total,
                "so_by_code_detail": so_by_code_detail,
                "ro_file_total": ro_file_total,
                "ro_cash_total": ro_cash_total,
                "ro_credit_total": ro_credit_total,
                "ro_by_code_detail": ro_by_code_detail,
                "so_ro_by_code": so_ro_by_code,
                "ro_manual_total": ro_manual_total,
                "mr_collected":             mr_collected,
                "mr_railway_collected":     mr_railway_collected,
                "mr_nonrailway_collected":  mr_nonrailway_collected,
                "yet_to_collect":           yet_to_collect,
                "mr_as_of":                 mr_as_of,
                "model_prediction":  model_prediction,
                "threshold_high":    _DISPLAY_THRESHOLD_HIGH,
                "threshold_watch":   _DISPLAY_THRESHOLD_WATCH,
                "total": total
            }
            st.session_state.awaiting_manual_input = True
            st.rerun()
        if st.session_state.awaiting_manual_input:
            with st.form("manual_prediction_form"):
                st.subheader("Enter Your Prediction")
                st.caption(f"Prediction date: {st.session_state.results['date']}")
                manual_cr = st.number_input("Your Manual Prediction - Total Cash Expected Today (₹ Cr)", min_value=0.0, step=0.01, format="%.2f")
                submitted = st.form_submit_button("Submit & View Model Prediction", type="primary")
            if submitted:
                _save_prediction_log(st.session_state.results, manual_cr)
                st.session_state.awaiting_manual_input = False
                st.rerun()
        elif st.session_state.results is None:
            st.info("Train models in Training tab, upload files, select date, click Predict.")
        else:
            R = st.session_state.results
            _live_rwy = (
                st.session_state.railway_override_cr * CRORE
                if st.session_state.railway_override_active
                else float(R["railway_predicted_total"])
            )
            _live_model = float(R["rr_total"]) + float(R["credit_total"]) + _live_rwy + float(R["so_total"]) + float(R["ro_file_total"])
            _live_total = float(R.get("mr_collected", 0.0)) + _live_model if mr_up is not None else _live_model
            _date_label = R["date"].strftime("%d %b %Y") if hasattr(R["date"], "strftime") else str(R["date"])
            if mr_up is not None:
                st.metric(f"Expected on {_date_label}", _fmt_cr(_live_total))
                _as_of = R.get("mr_as_of", "")
                _collected_label = f"Collected as of {_as_of}" if _as_of else "Already Collected"
                _mr_c1, _mr_c2 = st.columns(2)
                _mr_c1.metric(_collected_label, _fmt_cr(R.get("mr_collected", 0.0)))
                _mr_c2.metric("Yet to Collect – Rest of Day", _fmt_cr(_live_model))
                _mr_sub1, _mr_sub2, _mr_pad1, _mr_pad2 = st.columns(4)
                _mr_sub1.metric("↳ Railways", _fmt_cr(R.get("mr_railway_collected", 0.0)))
                _mr_sub2.metric("↳ Non-Railways", _fmt_cr(R.get("mr_nonrailway_collected", 0.0)))
                _progress_bar(R.get("mr_collected", 0.0), _live_total)
            else:
                st.metric(f"Expected on {_date_label}", _fmt_cr(_live_model))

            _rwy_col_input, _rwy_col_btn = st.columns([5, 1])
            with _rwy_col_input:
                _rwy_str = st.text_input(
                    "Enter expected collection from Indian Railways for rest of the day (₹ Cr) - leave 0 for model prediction",
                    value=f"{st.session_state.railway_override_cr:.2f}",
                    key="rwy_override_input"
                )
            with _rwy_col_btn:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("Override", key="rwy_override_btn"):
                    try:
                        st.session_state.railway_override_cr = max(0.0, float(_rwy_str))
                    except ValueError:
                        st.session_state.railway_override_cr = 0.0
                    st.session_state.railway_override_active = True
                    st.rerun()
            _rwy_hint = float(R.get("railway_predicted_total", 0.0))
            if st.session_state.railway_override_active:
                st.caption(f"Override active: {_fmt_cr(st.session_state.railway_override_cr * CRORE)}  |  Model predicted: {_fmt_cr(_rwy_hint)}")
            else:
                st.caption(f"Using model prediction: {_fmt_cr(_rwy_hint)}")
            breakup = pd.DataFrame([
                {"Component":"RR Surrender","Amount (₹ Cr)":float(R["rr_total"]) / CRORE},
                {"Component":"Credit Customers","Amount (₹ Cr)":float(R["credit_total"]) / CRORE},
                {"Component":"Indian Railways","Amount (₹ Cr)":_live_rwy / CRORE},
                {"Component":"Sales Order (S.O)","Amount (₹ Cr)":float(R["so_total"]) / CRORE},
                {"Component":"Release Order (R.O)","Amount (₹ Cr)":float(R["ro_file_total"]) / CRORE},
            ])
            a, b = st.columns([1.2, 1])
            with a:
                st.dataframe(_safe_df(breakup), use_container_width=True)
            with b:
                _donut(breakup, "Amount (₹ Cr)", "Component", "Rest of Day _Breakup")
            steps = breakup.copy()
            steps["step"] = steps["Component"]
            steps["amount"] = steps["Amount (₹ Cr)"]
            _waterfall(steps[["step","amount"]], "Rest of Day _How Total Builds Up")
    
    with tab_rr:
        st.subheader("RR Surrender")
        R = st.session_state.results
        if R is None:
            st.info("Run prediction from Summary View.")
        else:
            st.metric("RR Surrender _Rest of Day", _fmt_cr(R['rr_total']))
            if R.get("rr_daily") is not None and not R["rr_daily"].empty:
                _rr_heatmap(R["rr_daily"], R["date"])
            tbl = R.get("rr_today_by_code")
            if tbl is not None and not tbl.empty:
                _rr_sort_col = "RR Surrender _Rest of Day (₹)" if "RR Surrender _Rest of Day (₹)" in tbl.columns else tbl.columns[-1]
                st.dataframe(_safe_df(_to_cr_df(tbl.sort_values(_rr_sort_col, ascending=False))), use_container_width=True)
    
    with tab_credit:
        st.subheader("Credit Customers")
        R = st.session_state.results
        if R is None:
            st.info("Run prediction from Summary View.")
        
        
        
        else:
            st.metric("Credit Customers _Rest of Day Prediction", _fmt_cr(R['credit_total']))
            cc = R["credit_customers"]
            prob_col = "Probability of Payment"
            _thr_h = R.get("threshold_high", 0.70)
            _thr_w = R.get("threshold_watch", 0.50)
            high_conf = cc[cc[prob_col] >= _thr_h].reset_index(drop=True) if prob_col in cc.columns else cc
            watch     = cc[(cc[prob_col] >= _thr_w) & (cc[prob_col] < _thr_h)].reset_index(drop=True) if prob_col in cc.columns else cc.iloc[0:0]
            below     = cc[cc[prob_col] < _thr_w].reset_index(drop=True) if prob_col in cc.columns else cc.iloc[0:0]
            st.markdown(f"#### High Confidence &nbsp; `p ≥ {_thr_h:.0%}` &nbsp; - {len(high_conf)} customer(s)")
            if high_conf.empty:
                st.info("No high-confidence payers predicted today.")
            else:
                st.dataframe(_safe_df(_to_cr_df(high_conf)), use_container_width=True)
            st.markdown(f"#### Watch List &nbsp; `{_thr_w:.0%} ≤ p < {_thr_h:.0%}` &nbsp; - {len(watch)} customer(s)")
            if watch.empty:
                st.info("No watch-list customers today.")
            else:
                st.dataframe(_safe_df(_to_cr_df(watch)), use_container_width=True)
            st.markdown(f"#### All Others &nbsp; `p < {_thr_w:.0%}` &nbsp; - {len(below)} customer(s)")
            if not below.empty:
                st.dataframe(_safe_df(_to_cr_df(below)), use_container_width=True)
            st.divider()
            rwy = R.get("railway_customers", pd.DataFrame())
            rwy_pred = float(R.get("railway_predicted_total", 0.0))
            rwy_override = float(R.get("ro_manual_total", 0.0))
            rwy_used = float(R.get("railway_total", 0.0))
            st.markdown("#### Indian Railways (Excluded from Credit Total)")
            _rwy_m1, _rwy_m2 = st.columns(2)
            with _rwy_m1:
                st.metric("Predicted Collection", _fmt_cr(rwy_pred))
            with _rwy_m2:
                _override_label = f"Override Applied: {_fmt_cr(rwy_override)}" if rwy_override > 0 else "No Override - prediction used"
                st.metric("Used in Grand Total", _fmt_cr(rwy_used), delta=_override_label if rwy_override > 0 else None)
            if rwy.empty:
                st.info("No railway customers identified in credit data.")
            else:
                st.dataframe(_safe_df(_to_cr_df(rwy)), use_container_width=True)
            st.caption("To override the predicted Indian Railways total, use the override input in Summary View.")
    
    with tab_so_ro:
        st.subheader("Sales Order + Release Order")
        
        R = st.session_state.results
        if R is None:
            st.info("Run prediction from Summary View.")
        
        
        else:
            _so_c1, _so_c2, _so_c3 = st.columns(3)
            with _so_c1:
                st.metric("Sales Order (S.O) _Rest of Day", _fmt_cr(R['so_total']))
            with _so_c2:
                st.metric("S.O Cash (ZCAS)", _fmt_cr(R['so_cash_total']))
            with _so_c3:
                st.metric("S.O Credit (ZSCR)", _fmt_cr(R['so_credit_total']))
            _ro_c1, _ro_c2, _ro_c3 = st.columns(3)
            with _ro_c1:
                st.metric("Release Order (R.O) _Rest of Day", _fmt_cr(R['ro_file_total']))
            with _ro_c2:
                st.metric("R.O Cash (ZCAS)", _fmt_cr(R.get('ro_cash_total', 0.0)))
            with _ro_c3:
                st.metric("R.O Credit (ZSCR)", _fmt_cr(R.get('ro_credit_total', 0.0)))
            detail_so = R.get("so_by_code_detail")
            if detail_so is not None and not detail_so.empty:
                st.markdown("#### S.O Detail by Customer & Payment Type")
                _so_detail_sort = "S.O _Rest of Day (₹)" if "S.O _Rest of Day (₹)" in detail_so.columns else detail_so.columns[1]
                st.dataframe(_safe_df(_to_cr_df(detail_so.sort_values(_so_detail_sort, ascending=False))), use_container_width=True)
            detail_ro = R.get("ro_by_code_detail")
            if detail_ro is not None and not detail_ro.empty:
                st.markdown("#### R.O Detail by Customer & Payment Type")
                _ro_detail_sort = "R.O _Rest of Day (₹)" if "R.O _Rest of Day (₹)" in detail_ro.columns else detail_ro.columns[1]
                st.dataframe(_safe_df(_to_cr_df(detail_ro.sort_values(_ro_detail_sort, ascending=False))), use_container_width=True)

st.markdown(
    '<div style="text-align:right;color:#888;font-size:0.82em;'
    'padding:18px 0 4px 0;border-top:1px solid #333;">'
    'SAIL Kuber: A product of collective efforts by SDTD and CMO</div>',
    unsafe_allow_html=True
)