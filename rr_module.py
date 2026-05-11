import warnings
from datetime import date
from typing import Dict, Optional, Sequence, Set

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

CRORE = 10_000_000.0


TRAIN_COLS = [
    "retirement_date",
    "branch",
    "plant",
    "customer_code",
    "invoice_date",
    "sale_type",
    "rly_dest",
    "invoice_amount",
]

PRED_COLS = [
    "branch",
    "plant",
    "customer_code",
    "invoice_date",
    "sale_type",
    "rly_dest",
    "invoice_amount",
]


# ---------------------------------------------------------------------
# Branch → State mapping  (hardcoded, all 43 SAIL branches)
# ---------------------------------------------------------------------
BRANCH_STATE_MAP: Dict[str, str] = {
    "B001": "Odisha",           "B002": "Jharkhand",
    "B003": "West Bengal",      "B004": "West Bengal",
    "B006": "Bihar",            "B007": "Odisha",
    "B008": "Andhra Pradesh",   "B011": "Assam",
    "B013": "Uttar Pradesh",    "B014": "Delhi",
    "B015": "Haryana",          "B016": "Uttar Pradesh",
    "B017": "Uttar Pradesh",    "B020": "Jammu and Kashmir",
    "B022": "Punjab",           "B023": "Punjab",
    "B025": "Karnataka",        "B027": "Kerala",
    "B028": "Tamil Nadu",       "B029": "Tamil Nadu",
    "B030": "Telangana",        "B031": "Tamil Nadu",
    "B032": "Andhra Pradesh",   "B033": "Gujarat",
    "B034": "Gujarat",          "B035": "Maharashtra",
    "B036": "Maharashtra",      "B037": "Maharashtra",
    "B038": "Chhattisgarh",     "B039": "Madhya Pradesh",
    "B040": "Madhya Pradesh",   "B041": "Madhya Pradesh",
    "B042": "Rajasthan",        "B043": "Rajasthan",
}



# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def to_dt(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _pick_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _read_any_excel(uploaded_or_path):
    xls = pd.ExcelFile(uploaded_or_path, engine="openpyxl")
    frames = [pd.read_excel(xls, sheet_name=n, engine="openpyxl") for n in xls.sheet_names]
    return pd.concat(frames, ignore_index=True)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas extension types (StringDtype, etc.) to numpy-compatible dtypes.
    Needed because np.issubdtype cannot interpret pandas extension array types."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            continue
        if pd.api.types.is_extension_array_dtype(out[c].dtype):
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
            else:
                out[c] = out[c].astype(object)
    return out


def derive_cash_type(sale_type: str) -> str:
    s = str(sale_type).strip().lower()
    credit_tokens = ["credit", "ifc", "ibc", "interest", "deferred"]
    return "CREDIT" if any(tok in s for tok in credit_tokens) else "CASH"


# ---------------------------------------------------------------------
# Holiday + day-type utilities 
# ---------------------------------------------------------------------
def saturday_of_month_rank(dt) -> int:
    dt = pd.Timestamp(dt).normalize()
    if dt.weekday() != 5:
        return 0
    first = dt.replace(day=1)
    first_sat = first + pd.Timedelta(days=(5 - first.weekday()) % 7)
    return 1 + ((dt - first_sat).days // 7)


def is_weekend(dt) -> bool:
    dt = pd.Timestamp(dt).normalize()
    if dt.weekday() == 6:
        return True
    if dt.weekday() == 5:
        return saturday_of_month_rank(dt) in (2, 4)
    return False


def is_public_only(dt, holidays: Set[date]) -> bool:
    dt = pd.Timestamp(dt).normalize()
    if is_weekend(dt):
        return False
    return dt.date() in holidays


def is_working(dt, holidays: Set[date]) -> bool:
    dt = pd.Timestamp(dt).normalize()
    if is_weekend(dt):
        return False
    return dt.date() not in holidays


def day_type(dt, holidays: Set[date]) -> str:
    dt = pd.Timestamp(dt).normalize()
    if is_weekend(dt):
        return "WEEKEND"
    if dt.date() in holidays:
        return "PUBLIC"
    return "WORKING"


def shift_backward_to_working(dt, holidays: Set[date], max_back: int = 21) -> pd.Timestamp:
    dt = pd.Timestamp(dt).normalize()
    if is_working(dt, holidays):
        return dt
    cur = dt
    for _ in range(max_back):
        cur = cur - pd.Timedelta(days=1)
        if is_working(cur, holidays):
            return cur
    return dt


def shift_forward_to_working(dt, holidays: Set[date], max_fwd: int = 21) -> pd.Timestamp:
    dt = pd.Timestamp(dt).normalize()
    if is_working(dt, holidays):
        return dt
    cur = dt
    for _ in range(max_fwd):
        cur = cur + pd.Timedelta(days=1)
        if is_working(cur, holidays):
            return cur
    return dt


def build_state_holiday_lookup(master_df: Optional[pd.DataFrame]) -> Dict[str, Set[date]]:
    """Build a state → Set[date] lookup from the uploaded holiday master DataFrame."""
    if master_df is None or master_df.empty:
        return {}
    lookup: Dict[str, Set[date]] = {}
    for _, row in master_df.iterrows():
        state = str(row["state"]).strip()
        d = row["holiday_date"]
        if isinstance(d, date):
            lookup.setdefault(state, set()).add(d)
    return lookup


# ---------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------
def read_rr_training_excel(uploaded_or_path) -> pd.DataFrame:
    df0 = _read_any_excel(uploaded_or_path)
    cols = list(df0.columns)

    if len(cols) == 8 and all(isinstance(c, str) for c in cols):
        df0.columns = TRAIN_COLS
        df = df0.copy()
    else:
        c_ret = _pick_col(cols, ["retirement_date", "Retirement Date", "RR Retirement Date", "RR Surrender Date"])
        c_branch = _pick_col(cols, ["branch", "Branch"])
        c_plant = _pick_col(cols, ["plant", "Plant"])
        c_cust = _pick_col(cols, ["customer_code", "Customer Code", "Customer"])
        c_inv_dt = _pick_col(cols, ["invoice_date", "Plant Invoice Date", "Invoice Date", "Plant Inv Date", "Inv. Date"])
        c_sale = _pick_col(cols, ["sale_type", "Type of Sale", "Sale Type", "Rel. Ord Type"])
        c_dest = _pick_col(cols, ["rly_dest", "Rly Dest", "Destination", "Mode Of Transport", "Mode Of Transport."])
        c_amt = _pick_col(cols, ["invoice_amount", "Invoice Amount", "Outstanding Amount", "RO Value", "Value"])
        if None in (c_ret, c_branch, c_plant, c_cust, c_inv_dt, c_amt):
            raise ValueError("RR training file format not recognized.")

        df = pd.DataFrame()
        df["retirement_date"] = df0[c_ret]
        df["branch"] = df0[c_branch]
        df["plant"] = df0[c_plant]
        df["customer_code"] = df0[c_cust]
        df["invoice_date"] = df0[c_inv_dt]
        df["sale_type"] = df0[c_sale] if c_sale else "UNKNOWN"
        df["rly_dest"] = df0[c_dest] if c_dest else "UNKNOWN"
        df["invoice_amount"] = df0[c_amt]

    df["invoice_date"] = to_dt(df["invoice_date"])
    df["retirement_date"] = to_dt(df["retirement_date"])
    for c in ["customer_code", "branch", "plant", "sale_type", "rly_dest"]:
        df[c] = df[c].fillna("UNKNOWN").astype(str).str.strip()
    df["invoice_amount"] = pd.to_numeric(df["invoice_amount"], errors="coerce").fillna(0.0).clip(0, 1e12)
    df = df.dropna(subset=["invoice_date", "retirement_date"]).copy()
    df["days_to_retirement"] = (df["retirement_date"] - df["invoice_date"]).dt.days
    q95 = df["days_to_retirement"].quantile(0.95)
    q05 = df["days_to_retirement"].quantile(0.05)
    df = df[(df["days_to_retirement"] >= 0) & (df["days_to_retirement"] <= q95) & (df["days_to_retirement"] >= q05)].copy()
    return df[TRAIN_COLS + ["days_to_retirement"]].reset_index(drop=True)



def read_rr_prediction_excel(uploaded_or_path) -> pd.DataFrame:
    df0 = _read_any_excel(uploaded_or_path)

    # Keep rail rows only (Mode Of Transport == 3); road == 11
    c_mot = _pick_col(list(df0.columns), ["Mode Of Transport", "Mode Of Transport.", "mode_of_transport"])
    if c_mot is not None:
        df0 = df0[pd.to_numeric(df0[c_mot], errors="coerce") == 3].copy()

    cols = list(df0.columns)

    if len(cols) == 7 and all(isinstance(c, str) for c in cols):
        df0.columns = PRED_COLS
        df = df0.copy()
    else:
        c_branch = _pick_col(cols, ["branch", "Branch"])
        c_plant = _pick_col(cols, ["plant", "Plant"])
        c_cust = _pick_col(cols, ["customer_code", "Customer Code", "Customer"])
        c_inv_dt = _pick_col(cols, ["invoice_date", "Plant Invoice Date", "Invoice Date", "Plant Inv Date", "Inv. Date"])
        c_sale = _pick_col(cols, ["sale_type", "Type of Sale", "Sale Type", "Rel. Ord Type"])
        c_dest = _pick_col(cols, ["rly_dest", "Rly Dest", "Destination", "Mode Of Transport", "Mode Of Transport."])
        c_amt = _pick_col(cols, ["invoice_amount", "Invoice Amount", "Outstanding Amount", "RO Value", "Value"])
        if None in (c_branch, c_plant, c_cust, c_inv_dt, c_amt):
            raise ValueError("RR prediction file format not recognized.")

        df = pd.DataFrame()
        df["branch"] = df0[c_branch]
        df["plant"] = df0[c_plant]
        df["customer_code"] = df0[c_cust]
        df["invoice_date"] = df0[c_inv_dt]
        df["sale_type"] = df0[c_sale] if c_sale else "UNKNOWN"
        df["rly_dest"] = df0[c_dest] if c_dest else "UNKNOWN"
        df["invoice_amount"] = df0[c_amt]

    df["invoice_date"] = to_dt(df["invoice_date"])
    for c in ["customer_code", "branch", "plant", "sale_type", "rly_dest"]:
        df[c] = df[c].fillna("UNKNOWN").astype(str).str.strip()
    df["invoice_amount"] = pd.to_numeric(df["invoice_amount"], errors="coerce").fillna(0.0).clip(0, 1e12)
    df = df.dropna(subset=["invoice_date"]).copy()
    return df[PRED_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------
# Training feature engineering 
# ---------------------------------------------------------------------
def create_customer_focused_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    global_median = df["days_to_retirement"].median()
    global_std = df["days_to_retirement"].std()

    df = df.sort_values(["customer_code", "invoice_date"]).reset_index(drop=True)

    def add_customer_statistics(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("invoice_date").copy()
        group["customer_count"] = np.arange(len(group))
        group["customer_avg_days"] = group["days_to_retirement"].expanding().mean().shift(1)
        group["customer_median_days"] = group["days_to_retirement"].expanding().median().shift(1)
        group["customer_std_days"] = group["days_to_retirement"].expanding().std().shift(1)
        group["customer_min_days"] = group["days_to_retirement"].expanding().min().shift(1)
        group["customer_max_days"] = group["days_to_retirement"].expanding().max().shift(1)

        group["customer_avg_days"] = group["customer_avg_days"].fillna(global_median)
        group["customer_median_days"] = group["customer_median_days"].fillna(global_median)
        group["customer_std_days"] = group["customer_std_days"].fillna(global_std)
        group["customer_min_days"] = group["customer_min_days"].fillna(global_median)
        group["customer_max_days"] = group["customer_max_days"].fillna(global_median)

        group["customer_range"] = group["customer_max_days"] - group["customer_min_days"]
        group["customer_cv"] = group["customer_std_days"] / (group["customer_avg_days"] + 1e-8)

        group["recent_avg_days"] = group["days_to_retirement"].rolling(window=3, min_periods=1).mean().shift(1)
        group["recent_avg_days"] = group["recent_avg_days"].fillna(global_median)

        group["recent_trend"] = (
            group["days_to_retirement"]
            .rolling(window=3, min_periods=1)
            .apply(lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0)
            .shift(1)
        )
        group["recent_trend"] = group["recent_trend"].fillna(0.0)
        group["days_since_last_invoice"] = group["invoice_date"].diff().dt.days.fillna(30)
        group["customer_consistency"] = 1 / (1 + group["customer_std_days"].fillna(0.0))
        return group

    _cust_idx = df["customer_code"].copy()  # preserve index→customer_code mapping for pandas 3.0+
    df = df.groupby("customer_code", group_keys=False).apply(add_customer_statistics)
    if "customer_code" not in df.columns:
        df["customer_code"] = df.index.map(_cust_idx.to_dict())
    df = df.reset_index(drop=True)

    df = df.sort_values(["branch", "invoice_date"]).reset_index(drop=True)
    df["branch_avg_days"] = (
        df.groupby("branch")["days_to_retirement"].expanding().mean().shift(1).reset_index(level=0, drop=True)
    )
    df["branch_median_days"] = (
        df.groupby("branch")["days_to_retirement"].expanding().median().shift(1).reset_index(level=0, drop=True)
    )
    df["branch_std_days"] = (
        df.groupby("branch")["days_to_retirement"].expanding().std().shift(1).reset_index(level=0, drop=True)
    )

    df = df.sort_values(["plant", "invoice_date"]).reset_index(drop=True)
    df["plant_avg_days"] = (
        df.groupby("plant")["days_to_retirement"].expanding().mean().shift(1).reset_index(level=0, drop=True)
    )
    df["plant_median_days"] = (
        df.groupby("plant")["days_to_retirement"].expanding().median().shift(1).reset_index(level=0, drop=True)
    )

    df = df.sort_values(["sale_type", "invoice_date"]).reset_index(drop=True)
    df["sale_type_avg_days"] = (
        df.groupby("sale_type")["days_to_retirement"].expanding().mean().shift(1).reset_index(level=0, drop=True)
    )
    df["sale_type_median_days"] = (
        df.groupby("sale_type")["days_to_retirement"].expanding().median().shift(1).reset_index(level=0, drop=True)
    )

    df = df.sort_values(["branch", "sale_type", "invoice_date"]).reset_index(drop=True)
    df["branch_sale_avg_days"] = (
        df.groupby(["branch", "sale_type"])["days_to_retirement"]
        .expanding().mean().shift(1).reset_index(level=[0, 1], drop=True)
    )

    df = df.sort_values(["customer_code", "branch", "invoice_date"]).reset_index(drop=True)
    df["cust_branch_avg_days"] = (
        df.groupby(["customer_code", "branch"])["days_to_retirement"]
        .expanding().mean().shift(1).reset_index(level=[0, 1], drop=True)
    )

    df["customer_branch_combo"] = df["customer_code"].astype(str) + "_" + df["branch"].astype(str)
    df["branch_sale_combo"] = df["branch"].astype(str) + "_" + df["sale_type"].astype(str)

    customer_stats = df.groupby("customer_code")["days_to_retirement"].agg(["mean", "std", "count"]).reset_index()
    customer_stats.columns = ["customer_code", "segment_avg_days", "segment_std_days", "segment_frequency"]

    def categorize_customer(row):
        avg_days = row["segment_avg_days"]
        std_days = row["segment_std_days"]
        frequency = row["segment_frequency"]
        if avg_days <= 5:
            return "Fast_Retirement"
        elif avg_days <= 15:
            return "Medium_Consistent" if std_days <= 3 else "Medium_Variable"
        else:
            return "Slow_Frequent" if frequency >= 5 else "Slow_Occasional"

    customer_stats["customer_segment"] = customer_stats.apply(categorize_customer, axis=1)
    df = df.merge(customer_stats[["customer_code", "customer_segment"]], on="customer_code", how="left")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(global_median)
    df = df.sort_values("invoice_date").reset_index(drop=True)
    return df



def enhance_customer_data_quality(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["customer_reliability"] = (df["customer_count"] >= 3).astype(int)
    global_median = df["days_to_retirement"].median()

    def get_enhanced_customer_avg(row):
        if row["customer_reliability"] == 0:
            fallback = row.get("branch_sale_avg_days", np.nan)
            if pd.isna(fallback):
                fallback = global_median
            return 0.7 * fallback + 0.3 * row["customer_avg_days"]
        return row["customer_avg_days"]

    df["enhanced_customer_avg"] = df.apply(get_enhanced_customer_avg, axis=1)
    return df


HIGH_IMPACT_FEATURES = [
    "customer_avg_days",
    "customer_median_days",
    "recent_avg_days",
    "customer_std_days",
    "branch_avg_days",
    "branch_median_days",
    "customer_consistency",
    "customer_range",
    "customer_cv",
]

MODERATE_IMPACT_FEATURES = [
    "plant_avg_days",
    "plant_median_days",
    "sale_type_avg_days",
    "sale_type_median_days",
    "customer_count",
    "days_since_last_invoice",
    "recent_trend",
    "cust_branch_avg_days",
]

CATEGORICAL_FEATURES_SELECTED = [
    "customer_code",
    "branch",
    "plant",
    "sale_type",
    "customer_segment",
    "customer_branch_combo",
    "branch_sale_combo",
]

FEATURE_COLS = HIGH_IMPACT_FEATURES + MODERATE_IMPACT_FEATURES + CATEGORICAL_FEATURES_SELECTED + ["enhanced_customer_avg"]



def make_onehot(max_categories: int = 50):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=max_categories)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



def create_smart_preprocessor(categorical_cols, numerical_cols, max_categories: int = 50):
    categorical_preprocessor = make_onehot(max_categories=max_categories)
    numerical_preprocessor = StandardScaler()
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_preprocessor, categorical_cols),
            ("num", numerical_preprocessor, numerical_cols),
        ],
        remainder="drop",
    )



def create_customer_focused_models(preprocessor):
    return {
        "rf_customer": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


# ---------------------------------------------------------------------
# Inference reference stats 
# ---------------------------------------------------------------------
def build_reference_stats(history: pd.DataFrame, fallback_median: float) -> Dict[str, object]:
    if history is None or history.empty:
        gmedian = float(fallback_median)
        gstd = 5.0
        return {
            "global_median": gmedian,
            "global_std": gstd,
            "cust": pd.DataFrame(columns=["customer_code"]),
            "branch": pd.DataFrame(columns=["branch"]),
            "plant": pd.DataFrame(columns=["plant"]),
            "sale": pd.DataFrame(columns=["sale_type"]),
            "bs": pd.DataFrame(columns=["branch", "sale_type"]),
            "cb": pd.DataFrame(columns=["customer_code", "branch"]),
            "cs": pd.DataFrame(columns=["customer_code", "customer_segment"]),
        }

    gmedian = float(history["days_to_retirement"].median())
    gstd_raw = history["days_to_retirement"].std()
    gstd = float(gstd_raw) if pd.notna(gstd_raw) else 5.0
    h = history.sort_values(["customer_code", "invoice_date"]).copy()

    cust = (
        h.groupby("customer_code")
        .agg(
            customer_avg_days=("days_to_retirement", "mean"),
            customer_median_days=("days_to_retirement", "median"),
            customer_std_days=("days_to_retirement", "std"),
            customer_min_days=("days_to_retirement", "min"),
            customer_max_days=("days_to_retirement", "max"),
            customer_count=("days_to_retirement", "count"),
            last_invoice_date=("invoice_date", "max"),
        )
        .reset_index()
    )
    cust["customer_std_days"] = cust["customer_std_days"].fillna(gstd)
    cust["customer_range"] = cust["customer_max_days"] - cust["customer_min_days"]
    cust["customer_cv"] = cust["customer_std_days"] / (cust["customer_avg_days"] + 1e-8)
    cust["customer_consistency"] = 1 / (1 + cust["customer_std_days"].fillna(gstd))

    def last3_stats(x: pd.DataFrame):
        x = x.sort_values("invoice_date")
        tail = x.tail(3)["days_to_retirement"].to_numpy()
        recent_avg = float(np.mean(tail)) if len(tail) else gmedian
        trend = float((tail[-1] - tail[0]) / len(tail)) if len(tail) >= 2 else 0.0
        return pd.Series({"recent_avg_days": recent_avg, "recent_trend": trend})

    cust_recent = h.groupby("customer_code").apply(last3_stats).reset_index()
    cust = cust.merge(cust_recent, on="customer_code", how="left")
    cust["recent_avg_days"] = cust["recent_avg_days"].fillna(gmedian)
    cust["recent_trend"] = cust["recent_trend"].fillna(0.0)

    branch = (
        h.groupby("branch")
        .agg(
            branch_avg_days=("days_to_retirement", "mean"),
            branch_median_days=("days_to_retirement", "median"),
            branch_std_days=("days_to_retirement", "std"),
        )
        .reset_index()
    )
    branch["branch_std_days"] = branch["branch_std_days"].fillna(gstd)

    plant = (
        h.groupby("plant")
        .agg(plant_avg_days=("days_to_retirement", "mean"), plant_median_days=("days_to_retirement", "median"))
        .reset_index()
    )

    sale = (
        h.groupby("sale_type")
        .agg(sale_type_avg_days=("days_to_retirement", "mean"), sale_type_median_days=("days_to_retirement", "median"))
        .reset_index()
    )

    bs = (
        h.groupby(["branch", "sale_type"]).agg(branch_sale_avg_days=("days_to_retirement", "mean")).reset_index()
    )

    cb = (
        h.groupby(["customer_code", "branch"]).agg(cust_branch_avg_days=("days_to_retirement", "mean")).reset_index()
    )

    cs = (
        h.groupby("customer_code")
        .agg(segment_avg_days=("days_to_retirement", "mean"), segment_std_days=("days_to_retirement", "std"), segment_frequency=("days_to_retirement", "count"))
        .reset_index()
    )
    cs["segment_std_days"] = cs["segment_std_days"].fillna(gstd)

    def categorize_customer(row):
        avg_days = row["segment_avg_days"]
        std_days = row["segment_std_days"]
        frequency = row["segment_frequency"]
        if avg_days <= 5:
            return "Fast_Retirement"
        elif avg_days <= 15:
            return "Medium_Consistent" if std_days <= 3 else "Medium_Variable"
        return "Slow_Frequent" if frequency >= 5 else "Slow_Occasional"

    cs["customer_segment"] = cs.apply(categorize_customer, axis=1)
    cs = cs[["customer_code", "customer_segment"]]

    return {
        "global_median": gmedian,
        "global_std": gstd,
        "cust": cust,
        "branch": branch,
        "plant": plant,
        "sale": sale,
        "bs": bs,
        "cb": cb,
        "cs": cs,
    }



def engineer_features_for_prediction(df_new: pd.DataFrame, ref: Dict[str, object]) -> pd.DataFrame:
    df = df_new.copy()
    gmedian = float(ref["global_median"])
    gstd = float(ref["global_std"])

    df = df.merge(ref["cust"], on="customer_code", how="left")
    df = df.merge(ref["cs"], on="customer_code", how="left")

    last = pd.to_datetime(df.get("last_invoice_date", pd.NaT), errors="coerce")
    inv = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["days_since_last_invoice"] = (inv - last).dt.days

    for col, default in [
        ("customer_avg_days", gmedian),
        ("customer_median_days", gmedian),
        ("customer_std_days", gstd),
        ("customer_min_days", gmedian),
        ("customer_max_days", gmedian),
        ("customer_count", 0),
        ("recent_avg_days", gmedian),
        ("recent_trend", 0.0),
    ]:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce").fillna(default)

    df["customer_range"] = (df["customer_max_days"] - df["customer_min_days"]).fillna(0.0)
    df["customer_cv"] = (df["customer_std_days"] / (df["customer_avg_days"] + 1e-8)).fillna(1.0)
    df["customer_consistency"] = (1 / (1 + df["customer_std_days"])).fillna(0.5)
    df["days_since_last_invoice"] = pd.to_numeric(df["days_since_last_invoice"], errors="coerce").fillna(30).clip(lower=0)

    df = df.merge(ref["branch"], on="branch", how="left")
    df = df.merge(ref["plant"], on="plant", how="left")
    df = df.merge(ref["sale"], on="sale_type", how="left")
    df = df.merge(ref["bs"], on=["branch", "sale_type"], how="left")
    df = df.merge(ref["cb"], on=["customer_code", "branch"], how="left")

    for col in [
        "branch_avg_days",
        "branch_median_days",
        "branch_std_days",
        "plant_avg_days",
        "plant_median_days",
        "sale_type_avg_days",
        "sale_type_median_days",
        "branch_sale_avg_days",
        "cust_branch_avg_days",
    ]:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce").fillna(gmedian)

    df["customer_reliability"] = (df["customer_count"] >= 3).astype(int)
    df["enhanced_customer_avg"] = np.where(
        df["customer_reliability"] == 0,
        0.7 * df["branch_sale_avg_days"] + 0.3 * df["customer_avg_days"],
        df["customer_avg_days"],
    )

    df["customer_segment"] = df["customer_segment"].fillna("UNKNOWN").astype(str)
    df["customer_branch_combo"] = df["customer_code"].astype(str) + "_" + df["branch"].astype(str)
    df["branch_sale_combo"] = df["branch"].astype(str) + "_" + df["sale_type"].astype(str)
    return df


# ---------------------------------------------------------------------
# Dynamic factor computation
# ---------------------------------------------------------------------
def build_daily_cash_from_history(hist_df: pd.DataFrame) -> pd.DataFrame:
    x = hist_df.copy()
    x["retirement_date"] = pd.to_datetime(x["retirement_date"], errors="coerce", dayfirst=True).dt.normalize()
    x["invoice_amount"] = pd.to_numeric(x["invoice_amount"], errors="coerce").fillna(0.0)
    x = x.dropna(subset=["retirement_date"]).copy()
    daily = x.groupby("retirement_date", as_index=False)["invoice_amount"].sum()
    daily = daily.rename(columns={"retirement_date": "date", "invoice_amount": "cash"})
    cal = pd.DataFrame({"date": pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")})
    cal = cal.merge(daily, on="date", how="left")
    cal["cash"] = cal["cash"].fillna(0.0)
    return cal



def compute_dynamic_factors_full_year(daily_cash: pd.DataFrame, holidays: Set[date]):
    d = daily_cash.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["day_type"] = d["date"].apply(lambda z: day_type(pd.Timestamp(z), holidays))
    d["is_sun"] = d["date"].dt.weekday.eq(6)
    d["is_sat24"] = d["date"].apply(lambda z: pd.Timestamp(z).weekday() == 5 and saturday_of_month_rank(pd.Timestamp(z)) in (2, 4))

    working = d[d["day_type"] == "WORKING"]["cash"]
    if working.empty:
        return 1.0, 1.0, 1.0

    base = float(working.mean())
    sat24_cash = d[d["is_sat24"]]["cash"]
    sun_cash = d[d["is_sun"]]["cash"]
    pub_cash = d[d["day_type"] == "PUBLIC"]["cash"]

    sat24_factor = float(sat24_cash.mean() / base) if not sat24_cash.empty else 1.0
    sun_factor = float(sun_cash.mean() / base) if not sun_cash.empty else 1.0
    pub_factor = float(pub_cash.mean() / base) if not pub_cash.empty else 1.0

    sat24_factor = float(np.clip(sat24_factor, 0.0, 1.5))
    sun_factor = float(np.clip(sun_factor, 0.0, 1.5))
    pub_factor = float(np.clip(pub_factor, 0.0, 1.5))
    return sat24_factor, sun_factor, pub_factor


def compute_dynamic_factors_from_training(
    training_df: pd.DataFrame,
    holiday_master_df: Optional[pd.DataFrame] = None,
):
    """
    Compute sat24/sun/pub factors from training data.

    Formula:
        base      = total CASH invoice amount on working days  / number of distinct working days
        x         = total CASH invoice amount on Sundays       / number of distinct Sundays
        y         = total CASH invoice amount on 2nd/4th Sats  / number of distinct 2nd/4th Sats
        z         = total CASH invoice amount on public hols   / number of distinct public hol days
        factor_*  = x/base, y/base, z/base respectively

    A day is PUBLIC only when it is a holiday in >= 3 states (from holiday_master_df).
    If no master is supplied, no days are classified as PUBLIC (factor_pub defaults to 1.0).
    """
    df = training_df.copy()
    df["retirement_date"] = pd.to_datetime(df["retirement_date"], errors="coerce").dt.normalize()
    df["invoice_amount"] = pd.to_numeric(df["invoice_amount"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["retirement_date"]).copy()

    if df.empty:
        return 1.0, 1.0, 1.0

    # Build state-wise holiday lookup from uploaded master (empty dict if not provided)
    state_hols = build_state_holiday_lookup(holiday_master_df)

    # Step 1: aggregate to daily totals
    daily = df.groupby("retirement_date", as_index=False)["invoice_amount"].sum()
    daily.columns = ["date", "total_cash"]

    # Step 2: classify each day
    def classify_day(dt):
        ts = pd.Timestamp(dt)
        if ts.weekday() == 6:
            return "SUNDAY"
        if ts.weekday() == 5 and saturday_of_month_rank(ts) in (2, 4):
            return "SAT_2_4"
        d = ts.date()
        if state_hols and sum(1 for hols in state_hols.values() if d in hols) >= 3:
            return "PUBLIC"
        return "WORKING"

    daily["day_class"] = daily["date"].apply(classify_day)

    # Step 3: base = total working-day cash / number of distinct working days
    working_rows = daily[daily["day_class"] == "WORKING"]
    if working_rows.empty:
        return 1.0, 1.0, 1.0
    base = float(working_rows["total_cash"].sum()) / len(working_rows)
    if base == 0.0:
        return 1.0, 1.0, 1.0

    # Step 4: per-type average / base
    def _factor(class_name: str) -> float:
        subset = daily[daily["day_class"] == class_name]
        if subset.empty:
            return 1.0
        avg = float(subset["total_cash"].sum()) / len(subset)
        return float(np.clip(avg / base, 0.0, 1.5))

    counts = {
        "n_working":  int(len(daily[daily["day_class"] == "WORKING"])),
        "n_sunday":   int(len(daily[daily["day_class"] == "SUNDAY"])),
        "n_sat24":    int(len(daily[daily["day_class"] == "SAT_2_4"])),
        "n_public":   int(len(daily[daily["day_class"] == "PUBLIC"])),
    }
    return (
        _factor("SAT_2_4"),   # factor_sat24
        _factor("SUNDAY"),    # factor_sun
        _factor("PUBLIC"),    # factor_pub
        counts,
    )


def apply_branch_state_aware_redistribution(
    cash_by_date_state: pd.DataFrame,
    factor_pub: float,
    factor_sat24: float,
    factor_sun: float,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    state_holiday_lookup: Optional[Dict[str, Set[date]]] = None,
) -> pd.DataFrame:
    """
    Redistribute invoice amounts using:
    - Uploaded holiday master (state_holiday_lookup) for state-specific public holidays
    - Universal weekend logic (Sundays and 2nd/4th Saturdays)
    If state_holiday_lookup is empty/None, public holiday redistribution is skipped.
    Returns DataFrame with columns: date, final_cash
    """
    if state_holiday_lookup is None:
        state_holiday_lookup = {}

    result_entries = []

    for _, row in cash_by_date_state.iterrows():
        dt = pd.Timestamp(row["date"]).normalize()
        state = str(row.get("state", "UNKNOWN"))
        amount = float(row["invoice_amount"])
        state_hols = state_holiday_lookup.get(state, set())

        if is_public_only(dt, state_hols):
            factor, offsets = factor_pub, [-1, +1, +2]
        elif dt.weekday() == 6:
            factor, offsets = factor_sun, [-1, +1, +2]
        elif dt.weekday() == 5 and saturday_of_month_rank(dt) in (2, 4):
            factor, offsets = factor_sat24, [-2, -1, +2]
        else:
            result_entries.append((dt, amount))
            continue

        kept = amount * float(factor)
        excess = amount - kept
        result_entries.append((dt, kept))

        if excess > 0:
            w = equal_case_weights(offsets)
            for offset in offsets:
                target = direction_preserving_adjust_target(
                    offset, dt + pd.Timedelta(days=offset), state_hols
                )
                result_entries.append((target, excess * w[offset]))

    if not result_entries:
        cal = pd.DataFrame({"date": pd.date_range(start_ts, end_ts, freq="D")})
        cal["final_cash"] = 0.0
        return cal

    res = pd.DataFrame(result_entries, columns=["date", "cash"])
    res["date"] = pd.to_datetime(res["date"]).dt.normalize()
    daily = res.groupby("date", as_index=False)["cash"].sum().rename(columns={"cash": "final_cash"})
    cal = pd.DataFrame({"date": pd.date_range(start_ts, end_ts, freq="D")})
    cal = cal.merge(daily, on="date", how="left")
    cal["final_cash"] = cal["final_cash"].fillna(0.0)
    return cal


# ---------------------------------------------------------------------
# Train / predict API for app.py
# ---------------------------------------------------------------------
def train_rr_model(
    training_df: pd.DataFrame,
    holidays: Optional[Set[date]] = None,
    holiday_master_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    if holidays is None:
        holidays = set()
    if training_df is None or training_df.empty:
        raise ValueError("RR training data is empty.")

    df_features = create_customer_focused_features(training_df.copy())
    df_features = enhance_customer_data_quality(df_features)

    feature_cols = FEATURE_COLS.copy()
    X = df_features[feature_cols].copy()
    y = df_features["days_to_retirement"].copy()

    categorical_cols = CATEGORICAL_FEATURES_SELECTED.copy()
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    df_sorted = df_features.sort_values("invoice_date").reset_index(drop=True)
    split_idx = max(1, int(len(df_sorted) * 0.8))
    if split_idx >= len(df_sorted):
        split_idx = len(df_sorted) - 1
    if split_idx <= 0:
        raise ValueError("Not enough RR history rows to train/test split.")

    X_train = df_sorted[feature_cols].iloc[:split_idx]
    X_test = df_sorted[feature_cols].iloc[split_idx:]
    y_train = df_sorted["days_to_retirement"].iloc[:split_idx]
    y_test = df_sorted["days_to_retirement"].iloc[split_idx:]

    preprocessor = create_smart_preprocessor(categorical_cols, numerical_cols)
    models = create_customer_focused_models(preprocessor)

    trained_models = {}
    individual_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        trained_models[name] = model
        individual_scores[name] = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

    model_names = list(trained_models.keys())
    mae_values = [individual_scores[n]["mae"] for n in model_names]
    weights = [1.0 / max(m, 1e-8) for m in mae_values]
    weights = [w / sum(weights) for w in weights]

    preds = [trained_models[n].predict(X_test) for n in model_names]
    ensemble_pred = np.average(np.vstack(preds), weights=weights, axis=0)
    ensemble_mae = float(mean_absolute_error(y_test, ensemble_pred))
    ensemble_rmse = float(np.sqrt(mean_squared_error(y_test, ensemble_pred)))
    ensemble_r2 = float(r2_score(y_test, ensemble_pred))

    ref = build_reference_stats(training_df.copy(), fallback_median=float(training_df["days_to_retirement"].median()))
    factor_sat24, factor_sun, factor_pub, day_counts = compute_dynamic_factors_from_training(training_df.copy(), holiday_master_df)

    return {
        "ensemble_models": trained_models,
        "ensemble_weights": dict(zip(model_names, weights)),
        "feature_names": feature_cols,
        "categorical_features": categorical_cols,
        "numerical_features": numerical_cols,
        "global_median": float(training_df["days_to_retirement"].median()),
        "performance": {
            "individual_scores": individual_scores,
            "ensemble_mae": ensemble_mae,
            "ensemble_rmse": ensemble_rmse,
            "ensemble_r2": ensemble_r2,
        },
        "reference_stats": ref,
        "dynamic_factors": {
            "sat24": factor_sat24, "sun": factor_sun, "pub": factor_pub,
            **day_counts,
        },
        "model_version": "venky_rr_integrated_v1",
    }



def create_ensemble_prediction(models_dict: Dict[str, Pipeline], X: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    model_names = list(models_dict.keys())
    preds = [models_dict[n].predict(X) for n in model_names]
    w = np.array([weights[n] for n in model_names], dtype=float)
    w = w / w.sum()
    return np.average(np.vstack(preds), weights=w, axis=0)


# ---------------------------------------------------------------------
# Redistribution + invoice-level details
# ---------------------------------------------------------------------
def build_targets_for_date(dt: pd.Timestamp, holidays: Set[date], factor_sat24: float, factor_sun: float, factor_pub: float):
    dt = pd.Timestamp(dt).normalize()
    if is_public_only(dt, holidays):
        return "PUBLIC", float(factor_pub), [-1, +1, +2], [dt + pd.Timedelta(days=o) for o in [-1, +1, +2]]
    if dt.weekday() == 6:
        sat = dt - pd.Timedelta(days=1)
        offs = [-1, +1, +2] if is_working(sat, holidays) else [-2, +1, +2]
        return "SUNDAY", float(factor_sun), offs, [dt + pd.Timedelta(days=o) for o in offs]
    if dt.weekday() == 5 and saturday_of_month_rank(dt) in (2, 4):
        offs = [-2, -1, +2]
        return "SAT_2_4", float(factor_sat24), offs, [dt + pd.Timedelta(days=o) for o in offs]
    return None, None, None, None


def direction_preserving_adjust_target(offset: int, tgt: pd.Timestamp, holidays: Set[date]) -> pd.Timestamp:
    tgt = pd.Timestamp(tgt).normalize()
    if is_working(tgt, holidays):
        return tgt
    if offset < 0:
        return shift_backward_to_working(tgt, holidays)
    if offset > 0:
        return shift_forward_to_working(tgt, holidays)
    return tgt


def equal_case_weights(offsets: Sequence[int]) -> Dict[int, float]:
    k = len(offsets)
    return {o: 1.0 / k for o in offsets}


def apply_full_redistribution(cal_df: pd.DataFrame, holidays: Set[date], factor_sat24: float, factor_sun: float, factor_pub: float) -> pd.DataFrame:
    out = cal_df.copy()
    out["moved_in"] = 0.0
    out["kept_cash"] = out["base_cash"].astype(float)
    date_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(out["date"])}

    for i, r in out.iterrows():
        dt = pd.Timestamp(r["date"]).normalize()
        base = float(r["base_cash"])
        case, factor, offsets, targets = build_targets_for_date(dt, holidays, factor_sat24, factor_sun, factor_pub)
        if case is None:
            out.at[i, "kept_cash"] = base
            continue

        kept = base * float(factor)
        excess = base - kept
        out.at[i, "kept_cash"] = kept

        if excess <= 0:
            continue

        w_case = equal_case_weights(offsets)
        for off, tgt in zip(offsets, targets):
            tgt2 = direction_preserving_adjust_target(off, tgt, holidays)
            if tgt2 in date_to_idx:
                j = date_to_idx[tgt2]
                out.at[j, "moved_in"] += excess * float(w_case[off])

    out["final_cash"] = out["kept_cash"] + out["moved_in"]
    out["final_cash_cr"] = out["final_cash"] / CRORE
    out["base_cash_cr"] = out["base_cash"] / CRORE
    out["kept_cash_cr"] = out["kept_cash"] / CRORE
    out["moved_in_cr"] = out["moved_in"] / CRORE
    return out


def predict_rr_cashflow(prediction_df: pd.DataFrame, model_data: Dict[str, object], start_date, horizon_days: int = 31, holidays: Optional[Set[date]] = None, holiday_master_df: Optional[pd.DataFrame] = None):
    if holidays is None:
        holidays = set()
    state_holiday_lookup = build_state_holiday_lookup(holiday_master_df)

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = (start_ts + pd.Timedelta(days=int(horizon_days))).normalize()

    if prediction_df is None or prediction_df.empty:
        daily = pd.DataFrame({"date": pd.date_range(start_ts, end_ts, freq="D")})
        daily["rr_total"] = 0.0
        daily["day_type"] = daily["date"].apply(lambda d: day_type(pd.Timestamp(d), holidays))
        empty_cust = pd.DataFrame(columns=["date", "customer_code", "rr_total"])
        empty_inv = pd.DataFrame(columns=["pred_retirement_date", "customer_code", "invoice_amount"])
        return daily, empty_cust, empty_inv

    feature_cols = model_data["feature_names"]
    ref = model_data["reference_stats"]
    trained_models = model_data["ensemble_models"]
    weights = model_data["ensemble_weights"]
    dyn = model_data.get("dynamic_factors", {"sat24": 1.0, "sun": 1.0, "pub": 1.0})

    pred_feat = engineer_features_for_prediction(prediction_df.copy(), ref)
    missing_feats = [c for c in feature_cols if c not in pred_feat.columns]
    if missing_feats:
        raise ValueError(f"Model expects features missing after engineering: {missing_feats}")

    X_new = pred_feat[feature_cols].copy()
    y_days = create_ensemble_prediction(trained_models, X_new, weights=weights)
    y_days = np.clip(y_days, 0, 365)

    pred_feat["pred_days_to_retirement"] = y_days
    pred_feat["pred_retirement_date"] = (
        pd.to_datetime(pred_feat["invoice_date"], errors="coerce")
        + pd.to_timedelta(np.round(y_days).astype(int), unit="D")
    )
    pred_feat["pred_retirement_date"] = pd.to_datetime(pred_feat["pred_retirement_date"], errors="coerce").dt.normalize()

    # All invoices together — no cash/credit split
    pred_feat["state"] = pred_feat["branch"].astype(str).str.strip().str.upper().map(BRANCH_STATE_MAP).fillna("UNKNOWN")
    if not pred_feat.empty:
        all_by_date_state = pred_feat.groupby(
            ["pred_retirement_date", "state"], as_index=False
        )["invoice_amount"].sum().rename(columns={"pred_retirement_date": "date"})
    else:
        all_by_date_state = pd.DataFrame(columns=["date", "state", "invoice_amount"])

    daily_redist = apply_branch_state_aware_redistribution(
        all_by_date_state,
        factor_pub=float(dyn.get("pub", 1.0)),
        factor_sat24=float(dyn.get("sat24", 1.0)),
        factor_sun=float(dyn.get("sun", 1.0)),
        start_ts=start_ts,
        end_ts=end_ts,
        state_holiday_lookup=state_holiday_lookup,
    )
    daily_redist = daily_redist.rename(columns={"final_cash": "rr_total"})

    daily = pd.DataFrame({"date": pd.date_range(start_ts, end_ts, freq="D")})
    daily = daily.merge(daily_redist, on="date", how="left")
    daily["rr_total"] = pd.to_numeric(daily.get("rr_total", 0.0), errors="coerce").fillna(0.0)
    daily["rr_total_cr"] = daily["rr_total"] / CRORE
    daily["day_type"] = daily["date"].apply(lambda d: day_type(pd.Timestamp(d), holidays))

    # Customer/day table: all invoices retiring on that day
    cust_day = pd.DataFrame(columns=["date", "customer_code", "rr_total"])
    if not pred_feat.empty:
        cust_day = pred_feat.groupby(["pred_retirement_date", "customer_code"], as_index=False)["invoice_amount"].sum().rename(
            columns={"pred_retirement_date": "date", "invoice_amount": "rr_total"}
        )
        cust_day = cust_day[(cust_day["date"] >= start_ts) & (cust_day["date"] <= end_ts)].copy()

    base_cols = ["pred_retirement_date", "customer_code", "invoice_amount", "sale_type"]
    invoice_details = pred_feat[[c for c in base_cols if c in pred_feat.columns]].copy() if not pred_feat.empty else pd.DataFrame(columns=base_cols)
    if not invoice_details.empty:
        invoice_details["pred_retirement_date"] = pd.to_datetime(invoice_details["pred_retirement_date"]).dt.normalize()
        invoice_details = invoice_details[
            (invoice_details["pred_retirement_date"] >= start_ts) & (invoice_details["pred_retirement_date"] <= end_ts)
        ].copy()

    return (
        _coerce_dtypes(daily.reset_index(drop=True)),
        _coerce_dtypes(cust_day.reset_index(drop=True)),
        _coerce_dtypes(invoice_details.reset_index(drop=True)),
    )


# ---------------------------------------------------------------------
# Master-file API  (app.py uses these three functions)
# ---------------------------------------------------------------------

def read_rr_master_excel(uploaded_or_path) -> pd.DataFrame:
    """Load a single master Excel containing both historical rows (retirement_date filled)
    and pending rows (retirement_date empty). Returns all rows with TRAIN_COLS; retirement_date
    is NaT for pending rows."""
    df0 = _read_any_excel(uploaded_or_path)
    cols = list(df0.columns)

    if len(cols) == 8 and all(isinstance(c, str) for c in cols):
        df0.columns = TRAIN_COLS
        df = df0.copy()
    else:
        c_ret    = _pick_col(cols, ["retirement_date", "Retirement Date", "RR Retirement Date", "RR Surrender Date"])
        c_branch = _pick_col(cols, ["branch", "Branch"])
        c_plant  = _pick_col(cols, ["plant", "Plant"])
        c_cust   = _pick_col(cols, ["customer_code", "Customer Code", "Customer"])
        c_inv_dt = _pick_col(cols, ["invoice_date", "Plant Invoice Date", "Invoice Date", "Plant Inv Date", "Inv. Date"])
        c_sale   = _pick_col(cols, ["sale_type", "Type of Sale", "Sale Type", "Rel. Ord Type"])
        c_dest   = _pick_col(cols, ["rly_dest", "Rly Dest", "Destination", "Mode Of Transport", "Mode Of Transport."])
        c_amt    = _pick_col(cols, ["invoice_amount", "Invoice Amount", "Outstanding Amount", "RO Value", "Value"])
        if None in (c_branch, c_plant, c_cust, c_inv_dt, c_amt):
            raise ValueError("RR master file format not recognized.")

        df = pd.DataFrame()
        df["retirement_date"] = df0[c_ret] if c_ret else pd.NaT
        df["branch"]          = df0[c_branch]
        df["plant"]           = df0[c_plant]
        df["customer_code"]   = df0[c_cust]
        df["invoice_date"]    = df0[c_inv_dt]
        df["sale_type"]       = df0[c_sale] if c_sale else "UNKNOWN"
        df["rly_dest"]        = df0[c_dest] if c_dest else "UNKNOWN"
        df["invoice_amount"]  = df0[c_amt]

    df["invoice_date"]    = to_dt(df["invoice_date"])
    df["retirement_date"] = to_dt(df["retirement_date"])
    for c in ["customer_code", "branch", "plant", "sale_type", "rly_dest"]:
        df[c] = df[c].fillna("UNKNOWN").astype(str).str.strip()
    df["invoice_amount"] = pd.to_numeric(df["invoice_amount"], errors="coerce").fillna(0.0).clip(0, 1e12)
    df = df.dropna(subset=["invoice_date"]).copy()
    return df[TRAIN_COLS].reset_index(drop=True)


def _prepare_hist_from_master(master_df: pd.DataFrame) -> pd.DataFrame:
    """Extract historical rows (retirement_date filled) with days_to_retirement, outliers removed."""
    hist = master_df[master_df["retirement_date"].notna()].copy()
    if hist.empty:
        return hist
    hist["days_to_retirement"] = (hist["retirement_date"] - hist["invoice_date"]).dt.days
    q95 = hist["days_to_retirement"].quantile(0.95)
    q05 = hist["days_to_retirement"].quantile(0.05)
    hist = hist[
        (hist["days_to_retirement"] >= 0)
        & (hist["days_to_retirement"] <= q95)
        & (hist["days_to_retirement"] >= q05)
    ].copy()
    return hist[TRAIN_COLS + ["days_to_retirement"]].reset_index(drop=True)


def train_rr_model_from_master(
    master_df: pd.DataFrame,
    holidays: Optional[Set[date]] = None,
    holiday_master_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Train RR ensemble using historical rows from master file."""
    hist = _prepare_hist_from_master(master_df)
    if hist.empty:
        raise ValueError("No historical rows (with retirement_date) found in master file.")
    return train_rr_model(hist, holidays=holidays, holiday_master_df=holiday_master_df)


def predict_rr_cashflow_from_master(
    master_df: pd.DataFrame,
    model_data: Dict[str, object],
    start_date,
    horizon_days: int = 31,
    holidays: Optional[Set[date]] = None,
    holiday_master_df: Optional[pd.DataFrame] = None,
):
    """Predict RR cashflow from pending rows in master file.

    Returns:
        daily_df        : date-level DataFrame with 'rr_final_cash' column
        cust_df         : customer/date-level DataFrame with 'rr_final_cash' column
        invoice_details : invoice-level DataFrame with columns
                          ['pred_retirement_date', 'customer_code', 'invoice_amount', 'sale_type']
                          filtered to the prediction horizon. Credit/cash classification is
                          performed in app.py using the BG file as the authoritative signal.
    """
    if holidays is None:
        holidays = set()

    pending_mask = (
        master_df["retirement_date"].isna()
        if "retirement_date" in master_df.columns
        else pd.Series([True] * len(master_df), index=master_df.index)
    )
    pending = master_df[pending_mask][PRED_COLS].copy()

    daily, cust_day, invoice_details = predict_rr_cashflow(
        pending, model_data, start_date, horizon_days, holidays, holiday_master_df
    )

    # Rename rr_total → rr_final_cash
    rename_map = {"rr_total": "rr_final_cash", "rr_total_cr": "rr_final_cash_cr"}
    daily    = daily.rename(columns={k: v for k, v in rename_map.items() if k in daily.columns})
    cust_day = cust_day.rename(columns={"rr_total": "rr_final_cash"})

    _empty_inv = pd.DataFrame(columns=["pred_retirement_date", "customer_code", "invoice_amount", "sale_type"])
    if invoice_details.empty or "sale_type" not in invoice_details.columns:
        return _coerce_dtypes(daily), _coerce_dtypes(cust_day), _empty_inv

    return (
        _coerce_dtypes(daily),
        _coerce_dtypes(cust_day),
        _coerce_dtypes(invoice_details),
    )
