
from __future__ import annotations
import math
from datetime import date, timedelta
from calendar import monthrange
from typing import Optional, Set
import pandas as pd



BRANCH_STATE_MAP: dict[str, str] = {
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


def _sat_week_of_month(d: date) -> int:

    if d.weekday() != 5:
        return 0
    first = d.replace(day=1)
    
    first_sat_offset = (5 - first.weekday()) % 7
    first_sat = first.replace(day=1 + first_sat_offset)
    return 1 + (d - first_sat).days // 7

def is_default_closed(d: date) -> bool:

    dow = d.weekday()
    if dow == 6:
        return True
    if dow == 5 and _sat_week_of_month(d) in (2, 4):
        return True
    return False


def load_holiday_master(uploaded) -> pd.DataFrame:
    df = pd.read_excel(uploaded, sheet_name=0, engine="openpyxl")
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    
    state_col = next((c for c in df.columns if "state" in c), None)
    date_col  = next((c for c in df.columns if "date" in c), None)
    if state_col is None or date_col is None:
        raise ValueError("Holiday master must have 'State' and 'Date' columns.")
    df = df[[state_col, date_col]].copy()
    df.columns = ["state", "holiday_date"]
    df["state"] = df["state"].astype(str).str.strip()
    df["holiday_date"] = pd.to_datetime(df["holiday_date"], dayfirst=True, errors="coerce").dt.date
    df = df.dropna(subset=["holiday_date"])
    return df.reset_index(drop=True)


def _default_holidays_for_year(year: int) -> Set[date]:
    
    s: Set[date] = set()
    d = date(year, 1, 1)
    end = date(year, 12, 31)
    
    while d <= end:
        if is_default_closed(d):
            s.add(d)
        d += timedelta(days=1)
    return s

def _years_from_master(master_df: Optional[pd.DataFrame]) -> Set[int]:
    today = date.today()
    
    
    base = {today.year - 1, today.year, today.year + 1}
    if master_df is None or master_df.empty:
        return base
    return base | {d.year for d in master_df["holiday_date"] if isinstance(d, date)}

def build_holiday_set(branch: str, master_df: Optional[pd.DataFrame] = None) -> Set[date]:
    years = _years_from_master(master_df)
    years.add(date.today().year)
    holidays: Set[date] = set()
    for y in years:
        holidays |= _default_holidays_for_year(y)
    if master_df is not None and not master_df.empty:
        state = BRANCH_STATE_MAP.get(str(branch).strip().upper(), None)
        if state is None:
            
            bup = str(branch).strip().upper()
            state = next((v for k, v in BRANCH_STATE_MAP.items() if k.upper() == bup), None)
        
        if state:
            mask = master_df["state"].str.lower() == state.lower()
            for d in master_df.loc[mask, "holiday_date"]:
                if isinstance(d, date):
                    holidays.add(d)
    return holidays

def build_global_holiday_set(master_df: Optional[pd.DataFrame] = None) -> Set[date]:
    years = _years_from_master(master_df)
    
    years.add(date.today().year)
    holidays: Set[date] = set()
    for y in years:
        holidays |= _default_holidays_for_year(y)
    if master_df is not None and not master_df.empty:
        for d in master_df["holiday_date"]:
            
            if isinstance(d, date):
                holidays.add(d)
    return holidays


def build_national_holiday_set(master_df: Optional[pd.DataFrame] = None) -> Set[date]:
    if master_df is None or master_df.empty:
        return set()
    states = master_df["state"].dropna().unique()
    if len(states) == 0:
        return set()
    state_sets = [set(master_df.loc[master_df["state"] == s, "holiday_date"]) for s in states]
    return state_sets[0].intersection(*state_sets[1:])

def is_bank_holiday(d: date, holidays: Set[date]) -> bool:
    return (d in holidays) or is_default_closed(d)

def days_to_next_holiday(d: date, holidays: Set[date], horizon: int = 60) -> float:
    for i in range(horizon + 1):
        candidate = d + timedelta(days=i)
        
        if is_bank_holiday(candidate, holidays):
            return float(i)
    
    return float(horizon + 1)

def days_since_last_holiday(d: date, holidays: Set[date], lookback: int = 60) -> float:
    for i in range(1, lookback + 1):
        
        candidate = d - timedelta(days=i)
        if is_bank_holiday(candidate, holidays):
            return float(i)
    return float(lookback + 1)

def hol_proximity_flags(anchor_date: date, holidays: Set[date], n: int = 10) -> dict:
    flags: dict = {}
    for k in range(1, n + 1):
        fwd = anchor_date + timedelta(days=k)
        bwd = anchor_date - timedelta(days=k)
        flags[f"is_hol_pred_{k}"] = 1.0 if (is_bank_holiday(fwd, holidays) or is_bank_holiday(bwd, holidays)) else 0.0
    
    flags["days_to_next_hol"] = days_to_next_holiday(anchor_date, holidays)
    flags["days_since_last_hol"] = days_since_last_holiday(anchor_date, holidays)
    flags["is_pred_day_holiday"] = 1.0 if is_bank_holiday(anchor_date, holidays) else 0.0
    return flags

def hol_invoice_proximity_flags(invoice_date: date, holidays: Set[date], n: int = 10) -> dict:
    flags: dict = {}
    
    for k in range(1, n + 1):
        fwd = invoice_date + timedelta(days=k)
        flags[f"is_hol_inv_{k}"] = 1.0 if is_bank_holiday(fwd, holidays) else 0.0
    
    flags["days_inv_to_first_hol"] = days_to_next_holiday(invoice_date, holidays)
    return flags
