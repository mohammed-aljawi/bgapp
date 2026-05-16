from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).parent / "data"


CSV_FILES = {
    "tasks": DATA_DIR / "tasks.csv",
    "places": DATA_DIR / "places.csv",
    "parks": DATA_DIR / "parks.csv",
    "animal_care": DATA_DIR / "animal_care.csv",
    "profiles": DATA_DIR / "profiles.csv",
}


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = CSV_FILES[name]
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="|").fillna("")
    if "no_car_score" in df.columns:
        df["no_car_score"] = pd.to_numeric(df["no_car_score"], errors="coerce").fillna(0).astype(int)
    return df


def load_all_data() -> dict[str, pd.DataFrame]:
    return {name: load_csv(name) for name in CSV_FILES}


def combined_local_records(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for source_name, df in data.items():
        if source_name == "profiles":
            continue
        if df.empty:
            continue
        copy = df.copy()
        copy["_dataset"] = source_name
        frames.append(copy)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).fillna("")
