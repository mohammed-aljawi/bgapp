from __future__ import annotations

import re
from dataclasses import dataclass

from difflib import SequenceMatcher

import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:
    class _FallbackFuzz:
        @staticmethod
        def partial_ratio(left: str, right: str) -> float:
            if not left or not right:
                return 0.0
            if left in right:
                return 100.0
            return SequenceMatcher(None, left, right).ratio() * 100

        @staticmethod
        def token_set_ratio(left: str, right: str) -> float:
            left_tokens = set(left.split())
            right_tokens = set(right.split())
            if not left_tokens or not right_tokens:
                return 0.0
            overlap = len(left_tokens & right_tokens)
            return (2 * overlap / (len(left_tokens) + len(right_tokens))) * 100

    fuzz = _FallbackFuzz()


FIELD_WEIGHTS = {
    "task": 5,
    "category": 4,
    "place_name": 4,
    "keywords": 5,
    "who_it_helps": 3,
    "what_to_do": 3,
    "best_for": 3,
    "local_reality": 2,
    "transportation_reality": 2,
    "warning": 2,
}


SYNONYMS = {
    "food": "meal pantry groceries hunger eat feeding cheap grocery",
    "car": "bus transit walk walking ride transportation no car",
    "visitor": "tourist passing through day trip things to do attraction museum cave downtown events",
    "visit": "visitor tourist passing through day trip things to do attraction museum cave downtown events",
    "visiting": "visitor tourist passing through day trip things to do attraction museum cave downtown events",
    "tourist": "visitor passing through day trip things to do attraction museum cave downtown events",
    "attraction": "visitor tourist things to do museum cave corvette downtown park",
    "today": "things to do visitor food park downtown event quick stop",
    "free": "free low cost things to do park downtown walking fountain square",
    "mall": "shopping greenwood mall stores retail scottsville road",
    "shopping": "mall stores retail greenwood mall scottsville campbell lane",
    "business": "business directory chamber company service repair professional",
    "businesses": "business directory chamber company service repair professional",
    "dealership": "car dealer auto sales used cars scottsville road campbell lane",
    "dealerships": "car dealer auto sales used cars scottsville road campbell lane",
    "dealer": "car dealer auto sales used cars scottsville road campbell lane",
    "cars": "dealership auto sales used cars vehicle scottsville road",
    "stores": "shopping mall retail greenwood scottsville road campbell lane",
    "restaurant": "restaurants food dining downtown scottsville road campbell lane visitor",
    "restaurants": "restaurants food dining downtown scottsville road campbell lane visitor",
    "dining": "restaurants food downtown scottsville road campbell lane visitor",
    "coffee": "coffee cafe quiet study downtown wku library",
    "local": "resident nearby low cost things to do services parks food",
    "resident": "local nearby city services license housing healthcare parks food",
    "tennessee": "out of state new to kentucky visitor license real id things to do",
    "kentucky": "resident license real id local services transportation",
    "mover": "new resident out of state license housing groceries healthcare",
    "moving": "new resident out of state license housing groceries healthcare",
    "license": "driver license real id kytc driving permit identification",
    "pet": "animal dog cat humane lost found vaccines spay neuter adopt",
    "park": "playground walking dogs kids sports quiet family",
    "new": "newcomer refugee immigrant international student new american first 30 days",
    "help": "support services assistance intake case manager start here",
    "doctor": "health clinic healthcare medical urgent",
    "house": "housing shelter rent emergency",
    "job": "work employment workforce career",
}


@dataclass
class SearchResult:
    score: float
    row: pd.Series


def normalize(text: object) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9\s/+-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def expand_query(query: str) -> str:
    words = normalize(query).split()
    expansions = [query]
    for word in words:
        if word in SYNONYMS:
            expansions.append(SYNONYMS[word])
    if "do not have a car" in query.lower() or "no car" in query.lower() or "without a car" in query.lower():
        expansions.append("bus transit walking no car transportation")
    return " ".join(expansions)


def searchable_text(row: pd.Series) -> str:
    chunks = []
    for field, weight in FIELD_WEIGHTS.items():
        text = normalize(row.get(field, ""))
        if text:
            chunks.extend([text] * weight)
    return " ".join(chunks)


def search_records(df: pd.DataFrame, query: str, limit: int = 8) -> pd.DataFrame:
    if df.empty or not query.strip():
        return pd.DataFrame()

    expanded = normalize(expand_query(query))
    original = normalize(query)
    query_tokens = set(expanded.split())
    results: list[SearchResult] = []

    for _, row in df.iterrows():
        text = searchable_text(row)
        token_hits = sum(1 for token in query_tokens if token in text)
        partial = fuzz.partial_ratio(expanded, text)
        token_set = fuzz.token_set_ratio(expanded, text)
        score = (token_hits * 12) + (partial * 0.45) + (token_set * 0.35)

        category = normalize(row.get("category", ""))
        task = normalize(row.get("task", ""))
        keywords = normalize(row.get("keywords", ""))
        if task and (task in expanded or expanded in task):
            score += 120
        if any(phrase in original for phrase in ["things to do", "something free", "free to do", "what to do"]):
            if category in {"things to do", "parks", "visitor info"}:
                score += 55
            if category == "animal care":
                score -= 35
        if any(token in f"{category} {task} {keywords}" for token in query_tokens):
            score += 18
        if ("no car" in expanded or "transit" in expanded or "bus" in expanded) and row.get("no_car_score", 0):
            score += max(0, 12 - int(row.get("no_car_score", 0)))

        if score > 45:
            results.append(SearchResult(score=score, row=row))

    ranked = sorted(results, key=lambda item: item.score, reverse=True)[:limit]
    if not ranked:
        return pd.DataFrame()
    out = pd.DataFrame([item.row for item in ranked]).copy()
    out.insert(0, "_score", [round(item.score, 1) for item in ranked])
    return out


def rows_for_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    task_norm = normalize(task)
    mask = df.apply(
        lambda row: task_norm in normalize(row.get("task", ""))
        or task_norm in normalize(row.get("category", ""))
        or task_norm in normalize(row.get("keywords", "")),
        axis=1,
    )
    return df[mask].copy()
