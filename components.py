from __future__ import annotations

import pandas as pd
import streamlit as st


CATEGORY_ICONS = {
    "food": "Utensils",
    "groceries": "Shopping basket",
    "driver license": "Id card",
    "transportation": "Bus",
    "refugee support": "Handshake",
    "housing": "Home",
    "healthcare": "Heart pulse",
    "animal care": "Paw print",
    "parks": "Trees",
    "library/community": "Book open",
    "jobs": "Briefcase",
    "first 30 days": "Calendar check",
}


def page_style() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {max-width: 1180px; padding-top: 1.2rem;}
        div[data-testid="stMetric"] {background: #f6f8fb; border: 1px solid #e5e9f0; padding: .75rem; border-radius: 8px;}
        .newbg-card {border: 1px solid #e4e7ec; border-radius: 8px; padding: 1rem; background: #fff; margin-bottom: .75rem;}
        .newbg-card h4 {margin: 0 0 .35rem 0;}
        .newbg-muted {color: #5d6675; font-size: .92rem;}
        .newbg-warning {background: #fff8e6; border: 1px solid #f0d58c; padding: .75rem; border-radius: 8px;}
        .newbg-reality {background: #eef7f2; border: 1px solid #b7dfc5; padding: .75rem; border-radius: 8px;}
        .stButton>button {border-radius: 8px; min-height: 2.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def no_car_label(score: int | str) -> str:
    try:
        score_int = int(score)
    except (TypeError, ValueError):
        score_int = 0
    if score_int >= 8:
        return "Realistic without a car"
    if score_int >= 5:
        return "Possible, but plan carefully"
    if score_int >= 1:
        return "Hard without a car"
    return "Unknown"


def render_record_card(row: pd.Series, compact: bool = False) -> None:
    task = row.get("task", "Local option")
    place = row.get("place_name", "")
    category = row.get("category", "")
    address = row.get("address", "")
    phone = row.get("phone", "")
    website = row.get("website", "")
    score = row.get("no_car_score", "")

    st.markdown('<div class="newbg-card">', unsafe_allow_html=True)
    st.markdown(f"#### {task}")
    if place:
        st.write(f"**Where:** {place}")
    details = []
    if category:
        details.append(f"**Type:** {category}")
    if address:
        details.append(f"**Address:** {address}")
    if phone:
        details.append(f"**Phone:** {phone}")
    if details:
        st.markdown("  \n".join(details))
    if website:
        st.markdown(f"**Website:** [{website}]({website})")

    if not compact:
        if row.get("what_to_do"):
            st.write(f"**What to do:** {row.get('what_to_do')}")
        if row.get("what_to_bring"):
            st.write(f"**Bring/check:** {row.get('what_to_bring')}")
        if row.get("transportation_reality"):
            st.markdown(f'<div class="newbg-reality"><strong>No-car reality:</strong> {no_car_label(score)}<br>{row.get("transportation_reality")}</div>', unsafe_allow_html=True)
        if row.get("warning"):
            st.markdown(f'<div class="newbg-warning"><strong>Local warning:</strong> {row.get("warning")}</div>', unsafe_allow_html=True)
        footer = []
        if row.get("verified_date"):
            footer.append(f"Verified: {row.get('verified_date')}")
        if row.get("source_name"):
            source_url = row.get("source_url", "")
            source = row.get("source_name")
            footer.append(f"Source: [{source}]({source_url})" if source_url else f"Source: {source}")
        if footer:
            st.caption(" | ".join(footer))
    st.markdown("</div>", unsafe_allow_html=True)


def render_no_car_checker(row: pd.Series) -> None:
    st.subheader("No-Car Reality Checker")
    st.metric("Without-car score", f"{row.get('no_car_score', 'Unknown')}/10", no_car_label(row.get("no_car_score", 0)))
    st.write(row.get("transportation_reality") or "Transportation details are unknown. Call before going and plan the return trip.")
    st.write(f"**Bus limitation:** {row.get('bus_limitation', 'Check current GoBG Transit route and service hours.')}")
    st.write(f"**Walking difficulty:** {row.get('walking_difficulty', 'Unknown. Check sidewalks, heat, and crossings.')}")
    st.write(f"**After-hours issue:** {row.get('after_hours_issue', 'Many offices are weekday daytime only.')}")
    st.write(f"**Hidden transportation cost:** {row.get('hidden_transportation_cost', 'Rideshare or taxi may cost more than expected.')}")
    st.write(f"**Suggested workaround:** {row.get('suggested_workaround', 'Call first, combine trips, and ask agencies about rides or remote options.')}")
