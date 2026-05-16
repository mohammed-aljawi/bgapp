from __future__ import annotations

import pandas as pd
import streamlit as st
from folium import Icon, Map, Marker
from streamlit_folium import st_folium

from ai import LANGUAGE_NAMES, ask_ai, build_or_load_vector_index, build_report_without_ai, classify_audience, local_answer_without_ai, rag_search, records_to_context
from components import no_car_label, page_style, render_no_car_checker, render_record_card
from data_loader import combined_local_records, load_all_data
from search import search_records


st.set_page_config(page_title="NewBG", page_icon="N", layout="wide")
page_style()


TASK_BUTTONS = [
    "Build my plan",
    "What to do today",
    "Passing through essentials",
    "Get food today",
    "Cheap groceries",
    "Driver's license / REAL ID",
    "No car / bus help",
    "New resident help",
    "Visitor information",
    "Malls / shopping",
    "Businesses / services",
    "Car dealerships",
    "Housing & shelter",
    "Healthcare",
    "Animal care",
    "Parks & free places",
    "Library / computer access",
    "Jobs & work access",
    "First 30 days plan",
]


MAP_COLORS = {
    "food": "green",
    "groceries": "green",
    "driver license": "blue",
    "transportation": "cadetblue",
    "refugee support": "purple",
    "housing": "darkred",
    "healthcare": "red",
    "animal care": "orange",
    "parks": "darkgreen",
    "library/community": "lightblue",
    "visitor info": "pink",
    "things to do": "darkpurple",
    "shopping": "cadetblue",
    "business": "lightgray",
    "auto": "black",
    "jobs": "gray",
    "first 30 days": "beige",
}


def sidebar_settings() -> dict[str, str]:
    st.sidebar.title("NewBG settings")
    language = st.sidebar.selectbox("Answer language", LANGUAGE_NAMES)
    api_key = st.sidebar.text_input("Gemini API key", type="password")
    st.sidebar.caption("Gemini powers RAG retrieval and answer generation. Without a key, NewBG still uses local CSV keyword search.")
    st.sidebar.markdown("---")
    st.sidebar.write("Local facts come from CSV files. RAG retrieves from those files first.")
    return {"language": language, "api_key": api_key}


def filtered_task_records(records: pd.DataFrame, task: str) -> pd.DataFrame:
    aliases = {
        "Animal care": "lost pet vaccines spay neuter adopt dog animal care",
        "Parks & free places": "park kids dogs walking quiet family",
        "Healthcare": "healthcare clinic doctor medical",
        "Housing & shelter": "housing shelter eviction rent",
        "What to do today": "things to do visitor downtown park museum cave corvette free today",
        "Passing through essentials": "visitor passing through food restroom wifi gas downtown pharmacy urgent help",
        "New resident help": "new resident newcomer city support documents housing food transportation",
        "Visitor information": "visitor information tourism downtown things to do attractions",
        "Malls / shopping": "mall shopping greenwood stores retail scottsville road campbell lane",
        "Businesses / services": "business businesses chamber directory services repair professional companies",
        "Car dealerships": "dealership dealerships car dealer used cars auto sales scottsville road",
    }
    return search_records(records, aliases.get(task, task), limit=8)


def homepage(records: pd.DataFrame) -> None:
    st.title("NewBG")
    st.subheader("Plan a day, a move, or a quick stop in Bowling Green.")
    st.caption("For visitors, residents, students, workers, families, pet owners, and people passing through.")

    cols = st.columns(3)
    for idx, task in enumerate(TASK_BUTTONS):
        if cols[idx % 3].button(task, use_container_width=True):
            st.session_state["selected_task"] = task
            st.session_state["active_tab"] = "Task Finder"

    if "selected_task" in st.session_state:
        st.markdown("---")
        show_task(st.session_state["selected_task"], records, key_prefix="home_task")


def rag_admin(records: pd.DataFrame, data: dict[str, pd.DataFrame], settings: dict[str, str]) -> None:
    st.header("RAG Index")
    st.write("Create Gemini embeddings from the CSV files and save a local vector index.")
    if not settings["api_key"]:
        st.info("Paste a Gemini API key in the sidebar to create embeddings and FAISS index files.")
        return

    index_choice = st.selectbox(
        "CSV index to build",
        ["all_csv_records", "parks_csv", "animal_care_csv"],
        key="rag_admin_index_choice",
    )
    source = records
    if index_choice == "parks_csv":
        source = data["parks"]
    elif index_choice == "animal_care_csv":
        source = data["animal_care"]

    if st.button("Build / refresh RAG index", type="primary"):
        try:
            _, _, status, index_path, metadata_path = build_or_load_vector_index(source, settings["api_key"], index_choice)
            st.success(status)
            st.write(f"Index file: `{index_path}`")
            st.write(f"Metadata file: `{metadata_path}`")
        except Exception as exc:
            st.error(f"Could not build RAG index: {exc}")


def show_task(task: str, records: pd.DataFrame, key_prefix: str = "task") -> None:
    st.header(task)
    if task == "Build my plan":
        planner_checklist(records, key_prefix=f"{key_prefix}_plan")
        return
    matches = filtered_task_records(records, task)
    if matches.empty:
        st.info("NewBG does not have a local record for this yet. Verify with the city, library, or Start Here Warren County.")
        return
    for _, row in matches.head(5).iterrows():
        render_record_card(row)
    with st.expander("Check no-car reality for the first result", expanded=True):
        render_no_car_checker(matches.iloc[0])


def task_finder(records: pd.DataFrame) -> None:
    task = st.selectbox("Choose a task", TASK_BUTTONS, index=0, key="task_finder_choice")
    show_task(task, records, key_prefix="task_finder")


def smart_search(records: pd.DataFrame, settings: dict[str, str]) -> None:
    st.header("Ask NewBG Anything")
    query = st.text_input("Type what you need", value="Where are malls or car dealerships in Bowling Green?")
    if not query.strip():
        return
    audience, guidance = classify_audience(query)
    with st.expander("Why these results?"):
        st.write(f"NewBG matched this as: {audience}. {guidance}")
    rag_result = rag_search(records, query, settings["api_key"], limit=10, index_name="all_csv_records")
    results = rag_result.records
    with st.expander("RAG status"):
        st.write(rag_result.status)

    ai_answer, ai_error = ask_ai(
        settings["api_key"],
        f"Help answer this local request with practical Bowling Green next steps: {query}",
        records_to_context(results),
        settings["language"],
    )
    if ai_answer:
        st.markdown(ai_answer)
    else:
        st.markdown(local_answer_without_ai(query, results))
        if ai_error and settings["api_key"]:
            with st.expander("AI status"):
                st.write(ai_error)

    if not results.empty:
        st.subheader("Related local records")
        for _, row in results.iterrows():
            render_record_card(row)


def report_generator(records: pd.DataFrame, profiles: pd.DataFrame, settings: dict[str, str]) -> None:
    st.header("Local Plan Generator")
    profile_names = profiles["profile"].tolist() if not profiles.empty else ["visitor"]
    c1, c2, c3, c4 = st.columns(4)
    profile = c1.selectbox("Profile", profile_names)
    has_car = c2.radio("Has car", ["No", "Yes"], horizontal=True)
    has_kids = c3.radio("Has kids", ["No", "Yes"], horizontal=True)
    has_pets = c4.radio("Has pets", ["No", "Yes"], horizontal=True)
    priorities = st.multiselect(
        "Priorities",
        ["things to do", "food", "parks", "transportation", "visitor info", "housing", "license", "healthcare", "animal care", "jobs"],
        default=["things to do", "food", "transportation"],
    )

    query = " ".join([profile, has_car, has_kids, has_pets, " ".join(priorities)])
    matches = search_records(records, query, limit=10)
    local_report = build_report_without_ai(profile, has_car, has_kids, has_pets, priorities, matches)

    ai_answer, ai_error = ask_ai(
        settings["api_key"],
        f"Create a Bowling Green local plan for profile={profile}, has_car={has_car}, has_kids={has_kids}, has_pets={has_pets}, priorities={priorities}. Include today, next few days, longer-stay plan if relevant, top places, transportation warnings, hidden local reality, and what to verify.",
        records_to_context(matches),
        settings["language"],
    )
    if ai_answer:
        st.markdown(ai_answer)
    else:
        st.markdown(local_report)
        if ai_error and settings["api_key"]:
            with st.expander("AI status"):
                st.write(ai_error)

    st.subheader("Top local records")
    for _, row in matches.head(5).iterrows():
        render_record_card(row, compact=True)


def base_checklist(audience: str, stay_length: str, has_car: str, has_kids: str, has_pets: str) -> list[str]:
    items = [
        "Check hours before going",
        "Save the address and phone number",
        "Plan the return trip",
        "Bring water and phone charger",
    ]
    if has_car == "No":
        items.extend(["Check GoBG Transit before choosing destinations", "Avoid planning late return trips without a backup ride"])
    if has_kids == "Yes":
        items.extend(["Check stroller access, bathrooms, shade, and kid-friendly food nearby"])
    if has_pets == "Yes":
        items.extend(["Check pet rules before going", "Bring leash, waste bags, water, and vaccine records if needed"])
    if "visitor" in audience.lower() or "passing" in audience.lower():
        items.extend(["Pick one area instead of crossing town all day", "Verify tickets, parking, and weather"])
    if "out-of-state" in audience.lower():
        items.extend(["Check Kentucky license and REAL ID rules", "Save proof of address documents before office visits"])
    if "refugee" in audience.lower() or "new american" in audience.lower():
        items.extend(["Contact a trusted newcomer support office first", "Ask about interpretation and document requirements"])
    if "local resident" in audience.lower():
        items.extend(["Look for nearby options first", "Use the planner to compare cost, parking, and time"])
    if "resident" in audience.lower() or "moving" in audience.lower() or stay_length in ["A month or more", "I live here"]:
        items.extend(["Start a document folder", "Save library, transit, healthcare, and city service links"])
    return items


def plan_markdown(
    audience: str,
    stay_length: str,
    has_car: str,
    has_kids: str,
    has_pets: str,
    interests: list[str],
    checklist: list[str],
    matches: pd.DataFrame,
) -> str:
    lines = [
        "# My NewBG Plan",
        "",
        f"Audience: {audience}",
        f"Time in Bowling Green: {stay_length}",
        f"Has car: {has_car}",
        f"Kids: {has_kids}",
        f"Pets: {has_pets}",
        f"Interests: {', '.join(interests) if interests else 'Not selected'}",
        "",
        "## Checklist",
    ]
    lines.extend([f"- [ ] {item}" for item in checklist])
    lines.extend(["", "## Suggested Local Stops"])
    if matches.empty:
        lines.append("- No local matches yet. Try adding more interests.")
    else:
        for _, row in matches.head(6).iterrows():
            place = row.get("place_name") or row.get("task")
            task = row.get("task")
            address = row.get("address")
            lines.append(f"- {place}: {task}" + (f" ({address})" if address else ""))
    lines.extend(
        [
            "",
            "## Verify Before Going",
            "- Hours",
            "- Cost or tickets",
            "- Parking or transit",
            "- Weather",
            "- Pet and kid rules",
            "- Whether you can safely get back",
        ]
    )
    return "\n".join(lines)


def planner_checklist(records: pd.DataFrame, key_prefix: str = "planner") -> None:
    st.header("Planner & Checklist")
    st.write("Build a simple Bowling Green plan for today, a short visit, or settling in.")

    c1, c2, c3 = st.columns(3)
    audience = c1.selectbox(
        "Who is this for?",
        [
            "Visitor / day trip",
            "Passing through",
            "New resident",
            "Out-of-state mover",
            "Local resident",
            "Refugee / New American",
            "International student",
            "Worker",
            "Family",
            "Pet owner",
            "No-car traveler",
        ],
        key=f"{key_prefix}_audience",
    )
    stay_length = c2.selectbox("How long?", ["A few hours", "One day", "A weekend", "A week", "A month or more", "I live here"], key=f"{key_prefix}_stay")
    has_car = c3.radio("Has car", ["No", "Yes"], horizontal=True, key=f"{key_prefix}_car")

    c4, c5 = st.columns(2)
    has_kids = c4.radio("Kids with you?", ["No", "Yes"], horizontal=True, key=f"{key_prefix}_kids")
    has_pets = c5.radio("Pets with you?", ["No", "Yes"], horizontal=True, key=f"{key_prefix}_pets")

    interests = st.multiselect(
        "What do you want to include?",
        [
            "things to do",
            "food",
            "coffee / quiet place",
            "parks",
            "walking",
            "kids",
            "pets",
            "transportation",
            "visitor info",
            "groceries",
            "healthcare",
            "jobs",
            "housing",
            "documents",
        ],
        default=["things to do", "food", "parks"],
        key=f"{key_prefix}_interests",
    )

    custom_item = st.text_input("Add your own checklist item", key=f"{key_prefix}_custom_item")
    custom_items_key = f"{key_prefix}_custom_checklist_items"
    if st.button("Add item", use_container_width=True, key=f"{key_prefix}_add_item") and custom_item.strip():
        st.session_state.setdefault(custom_items_key, [])
        st.session_state[custom_items_key].append(custom_item.strip())

    checklist = base_checklist(audience, stay_length, has_car, has_kids, has_pets)
    checklist.extend(st.session_state.get(custom_items_key, []))

    st.subheader("My checklist")
    left, right = st.columns(2)
    for idx, item in enumerate(checklist):
        target = left if idx % 2 == 0 else right
        target.checkbox(item, key=f"{key_prefix}_checklist_{idx}_{item}")

    query = " ".join([audience, stay_length, has_car, has_kids, has_pets, " ".join(interests)])
    matches = search_records(records, query, limit=8)

    st.subheader("Suggested local stops")
    if matches.empty:
        st.info("No suggestions yet. Try adding interests like parks, food, visitor info, pets, or transportation.")
    else:
        for _, row in matches.head(5).iterrows():
            render_record_card(row, compact=True)

    plan_text = plan_markdown(audience, stay_length, has_car, has_kids, has_pets, interests, checklist, matches)
    st.download_button("Download my plan", data=plan_text, file_name="newbg-plan.md", mime="text/markdown", use_container_width=True, key=f"{key_prefix}_download")


def no_car_checker(records: pd.DataFrame) -> None:
    st.header("No-Car Reality Checker")
    st.write("Pick a place or task. NewBG shows the practical transportation issues, not just map distance.")
    labels = (
        records.apply(lambda row: f"{row.get('task', 'Task')} - {row.get('place_name', 'Place')}", axis=1).tolist()
        if not records.empty
        else []
    )
    if not labels:
        st.warning("No local records loaded.")
        return
    choice = st.selectbox("Task or place", labels)
    row = records.iloc[labels.index(choice)]
    render_record_card(row, compact=True)
    render_no_car_checker(row)


def local_map(records: pd.DataFrame) -> None:
    st.header("Local Map")
    categories = sorted([item for item in records.get("category", pd.Series(dtype=str)).dropna().unique() if item])
    selected = st.multiselect("Show types", categories, default=categories)
    map_df = records[records["category"].isin(selected)].copy() if selected else records.head(0)
    map_df["lat"] = pd.to_numeric(map_df.get("lat"), errors="coerce")
    map_df["lon"] = pd.to_numeric(map_df.get("lon"), errors="coerce")
    map_df = map_df.dropna(subset=["lat", "lon"])

    bg_map = Map(location=[36.9685, -86.4808], zoom_start=12, tiles="CartoDB positron")
    for _, row in map_df.iterrows():
        popup = f"""
        <b>{row.get('place_name') or row.get('task')}</b><br>
        {row.get('task', '')}<br>
        {row.get('address', '')}<br>
        No-car: {no_car_label(row.get('no_car_score', 0))}<br>
        {row.get('transportation_reality', '')}
        """
        Marker(
            [row["lat"], row["lon"]],
            popup=popup,
            tooltip=row.get("place_name") or row.get("task"),
            icon=Icon(color=MAP_COLORS.get(row.get("category"), "blue")),
        ).add_to(bg_map)
    st_folium(bg_map, height=520, use_container_width=True)


def animal_section(animal_df: pd.DataFrame, settings: dict[str, str]) -> None:
    st.header("Animal Care")
    question = st.text_input("Ask about animal care", value="I found a lost pet", key="animal_rag_question")
    rag_result = rag_search(animal_df, question, settings["api_key"], limit=5, index_name="animal_care_csv")
    with st.expander("Animal care RAG status"):
        st.write(rag_result.status)
    st.markdown(local_answer_without_ai(question, rag_result.records))

    st.subheader("Animal care tasks")
    tasks = animal_df["task"].tolist()
    selected = st.selectbox("Animal task", tasks)
    row = animal_df[animal_df["task"] == selected].iloc[0]
    render_record_card(row)
    render_no_car_checker(row)


def parks_section(parks_df: pd.DataFrame, settings: dict[str, str]) -> None:
    st.header("Parks & Free Places")
    question = st.text_input("Ask about parks", value="I want a park for kids and walking", key="parks_rag_question")
    rag_result = rag_search(parks_df, question, settings["api_key"], limit=5, index_name="parks_csv")
    with st.expander("Parks RAG status"):
        st.write(rag_result.status)
    st.markdown(local_answer_without_ai(question, rag_result.records))

    st.subheader("Filter parks")
    filters = st.multiselect(
        "Park characteristics",
        ["good_for_kids", "good_for_dogs", "good_for_walking", "quiet", "sports_fields", "near_wku", "good_for_families"],
        default=["good_for_kids"],
    )
    filtered = parks_df.copy()
    for field in filters:
        if field in filtered.columns:
            filtered = filtered[filtered[field].astype(str).str.lower() == "yes"]
    if filtered.empty:
        st.info("No park matches those filters in the demo data.")
        return
    for _, row in filtered.iterrows():
        render_record_card(row)


def ai_assistant(records: pd.DataFrame, settings: dict[str, str]) -> None:
    st.header("Broad Assistant")
    st.write("Ask about local life, visitors, businesses, shopping, dealerships, transit, parks, documents, pets, or general planning. NewBG retrieves from CSV data with RAG first.")
    question = st.text_area("Question", value="Where can I shop, find businesses, or look for car dealerships in Bowling Green?")
    if st.button("Answer", type="primary"):
        rag_result = rag_search(records, question, settings["api_key"], limit=8, index_name="all_csv_records")
        matches = rag_result.records
        with st.expander("RAG status"):
            st.write(rag_result.status)
        answer, error = ask_ai(
            settings["api_key"],
            question,
            records_to_context(matches),
            settings["language"],
        )
        if answer:
            st.markdown(answer)
        else:
            st.markdown(local_answer_without_ai(question, matches))
            if error and settings["api_key"]:
                with st.expander("AI status"):
                    st.write(error)
        if not matches.empty:
            st.subheader("Related local records")
            for _, row in matches.iterrows():
                render_record_card(row, compact=True)


def main() -> None:
    settings = sidebar_settings()
    data = load_all_data()
    records = combined_local_records(data)

    tab_names = [
        "Home",
        "Planner & Checklist",
        "RAG Index",
        "Task Finder",
        "Ask Anything",
        "Local Plan",
        "No-Car Checker",
        "Map",
        "Animal Care",
        "Parks",
        "Broad Assistant",
    ]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        homepage(records)
    with tabs[1]:
        planner_checklist(records, key_prefix="main_planner")
    with tabs[2]:
        rag_admin(records, data, settings)
    with tabs[3]:
        task_finder(records)
    with tabs[4]:
        smart_search(records, settings)
    with tabs[5]:
        report_generator(records, data["profiles"], settings)
    with tabs[6]:
        no_car_checker(records)
    with tabs[7]:
        local_map(records)
    with tabs[8]:
        animal_section(data["animal_care"], settings)
    with tabs[9]:
        parks_section(data["parks"], settings)
    with tabs[10]:
        ai_assistant(records, settings)


if __name__ == "__main__":
    main()
