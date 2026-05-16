from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from search import search_records


LANGUAGE_NAMES = ["English", "Arabic", "Spanish", "Swahili", "Somali", "Nepali", "Burmese", "Karen"]
EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_DIR = Path(__file__).parent / "vector_index"

RAG_FIELDS = [
    "task",
    "category",
    "place_name",
    "address",
    "phone",
    "website",
    "who_it_helps",
    "what_to_do",
    "what_to_bring",
    "local_reality",
    "transportation_reality",
    "best_for",
    "warning",
    "keywords",
    "source_name",
]


SYSTEM_RULES = """You are NewBG, a broad but careful local guide for Bowling Green, Kentucky.
You help visitors, people passing through, new residents, refugees, immigrants, international students, workers, families, no-car residents, and pet owners.
Use local records provided by the app first for Bowling Green facts.
You may give general planning advice for broad questions, but do not invent local addresses, phone numbers, eligibility rules, hours, fees, bus routes, business inventory, or services.
If a local fact is missing from the records, say it is not in NewBG's local data and suggest how to verify it.
Use plain language, short sections, and practical next steps."""


AUDIENCE_RULES = [
    (
        "Refugee / New American",
        ["refugee", "asylum", "new american", "immigrant", "resettlement", "international center"],
        "Prioritize trusted support offices, interpretation, documents, food, housing, healthcare, transportation, and school/work setup.",
    ),
    (
        "International student",
        ["international student", "wku", "campus", "student visa", "f1", "j1", "study abroad"],
        "Start with campus services and nearby needs, then separate student-only resources from city resources.",
    ),
    (
        "Out-of-state mover",
        ["moving from", "moved from", "from tennessee", "from ohio", "from indiana", "from florida", "from texas", "out of state", "new to kentucky"],
        "Treat this person as a new Kentucky resident. Focus on license/REAL ID, proof of address, local services, healthcare, groceries, and transportation.",
    ),
    (
        "Local resident",
        ["i live here", "local", "from here", "born here", "resident", "already live", "bowling green resident"],
        "Skip basic newcomer explanations. Focus on practical local options, shortcuts, free/low-cost resources, and current task needs.",
    ),
    (
        "Visitor / tourist",
        ["visitor", "visiting", "tourist", "day trip", "weekend trip", "vacation", "things to do"],
        "Focus on attractions, food, parks, parking, hours, tickets, weather, and simple short-stay plans.",
    ),
    (
        "Passing through",
        ["passing through", "road trip", "on the way", "driving through", "few hours", "quick stop", "stopping by"],
        "Focus on fast, low-friction stops: food, restrooms, parking, parks, pet/kid breaks, safety, and getting back on route.",
    ),
    (
        "Worker",
        ["worker", "job", "work", "shift", "employment", "factory", "warehouse"],
        "Focus on commute reality, jobs, documents, healthcare, food, and late-shift transportation.",
    ),
    (
        "Family",
        ["family", "kids", "children", "school", "playground", "baby", "stroller"],
        "Focus on kid-friendly places, bathrooms, shade, schools, clinics, libraries, food, and safe transportation.",
    ),
    (
        "Pet owner",
        ["pet", "dog", "cat", "animal", "lost pet", "found pet", "vaccine", "spay", "neuter"],
        "Focus on animal care, pet rules, parks, emergency vet reality, housing pet rules, and transportation with pets.",
    ),
    (
        "No-car traveler",
        ["no car", "do not have a car", "don't have a car", "without a car", "bus", "transit", "walk", "walking", "uber", "lyft"],
        "Plan around GoBG Transit, sidewalks, heat, road crossings, late returns, and ride costs before choosing a destination.",
    ),
]


@dataclass
class RagResult:
    records: pd.DataFrame
    status: str
    used_rag: bool
    index_path: str = ""
    metadata_path: str = ""


def classify_audience(text: str) -> tuple[str, str]:
    lowered = text.lower()
    for label, keywords, guidance in AUDIENCE_RULES:
        for keyword in keywords:
            if " " in keyword or "-" in keyword:
                if keyword in lowered:
                    return label, guidance
            elif re.search(rf"\b{re.escape(keyword)}\b", lowered):
                return label, guidance
    return (
        "General local user",
        "Ask what they need first, then tailor guidance by time, transportation, kids, pets, documents, and urgency.",
    )


def records_to_context(records: pd.DataFrame, max_rows: int = 8) -> str:
    if records.empty:
        return "No matching local records were found."
    lines: list[str] = []
    fields = [
        "task",
        "category",
        "place_name",
        "address",
        "phone",
        "website",
        "what_to_do",
        "what_to_bring",
        "local_reality",
        "transportation_reality",
        "warning",
        "verified_date",
        "source_name",
        "source_url",
    ]
    for idx, row in records.head(max_rows).iterrows():
        lines.append(f"Record {idx + 1}:")
        for field in fields:
            value = str(row.get(field, "")).strip()
            if value:
                lines.append(f"- {field}: {value}")
    return "\n".join(lines)


def records_signature(records: pd.DataFrame) -> str:
    if records.empty:
        return "empty"
    safe = records.fillna("").astype(str)
    payload = safe.to_csv(index=False, sep="|")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def row_to_document(row: pd.Series) -> str:
    parts = []
    for field in RAG_FIELDS:
        value = str(row.get(field, "")).strip()
        if value:
            parts.append(f"{field}: {value}")
    return "\n".join(parts)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (vectors / norms).astype("float32")


def gemini_embed(api_key: str, texts: list[str]) -> np.ndarray:
    from google import genai

    client = genai.Client(api_key=api_key.strip())
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    vectors = [embedding.values for embedding in response.embeddings]
    return normalize_vectors(np.array(vectors, dtype="float32"))


def index_paths(index_name: str, signature: str) -> tuple[Path, Path, Path]:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", index_name).strip("_") or "records"
    base = VECTOR_DIR / f"{safe_name}_{signature}"
    return base.with_suffix(".faiss"), base.with_suffix(".npy"), base.with_suffix(".json")


def save_metadata(path: Path, records: pd.DataFrame, docs: list[str], signature: str) -> None:
    payload = {
        "signature": signature,
        "embedding_model": EMBEDDING_MODEL,
        "count": len(docs),
        "documents": docs,
        "records": records.fillna("").to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_or_load_vector_index(
    records: pd.DataFrame,
    api_key: str,
    index_name: str = "local_records",
) -> tuple[np.ndarray, pd.DataFrame, str, Path, Path]:
    VECTOR_DIR.mkdir(exist_ok=True)
    signature = records_signature(records)
    faiss_path, npy_path, metadata_path = index_paths(index_name, signature)

    if metadata_path.exists() and (faiss_path.exists() or npy_path.exists()):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stored_records = pd.DataFrame(metadata.get("records", [])).fillna("")
        if faiss_path.exists():
            try:
                import faiss

                index = faiss.read_index(str(faiss_path))
                vectors = np.vstack([index.reconstruct(i) for i in range(index.ntotal)]).astype("float32")
                return vectors, stored_records, "Loaded existing FAISS index", faiss_path, metadata_path
            except Exception:
                pass
        vectors = np.load(npy_path)
        return vectors, stored_records, "Loaded existing NumPy embedding index", npy_path, metadata_path

    docs = [row_to_document(row) for _, row in records.iterrows()]
    vectors = gemini_embed(api_key, docs)
    np.save(npy_path, vectors)

    engine = "Created NumPy embedding index"
    index_file = npy_path
    try:
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(faiss_path))
        engine = "Created FAISS index"
        index_file = faiss_path
    except Exception:
        pass

    save_metadata(metadata_path, records, docs, signature)
    return vectors, records.copy(), engine, index_file, metadata_path


def vector_search(vectors: np.ndarray, query_vector: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray, str]:
    scores = vectors @ query_vector[0]
    ids = np.argsort(scores)[::-1][:limit]
    return scores[ids], ids, "vector search"


def rag_search(records: pd.DataFrame, query: str, api_key: str = "", limit: int = 8, index_name: str = "local_records") -> RagResult:
    if records.empty or not query.strip():
        return RagResult(pd.DataFrame(), "No records to search.", False)

    if not api_key.strip():
        fallback = search_records(records, query, limit=limit)
        return RagResult(fallback, "No Gemini key set, so NewBG used local keyword search. No embedding index was created.", False)

    try:
        vectors, indexed_records, engine, index_path, metadata_path = build_or_load_vector_index(records, api_key, index_name)
        query_vector = gemini_embed(api_key, [query])
        scores, ids, search_engine = vector_search(vectors, query_vector, limit)
        out = indexed_records.iloc[[int(item) for item in ids if int(item) >= 0]].copy()
        out.insert(0, "_rag_score", [round(float(score), 4) for score in scores[: len(out)]])
        status = f"{engine}; searched with Gemini query embedding. Index file: {index_path.name}"
        return RagResult(out, status, True, str(index_path), str(metadata_path))
    except Exception as exc:
        fallback = search_records(records, query, limit=limit)
        return RagResult(fallback, f"RAG could not run, so NewBG used local keyword search. Reason: {exc}", False)


def friendly_ai_error(exc: Exception) -> str:
    raw = str(exc).lower()
    if "api key" in raw or "authentication" in raw or "unauthorized" in raw or "401" in raw:
        return "The AI key was missing or rejected. NewBG is still showing local results from its CSV files."
    if "quota" in raw or "rate limit" in raw or "429" in raw:
        return "The AI service is out of quota or temporarily rate limited. The local guide still works."
    if "model" in raw or "not found" in raw or "404" in raw:
        return "The selected AI model was not available. The local results below are still usable."
    if "network" in raw or "connection" in raw or "timeout" in raw:
        return "The AI service could not be reached. This may be a network issue, but local results still work."
    return "AI could not finish this request. NewBG kept the app usable and showed local information instead."


def has_key(key: str) -> bool:
    return bool(key and key.strip())


def ask_ai(api_key: str, prompt: str, context: str, language: str = "English") -> tuple[str | None, str | None]:
    if not has_key(api_key):
        return None, "No Gemini key is set. Showing local CSV results only."

    final_prompt = f"{SYSTEM_RULES}\n\nAnswer language: {language}\n\nLocal records:\n{context}\n\nUser request:\n{prompt}"

    try:
        from google import genai

        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=final_prompt,
        )
        return getattr(response, "text", ""), None
    except Exception as exc:  # Keep the civic guide alive even when AI is cranky.
        return None, friendly_ai_error(exc)


def local_answer_without_ai(prompt: str, records: pd.DataFrame) -> str:
    if records.empty:
        broad = broad_fallback_answer(prompt)
        return f"""{broad}

Check Google Maps, the business website, Visit Bowling Green, the Chamber directory, or call before going for current hours and details."""

    first = records.iloc[0]
    place = first.get("place_name") or first.get("task") or "the first matching local option"
    task = first.get("task") or "Local option"
    why = first.get("what_to_do") or "Start with this local record and verify details before going."
    transport = first.get("transportation_reality") or "Plan transportation before you go, especially if you do not have a car."
    warning = first.get("warning") or "Check hours, cost, documents, and availability before going."

    next_steps = []
    for _, row in records.iloc[1:3].iterrows():
        label = row.get("place_name") or row.get("task")
        action = row.get("task")
        address = row.get("address")
        detail = f"- **{label}**"
        if action and action != label:
            detail += f": {action}"
        if address:
            detail += f" - {address}"
        next_steps.append(detail)
    other_matches = "\n".join(next_steps)

    answer = f"""### {place}
{why}"""

    if first.get("address"):
        answer += f"\n\n**Address:** {first.get('address')}"
    if first.get("phone"):
        answer += f"\n\n**Phone:** {first.get('phone')}"
    if first.get("website"):
        answer += f"\n\n**Website:** {first.get('website')}"

    if transport:
        answer += f"\n\n**Getting there:** {transport}"

    if warning:
        answer += f"\n\n**Before you go:** {warning}"

    if other_matches:
        answer += f"\n\n**Other nearby/useful options:**\n{other_matches}"

    answer += "\n\nVerify hours and details before going."
    return answer


def broad_fallback_answer(prompt: str) -> str:
    text = prompt.lower()
    if any(word in text for word in ["mall", "shopping", "shop", "store", "clothes", "shoes"]):
        return """### Shopping guidance
- Start with Greenwood Mall or the Scottsville Road / Campbell Lane retail area.
- For big-box stores, groceries, shoes, clothing, phone stores, and restaurants, the Scottsville Road corridor is usually the main commercial area.
- If you do not have a car, check transportation first. Shopping areas can be spread out and hard to walk safely.
- Before going, verify hours, store availability, parking, and whether the specific store is still open."""

    if any(word in text for word in ["dealership", "dealer", "car dealer", "used car", "buy a car", "auto sales"]):
        return """### Car dealership guidance
- Start with the Scottsville Road / Campbell Lane dealership corridor.
- Compare at least three places before buying: price, fees, warranty, financing, reviews, and service department.
- Ask for the out-the-door price before discussing monthly payment.
- If buying used, check vehicle history, inspection, title status, and whether a trusted mechanic can look at it.
- NewBG does not verify live inventory. Call or check dealer websites before going."""

    if any(word in text for word in ["business", "businesses", "company", "companies", "service", "repair", "plumber", "lawyer", "bank"]):
        return """### Business search guidance
- For a specific business type, search the Bowling Green Area Chamber directory, Google Maps, and official business websites.
- Compare location, hours, reviews, phone number, accessibility, parking, and whether appointments are required.
- For urgent repair or professional services, call first and ask about price, availability, and service area.
- NewBG can guide the process, but it does not verify every private business in town yet."""

    if any(word in text for word in ["restaurant", "food", "eat", "coffee", "cafe", "halal"]):
        return """### Food guidance
- Decide first: quick food, sit-down restaurant, groceries, international food, coffee, or emergency food help.
- For visitor food, downtown and Scottsville Road / Campbell Lane are useful starting areas.
- For low-cost groceries, compare big-box grocery areas and plan transportation if you do not have a car.
- For halal or specialty groceries, call the store first because inventory changes."""

    return """### General answer
- Tell NewBG what you are trying to do, how much time you have, whether you have a car, and whether kids or pets are with you.
- If it is a local place or business, verify hours, address, phone number, cost, parking, and transit before going.
- If the question is not in NewBG's local CSV yet, the app can still give general planning steps and show related local records."""


def build_report_without_ai(
    profile: str,
    has_car: str,
    has_kids: str,
    has_pets: str,
    priorities: Iterable[str],
    matches: pd.DataFrame,
) -> str:
    priorities_text = ", ".join(priorities) if priorities else "basic setup"
    no_car = has_car == "No"
    is_visitor = any(word in profile.lower() for word in ["visitor", "passing", "day trip"])
    top_places = []
    for _, row in matches.head(6).iterrows():
        place = row.get("place_name") or row.get("task")
        task = row.get("task")
        phone = row.get("phone")
        top_places.append(f"- {place}: {task}" + (f" ({phone})" if phone else ""))
    top_places_text = "\n".join(top_places) if top_places else "- Start Here Warren County: ask for the right local referral."

    transportation = (
        "Plan every trip around bus hours, heat, sidewalks, and return rides. Places can look close on a map but be hard to reach safely."
        if no_car
        else "Even with a car, check parking, office hours, and whether the office requires an appointment."
    )
    pets = "If pets are part of the household, keep vaccine records and call before bringing an animal inside an office." if has_pets == "Yes" else ""
    kids = "For kids, ask about school enrollment documents, immunization records, and nearby library support." if has_kids == "Yes" else ""

    if is_visitor:
        return f"""### Today
- Pick one easy area first: downtown, a park, WKU area, or a major attraction.
- Confirm hours, parking, tickets, pet rules, and weather before driving across town.
- Plan food, restroom access, phone battery, and the return trip.

### Next few days
- Work through priorities: {priorities_text}.
- Mix one paid attraction with free stops like parks, downtown walking, or library/community spaces.
- If you do not have a car, choose destinations by transit and walking comfort before choosing by distance.

### If you stay longer
- Add groceries, healthcare, library access, work needs, and transportation routines.
- Save official pages for parks, transit, visitor information, and city services.

### Top places to contact
{top_places_text}

### Transportation warnings
{transportation}

### Hidden local reality
Bowling Green can look simple on a map, but bus hours, missing sidewalks, heat, road design, and after-hours return trips can change the real difficulty.

### What to verify before going
Hours, tickets or fees, parking, pet rules, weather, restroom access, and whether you can safely get back.

{kids}
{pets}
"""

    return f"""### First 48 hours
- Confirm food, safe place to sleep, phone access, and transportation for appointments or errands.
- Contact the most relevant local support office before traveling.
- Write down documents you already have: ID, passport, proof of address, medical records, pet records, and school papers.

### First week
- Work through priorities: {priorities_text}.
- Call ahead for hours, eligibility, appointment rules, and documents.
- Save GoBG Transit, Warren County Public Library, parks, and city services as practical support points.

### First 30 days
- Build a repeatable plan for groceries, healthcare, work/school, transportation, recreation, and pet needs if relevant.
- If you need a Kentucky license or REAL ID, verify documents with KYTC before going.
- Keep a small folder with IDs, mail/proof of address, lease papers, school papers, and vaccine records.

### Top places to contact
{top_places_text}

### Transportation warnings
{transportation}

### Hidden local reality
Bowling Green can look simple on a map, but bus hours, missing sidewalks, heat, road design, and after-hours return trips can change the real difficulty.

### What to verify before going
Hours, appointment rules, required documents, fees, whether the service fits your situation, and whether you can safely get back home.

{kids}
{pets}
"""
