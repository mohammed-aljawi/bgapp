# NewBG

NewBG is a local-first guide for Bowling Green, Kentucky. It is built as a practical Streamlit civic-tech product for visitors, people passing through, new residents, refugees, immigrants, international students, workers, families, no-car residents, and pet owners.

## What it does

- Task Finder for specific local needs
- Smart Search over local CSV data
- Planner & Checklist for visitors, residents, families, workers, students, pet owners, and no-car travelers
- Local Plan Generator for visitors, newcomers, and residents
- Smart audience detection for locals, out-of-state movers, visitors, people passing through, refugees/New Americans, students, workers, families, pet owners, and no-car users
- Broad assistant behavior for shopping, malls, businesses, dealerships, services, restaurants, and general city questions
- No-Car Reality Checker
- Folium local map
- Animal care and parks sections
- Optional AI assistance with OpenAI or Gemini
- Friendly fallback behavior when no API key exists or AI fails

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app works without an API key. Add an OpenAI or Gemini key in the sidebar only if you want AI summaries, translation, or personalization.

## Data

Local facts come from CSV files in `data/`:

- `tasks.csv`
- `places.csv`
- `profiles.csv`
- `parks.csv`
- `animal_care.csv`

AI is instructed to use these records only for local facts and to say when something is unknown.

## Customize it

You can add or change local information by editing the files in `data/`.

Use the same pipe-separated format already in the files:

```text
task|category|place_name|address|phone|website|lat|lon|...
```

Good beginner edits:

- Add a new place to `data/places.csv`
- Add malls, restaurants, businesses, dealerships, salons, repair shops, or stores to `data/places.csv`
- Add a new park to `data/parks.csv`
- Add a new pet task to `data/animal_care.csv`
- Add a new audience type to `data/profiles.csv`
- Change homepage buttons in `TASK_BUTTONS` inside `app.py`

After editing, restart Streamlit:

```bash
streamlit run app.py
```
