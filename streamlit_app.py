from pathlib import Path

import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

def generate_ai_briefing(visit, pets):
    """Generate a pre-visit briefing grounded only in workbook data."""

    pet_details = pets[
        [
            "pet_name",
            "species",
            "feeding",
            "medication",
            "behavior_notes",
        ]
    ].to_dict(orient="records")

    prompt = f"""
You are an AI field service assistant helping an employee prepare for a client visit.

Generate a concise, professional pre-visit briefing using ONLY the information
provided below.

Include these sections:
1. Visit Summary
2. Entry and Access
3. Pet Care Instructions
4. Medication
5. Safety and Behavior Considerations
6. Supplies and Preparation
7. Missing Information

Rules:
- Do not invent or infer facts.
- Clearly state when information is unavailable.
- Highlight medication or safety information.
- Use clear bullet points.
- Keep the briefing practical and easy to scan.

VISIT DATA:
Client: {visit['client_name']}
Date: {visit['visit_date']}
Time: {visit['visit_time']}
Service: {visit['service_type']}
Assigned worker: {visit['assigned_to']}
Phone: {visit['phone']}
Address: {visit['address']}
Entry instructions: {visit['entry_instructions']}
Supply location: {visit['supply_location']}
Communication style: {visit['communication_style']}

PET DATA:
{pet_details}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )

    return response.output_text

st.set_page_config(
    page_title="AI Field Service Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("AI Field Service Assistant")
st.caption(
    "A context-aware workflow agent that prepares field workers "
    "for customer visits and drafts post-visit communication."
)

st.info("Demo workflow: Excel data → Pre-visit briefing → Client update")


DATA_FILE = Path("data/demo_data.xlsx")

REQUIRED_COLUMNS = {
    "Clients": {
        "client_id",
        "client_name",
        "phone",
        "address",
        "entry_instructions",
        "supply_location",
        "communication_style",
    },
    "Pets": {
        "pet_id",
        "client_id",
        "pet_name",
        "species",
        "feeding",
        "medication",
        "behavior_notes",
    },
    "Visits": {
        "visit_id",
        "client_id",
        "visit_date",
        "visit_time",
        "service_type",
        "assigned_to",
    },
}

def generate_client_update(visit, pets, completed_tasks, visit_notes):
    """Draft a client update using only the submitted visit information."""

    pet_names = pets["pet_name"].dropna().astype(str).tolist()

    prompt = f"""
You are an AI field service assistant drafting a post-visit message to a client.

Write a concise, warm, professional update using ONLY the information provided.

Rules:
- Do not invent activities, observations, or outcomes.
- Mention only tasks included in COMPLETED TASKS.
- Use the worker's notes when relevant.
- Do not mention internal instructions, missing data, or AI.
- Match the client's preferred communication style.
- Keep the message between 3 and 6 sentences.
- End with the assigned worker's first name.

CLIENT:
Name: {visit['client_name']}
Preferred communication style: {visit['communication_style']}

VISIT:
Service: {visit['service_type']}
Date: {visit['visit_date']}
Time: {visit['visit_time']}
Assigned worker: {visit['assigned_to']}
Pets: {pet_names}

COMPLETED TASKS:
{completed_tasks}

WORKER NOTES:
{visit_notes if visit_notes.strip() else "No additional notes provided."}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )

    return response.output_text

@st.cache_data
def load_demo_data(file_path: Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(file_path, sheet_name=None)


def validate_workbook(
    workbook: dict[str, pd.DataFrame],
) -> list[str]:
    errors = []

    for sheet_name, required_columns in REQUIRED_COLUMNS.items():
        if sheet_name not in workbook:
            errors.append(f"Missing worksheet: {sheet_name}")
            continue

        actual_columns = set(workbook[sheet_name].columns)
        missing_columns = required_columns - actual_columns

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            errors.append(
                f"{sheet_name} is missing columns: {missing}"
            )

    return errors


if not DATA_FILE.exists():
    st.error(
        "The demo workbook was not found. "
        "Confirm that it is saved as data/demo_data.xlsx."
    )
    st.stop()

try:
    workbook = load_demo_data(DATA_FILE)
except Exception as error:
    st.error(f"Unable to read the workbook: {error}")
    st.stop()

validation_errors = validate_workbook(workbook)

if validation_errors:
    st.error("The workbook did not pass validation.")

    for error in validation_errors:
        st.write(f"- {error}")

    st.stop()

clients = workbook["Clients"]
pets = workbook["Pets"]
visits = workbook["Visits"]

st.success("Demo operational data loaded and validated.")

client_metric, pet_metric, visit_metric = st.columns(3)

client_metric.metric("Clients", len(clients))
pet_metric.metric("Pets", len(pets))
visit_metric.metric("Scheduled Visits", len(visits))

with st.expander("View source data"):
    clients_tab, pets_tab, visits_tab = st.tabs(
        ["Clients", "Pets", "Visits"]
    )

    with clients_tab:
        st.dataframe(clients, use_container_width=True)

    with pets_tab:
        st.dataframe(pets, use_container_width=True)

    with visits_tab:
        st.dataframe(visits, use_container_width=True)

st.divider()

st.subheader("Select a scheduled visit")

# Convert date and time columns into display-friendly values.
visits["visit_date"] = pd.to_datetime(
    visits["visit_date"]
).dt.strftime("%Y-%m-%d")

visits["visit_time"] = visits["visit_time"].astype(str)

# Join visits with client information.
visit_options = visits.merge(
    clients,
    on="client_id",
    how="left",
)

visit_options["visit_label"] = (
    visit_options["visit_date"]
    + " at "
    + visit_options["visit_time"]
    + " | "
    + visit_options["client_name"]
    + " | "
    + visit_options["service_type"]
)

selected_label = st.selectbox(
    "Choose a visit",
    visit_options["visit_label"].tolist(),
)

selected_visit = visit_options.loc[
    visit_options["visit_label"] == selected_label
].iloc[0]

selected_pets = pets.loc[
    pets["client_id"] == selected_visit["client_id"]
]

st.subheader("Visit context")

overview_col, access_col = st.columns(2)

with overview_col:
    st.markdown("### Visit overview")
    st.write(f"**Client:** {selected_visit['client_name']}")
    st.write(f"**Date:** {selected_visit['visit_date']}")
    st.write(f"**Time:** {selected_visit['visit_time']}")
    st.write(f"**Service:** {selected_visit['service_type']}")
    st.write(f"**Assigned to:** {selected_visit['assigned_to']}")
    st.write(f"**Address:** {selected_visit['address']}")

with access_col:
    st.markdown("### Access and supplies")
    st.write(
        f"**Entry instructions:** "
        f"{selected_visit['entry_instructions']}"
    )
    st.write(
        f"**Supply location:** "
        f"{selected_visit['supply_location']}"
    )
    st.write(
        f"**Communication style:** "
        f"{selected_visit['communication_style']}"
    )

st.markdown("### Pets included in this household")

if selected_pets.empty:
    st.warning("No pets are linked to this client.")
else:
    st.dataframe(
        selected_pets[
            [
                "pet_name",
                "species",
                "feeding",
                "medication",
                "behavior_notes",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("Pre-visit briefing")

if st.button("Generate pre-visit briefing", type="primary"):
    if client is None:
        st.error(
            "OpenAI API key was not found. "
            "Check that OPENAI_API_KEY is saved in your .env file."
        )
    else:
        try:
            with st.spinner("Analyzing visit information..."):
                ai_briefing = generate_ai_briefing(
                    selected_visit,
                    selected_pets,
                )

            st.session_state["ai_briefing"] = ai_briefing

        except Exception as error:
            st.error(f"Briefing generation failed: {error}")

if "ai_briefing" in st.session_state:
    st.subheader("AI Pre-Visit Briefing")
    st.markdown(st.session_state["ai_briefing"])

    st.caption(
        "AI-generated draft grounded in the selected workbook records. "
        "The assigned worker should review the briefing before the visit."
    )

    

    st.markdown("### Supplies")
    st.write(selected_visit["supply_location"])

    missing_information = []

    if selected_pets["feeding"].isna().any():
        missing_information.append("Feeding instructions are incomplete.")

    if selected_pets["medication"].isna().any():
        missing_information.append("Medication information is incomplete.")

    if pd.isna(selected_visit["entry_instructions"]):
        missing_information.append("Entry instructions are missing.")

    if missing_information:
        st.warning("Information requiring confirmation:")
        for item in missing_information:
            st.write(f"- {item}")
    else:
        st.info("No missing operational information detected.")


st.divider()

st.subheader("AI connection")

if client is None:
    st.warning(
        "OpenAI API key was not found. "
        "Check that OPENAI_API_KEY is saved in the .env file."
    )
else:
    st.success("OpenAI API key loaded securely.")

    if st.button("Test AI connection"):
        try:
            with st.spinner("Testing connection..."):
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=(
                        "Reply with exactly this sentence: "
                        "AI connection successful."
                    ),
                )

            st.success(response.output_text)

        except Exception as error:
            st.error(f"AI connection failed: {error}")

st.divider()
st.header("Complete Visit")

st.write(
    "Record the completed work, then generate a client-ready update for review."
)

col1, col2 = st.columns(2)

with col1:
    walk_completed = st.checkbox("Walk completed")
    food_completed = st.checkbox("Food provided")
    water_completed = st.checkbox("Water refreshed")

with col2:
    medication_completed = st.checkbox("Medication administered")
    playtime_completed = st.checkbox("Playtime completed")
    potty_completed = st.checkbox("Potty break completed")

visit_notes = st.text_area(
    "Visit notes",
    placeholder=(
        "Example: Max ate his full meal and was energetic during the walk."
    ),
    height=120,
)

if st.button("Draft client update", type="primary"):
    completed_tasks = []

    if walk_completed:
        completed_tasks.append("Walk completed")
    if food_completed:
        completed_tasks.append("Food provided")
    if water_completed:
        completed_tasks.append("Water refreshed")
    if medication_completed:
        completed_tasks.append("Medication administered")
    if playtime_completed:
        completed_tasks.append("Playtime completed")
    if potty_completed:
        completed_tasks.append("Potty break completed")

    if client is None:
        st.error(
            "OpenAI API key was not found. "
            "Check that OPENAI_API_KEY is saved in your .env file."
        )

    elif not completed_tasks and not visit_notes.strip():
        st.warning(
            "Select at least one completed task or enter a visit note."
        )

    else:
        try:
            with st.spinner("Drafting client update..."):
                client_update = generate_client_update(
                    selected_visit,
                    selected_pets,
                    completed_tasks,
                    visit_notes,
                )

            st.session_state["client_update"] = client_update

        except Exception as error:
            st.error(f"Client update generation failed: {error}")

if "client_update" in st.session_state:
    st.subheader("Client Communication Draft")

    reviewed_update = st.text_area(
        "Review and edit before sending",
        value=st.session_state["client_update"],
        height=180,
        key="reviewed_client_update",
    )

    st.caption(
        "Human review required. This demo drafts the communication but "
        "does not automatically send it."
    )

    if st.button("Approve client update"):
        st.success("Client update approved and ready to send.")
        st.code(reviewed_update)