from pathlib import Path

import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


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
    pet_names = ", ".join(selected_pets["pet_name"].astype(str))

    feeding_items = []
    medication_items = []
    behavior_items = []

    for _, pet in selected_pets.iterrows():
        feeding_items.append(
            f"- **{pet['pet_name']}:** {pet['feeding']}"
        )
        medication_items.append(
            f"- **{pet['pet_name']}:** {pet['medication']}"
        )
        behavior_items.append(
            f"- **{pet['pet_name']}:** {pet['behavior_notes']}"
        )

    st.success("Pre-visit briefing generated.")

    st.markdown("### Visit overview")
    st.write(
        f"Visit **{selected_visit['client_name']}** at "
        f"**{selected_visit['visit_time']}** for a "
        f"**{selected_visit['service_type']}**."
    )
    st.write(f"**Address:** {selected_visit['address']}")
    st.write(f"**Pets:** {pet_names}")

    st.markdown("### Access")
    st.write(selected_visit["entry_instructions"])

    st.markdown("### Feeding")
    st.markdown("\n".join(feeding_items))

    st.markdown("### Medication")
    st.markdown("\n".join(medication_items))

    st.markdown("### Safety and behavior")
    st.markdown("\n".join(behavior_items))

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

st.subheader("Complete visit")

with st.form("visit_completion_form"):
    st.write("Record what happened during the visit.")

    walk_completed = st.checkbox("Walk completed")
    food_provided = st.checkbox("Food provided")
    water_refreshed = st.checkbox("Water refreshed")
    medication_administered = st.checkbox("Medication administered")
    playtime_completed = st.checkbox("Playtime completed")
    potty_completed = st.checkbox("Potty break completed")

    visit_notes = st.text_area(
        "Additional visit notes",
        placeholder=(
            "Example: Max ate all of his food, played fetch for "
            "10 minutes, and seemed calm when I left."
        ),
    )

    generate_update = st.form_submit_button(
        "Draft client update",
        type="primary",
    )

if generate_update:
    completed_activities = []

    if walk_completed:
        completed_activities.append("completed the scheduled walk")

    if food_provided:
        completed_activities.append("provided food")

    if water_refreshed:
        completed_activities.append("refreshed the water")

    if medication_administered:
        completed_activities.append("administered the scheduled medication")

    if playtime_completed:
        completed_activities.append("included playtime")

    if potty_completed:
        completed_activities.append("completed a potty break")

    pet_names = ", ".join(selected_pets["pet_name"].astype(str))

    if completed_activities:
        activities_text = ", ".join(completed_activities)
    else:
        activities_text = "completed the scheduled visit"

    if selected_visit["communication_style"].lower() == "professional":
        greeting = f"Hello {selected_visit['client_name']},"
        closing = "Please let me know if you have any questions."
    else:
        greeting = f"Hi {selected_visit['client_name']}!"
        closing = "Everything is all set. Have a great day!"

    client_update = (
        f"{greeting}\n\n"
        f"I just finished visiting {pet_names}. I {activities_text}."
    )

    if visit_notes.strip():
        client_update += f"\n\nAdditional update: {visit_notes.strip()}"

    client_update += f"\n\n{closing}"

    st.success("Client update drafted.")

    st.text_area(
        "Review and edit before sending",
        value=client_update,
        height=220,
    )

    st.caption(
        "Human approval is required before any message is sent."
    )

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