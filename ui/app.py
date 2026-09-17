import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Support Ticket Assistant",
    page_icon="🎫",
    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.title("🎫 AI Support Ticket Assistant")

st.write(
    "Ask questions about support tickets using natural language "
    "or explore detected anomalies."
)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("Dashboard")


if st.sidebar.button("Check API"):

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=10
        )

        if response.status_code == 200:

            st.sidebar.success(
                "API is running"
            )

        else:

            st.sidebar.error(
                "API returned an error"
            )

    except requests.RequestException:

        st.sidebar.error(
            "Cannot connect to FastAPI"
        )


# ==================================================
# TICKET COUNT
# ==================================================

try:

    response = requests.get(
        f"{API_URL}/tickets/count",
        timeout=10
    )

    if response.status_code == 200:

        data = response.json()

        st.metric(
            "Total Tickets",
            data["total_tickets"]
        )

except requests.RequestException:

    st.warning(
        "FastAPI is not running. Start it with: "
        "uvicorn app.main:app --reload"
    )


st.divider()


# ==================================================
# NATURAL LANGUAGE QUERY
# ==================================================

st.header("🔎 Ask About Tickets")


question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: How many high priority tickets are unresolved?"
    ),
    height=100,
    key="ticket_question"
)


if st.button(
    "Ask AI",
    key="ask_ai_button"
):

    if not question.strip():

        st.warning(
            "Please enter a question in the box above."
        )

    else:

        with st.spinner(
            "Analyzing your question..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": question
                    },
                    timeout=120
                )

                if response.status_code == 200:

                    data = response.json()

                    answer = data["answer"]
                    query = data["query"]

                    st.success(
                        "Query completed successfully!"
                    )

                    # ----------------------------------
                    # Matching Tickets
                    # ----------------------------------

                    matching_tickets = answer.get(
                        "matching_tickets",
                        0
                    )

                    result = answer.get(
                        "result"
                    )

                    operation = query.get(
                        "operation",
                        "count"
                    )

                    # ----------------------------------
                    # GROUPED RESULT
                    # ----------------------------------

                    if operation == "group_count":

                        st.metric(
                            "Matching Tickets",
                            matching_tickets
                        )

                        st.subheader(
                            "Tickets by Agent"
                        )

                        if result:

                            st.write(
                                f"Found {len(result)} agents."
                            )

                            st.dataframe(
                                result,
                                use_container_width=True,
                                hide_index=True
                            )

                            # ----------------------------------
                            # Find Top Agent(s)
                            # ----------------------------------

                            max_count = max(
                                item.get(
                                    "ticket_count",
                                    0
                                )
                                for item in result
                            )

                            top_agents = [
                                item.get("agent_id")
                                for item in result
                                if item.get(
                                    "ticket_count",
                                    0
                                ) == max_count
                            ]

                            # ----------------------------------
                            # Display Top Agent(s)
                            # ----------------------------------

                            if len(top_agents) == 1:

                                st.success(
                                    f"Top Agent: "
                                    f"{top_agents[0]} "
                                    f"with {max_count} "
                                    f"tickets."
                                )

                            else:

                                st.success(
                                    f"Top Agents: "
                                    f"{', '.join(top_agents)} "
                                    f"with {max_count} "
                                    f"tickets each."
                                )

                        else:

                            st.info(
                                "No grouped results found."
                            )

                    # ----------------------------------
                    # NORMAL LIST RESULT
                    # ----------------------------------

                    elif isinstance(result, list):

                        st.metric(
                            "Matching Tickets",
                            matching_tickets
                        )

                        st.subheader(
                            "Result"
                        )

                        if result:

                            st.write(
                                f"Found {len(result)} results."
                            )

                            st.dataframe(
                                result,
                                use_container_width=True,
                                hide_index=True
                            )

                        else:

                            st.info(
                                "No results found."
                            )

                    # ----------------------------------
                    # NORMAL NUMERIC RESULT
                    # ----------------------------------

                    else:

                        col1, col2 = st.columns(2)

                        with col1:

                            st.metric(
                                "Matching Tickets",
                                matching_tickets
                            )

                        with col2:

                            st.metric(
                                "Result",
                                result
                            )

                    # ----------------------------------
                    # QUERY INTERPRETATION
                    # ----------------------------------

                    st.subheader(
                        "Query Interpretation"
                    )

                    st.json(
                        query
                    )

                else:

                    st.error(
                        f"API Error: {response.text}"
                    )

            except requests.RequestException as exc:

                st.error(
                    f"Could not connect to FastAPI: {exc}"
                )


st.divider()


# ==================================================
# ANOMALY DETECTION
# ==================================================

st.header("🚨 Anomaly Detection")


if st.button(
    "Detect Anomalies",
    key="anomaly_button"
):

    with st.spinner(
        "Detecting unusual tickets..."
    ):

        try:

            response = requests.get(
                f"{API_URL}/anomalies",
                timeout=60
            )

            if response.status_code == 200:

                data = response.json()

                # ----------------------------------
                # Summary Metrics
                # ----------------------------------

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Total Tickets",
                        data["total_tickets"]
                    )

                with col2:

                    st.metric(
                        "Anomalies",
                        data["total_anomalies"]
                    )

                with col3:

                    st.metric(
                        "Response Threshold",
                        f'{data["thresholds"]["response_time_upper"]} hrs'
                    )

                # ----------------------------------
                # Thresholds
                # ----------------------------------

                st.subheader(
                    "Anomaly Thresholds"
                )

                st.write(
                    f'Response time > '
                    f'{data["thresholds"]["response_time_upper"]} hrs'
                )

                st.write(
                    f'Resolution time > '
                    f'{data["thresholds"]["resolution_time_upper"]} hrs'
                )

                st.write(
                    f'Customer rating <= '
                    f'{data["thresholds"]["low_rating_threshold"]}'
                )

                # ----------------------------------
                # Anomaly Table
                # ----------------------------------

                st.subheader(
                    "Detected Anomalies"
                )

                anomalies = data.get(
                    "anomalies",
                    []
                )

                if anomalies:

                    st.dataframe(
                        anomalies,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No anomalies detected."
                    )

            else:

                st.error(
                    f"API Error: {response.text}"
                )

        except requests.RequestException as exc:

            st.error(
                f"Could not connect to API: {exc}"
            )