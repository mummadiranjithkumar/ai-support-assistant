import re
import pandas as pd

from app.data_loader import load_tickets
from app.llm_parser import parse_with_llm

# ============================================================
# LOAD TICKETS
# ============================================================

def get_tickets():
    """
    Load support tickets from the data loader.
    """
    df = load_tickets().copy()

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(
            df["created_at"],
            errors="coerce"
        )

    return df


# ============================================================
# NORMALIZE VALUES
# ============================================================

def normalize_status(value):
    value = str(value).strip().lower()

    mapping = {
        "open": "Open",
        "opened": "Open",
        "resolved": "Resolved",
        "closed": "Resolved",
        "pending": "Pending",
        "in progress": "In Progress",
        "in-progress": "In Progress",
    }

    return mapping.get(value, str(value).title())


def normalize_priority(value):
    value = str(value).strip().lower()

    mapping = {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "critical": "Critical",
    }

    return mapping.get(value, str(value).title())


# ============================================================
# APPLY ONE FILTER
# ============================================================

def apply_filter(df, key, value):
    """
    Apply a filter to the dataframe.

    Special case:

        customer_rating = Low
        means customer_rating <= 2
    """

    if key not in df.columns:
        return df

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if key == "status":

        value = normalize_status(value)

        return df[
            df["status"]
            .astype(str)
            .str.lower()
            == value.lower()
        ]


    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    if key == "priority":

        value = normalize_priority(value)

        return df[
            df["priority"]
            .astype(str)
            .str.lower()
            == value.lower()
        ]


    # --------------------------------------------------------
    # CUSTOMER RATING
    # --------------------------------------------------------

    if key == "customer_rating":

        ratings = pd.to_numeric(
            df["customer_rating"],
            errors="coerce"
        )

        if isinstance(value, str):

            value_lower = value.strip().lower()

            # LOW = 1 or 2
            if value_lower == "low":

                return df[
                    ratings <= 2
                ]

            # MEDIUM = 3
            if value_lower == "medium":

                return df[
                    ratings == 3
                ]

            # HIGH = 4 or 5
            if value_lower == "high":

                return df[
                    ratings >= 4
                ]

            # Numeric value written as string
            try:

                number = float(value)

                return df[
                    ratings == number
                ]

            except ValueError:

                return df

        # Numeric value
        try:

            number = float(value)

            return df[
                ratings == number
            ]

        except (ValueError, TypeError):

            return df


    # --------------------------------------------------------
    # NORMAL TEXT COLUMN
    # --------------------------------------------------------

    if df[key].dtype == "object":

        return df[
            df[key]
            .astype(str)
            .str.lower()
            == str(value).lower()
        ]


    # --------------------------------------------------------
    # NUMERIC COLUMN
    # --------------------------------------------------------

    try:

        number = float(value)

        return df[
            pd.to_numeric(
                df[key],
                errors="coerce"
            ) == number
        ]

    except (ValueError, TypeError):

        return df


# ============================================================
# APPLY FILTERS
# ============================================================

def apply_filters(df, filters):
    """
    Apply all query filters.
    """

    if not filters:
        return df

    result = df.copy()

    for key, value in filters.items():

        # Special condition
        if key == "condition":

            result = apply_condition(
                result,
                value
            )

            continue


        # Nested condition
        if isinstance(value, dict):

            operator = value.get("operator")
            condition_value = value.get("value")

            if operator:

                result = apply_numeric_condition(
                    result,
                    key,
                    operator,
                    condition_value
                )

            else:

                result = apply_filter(
                    result,
                    key,
                    condition_value
                )

            continue


        # Normal filter
        result = apply_filter(
            result,
            key,
            value
        )

    return result


# ============================================================
# NUMERIC CONDITION
# ============================================================

def apply_numeric_condition(
    df,
    key,
    operator,
    value
):
    """
    Apply:

        >
        <
        >=
        <=
        ==
        !=
    """

    if key not in df.columns:
        return df

    try:

        column = pd.to_numeric(
            df[key],
            errors="coerce"
        )

        number = float(value)

    except (ValueError, TypeError):

        return df


    if operator == ">":

        mask = column > number

    elif operator == "<":

        mask = column < number

    elif operator == ">=":

        mask = column >= number

    elif operator == "<=":

        mask = column <= number

    elif operator in ("=", "=="):

        mask = column == number

    elif operator == "!=":

        mask = column != number

    else:

        return df

    return df[mask]


# ============================================================
# SPECIAL CONDITIONS
# ============================================================

def apply_condition(df, condition):
    """
    Handle special conditions such as unresolved tickets.
    """

    if not condition:
        return df

    condition_type = condition.get("type")


    # --------------------------------------------------------
    # UNRESOLVED
    # --------------------------------------------------------

    if condition_type == "unresolved":

        if "status" not in df.columns:
            return df

        return df[
            df["status"]
            .astype(str)
            .str.lower()
            != "resolved"
        ]


    # --------------------------------------------------------
    # NOT RESOLVED WITHIN X HOURS
    # --------------------------------------------------------

    if condition_type == "not_resolved_within":

        hours = condition.get(
            "hours",
            24
        )

        if (
            "status" not in df.columns
            or "created_at" not in df.columns
        ):
            return df

        reference_time = df["created_at"].max()

        if pd.isna(reference_time):
            return df

        age_hours = (
            reference_time
            - df["created_at"]
        ).dt.total_seconds() / 3600

        mask = (
            (
                df["status"]
                .astype(str)
                .str.lower()
                != "resolved"
            )
            &
            (age_hours > float(hours))
        )

        return df[mask]


    return df


# ============================================================
# COUNT
# ============================================================

def execute_count(df, field="ticket_id"):
    """
    Count matching tickets.
    """

    if field not in df.columns:

        return int(len(df))

    return int(
        df[field].count()
    )


# ============================================================
# LIST
# ============================================================

def execute_list(df, field="ticket_id"):
    """
    Return matching values as a list.
    """

    if field not in df.columns:
        return []

    return (
        df[field]
        .dropna()
        .tolist()
    )


# ============================================================
# GROUP COUNT
# ============================================================

def execute_group_count(
    df,
    field="ticket_id",
    group_by="agent_id"
):
    """
    Group tickets and count them.

    Example:

        Which agent resolved the most tickets?
    """

    if field not in df.columns:
        return []

    if group_by not in df.columns:
        return []

    grouped = (
        df.groupby(group_by)[field]
        .count()
        .reset_index(name="ticket_count")
        .sort_values(
            "ticket_count",
            ascending=False
        )
    )

    return grouped.to_dict(
        orient="records"
    )


# ============================================================
# EXECUTE STRUCTURED QUERY
# ============================================================

def execute_query(query):
    """
    Execute a structured ticket query.
    """

    df = get_tickets()

    operation = query.get(
        "operation",
        "count"
    )

    field = query.get(
        "field",
        "ticket_id"
    )

    filters = query.get(
        "filters",
        {}
    )


    # Apply filters
    filtered_df = apply_filters(
        df,
        filters
    )


    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    if operation == "count":

        result = execute_count(
            filtered_df,
            field
        )

        return {
            "matching_tickets": int(
                len(filtered_df)
            ),
            "result": result
        }


    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------

    if operation == "list":

        result = execute_list(
            filtered_df,
            field
        )

        return {
            "matching_tickets": int(
                len(filtered_df)
            ),
            "result": result
        }


    # --------------------------------------------------------
    # GROUP COUNT
    # --------------------------------------------------------

    if operation == "group_count":

        group_by = query.get(
            "group_by",
            "agent_id"
        )

        result = execute_group_count(
            filtered_df,
            field,
            group_by
        )

        return {
            "matching_tickets": int(
                len(filtered_df)
            ),
            "result": result
        }


    return {
        "matching_tickets": 0,
        "result": None,
        "error": (
            f"Unsupported operation: {operation}"
        )
    }


# ============================================================
# NATURAL LANGUAGE PARSER
# ============================================================

def parse_question(question):
    """
    Convert natural language into a structured query.

    This avoids incorrect interpretations such as:

        customer_rating = "Low"

    and converts it to:

        customer_rating = Low

    which execute_query interprets as:

        customer_rating <= 2
    """

    q = question.strip().lower()


    # ========================================================
    # LOW CUSTOMER RATINGS
    # ========================================================

    if (
        "low customer rating" in q
        or "low customer ratings" in q
        or "poor customer rating" in q
        or "poor customer ratings" in q
    ):

        return {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "customer_rating": "Low"
            }
        }


    # ========================================================
    # HIGH CUSTOMER RATINGS
    # ========================================================

    if (
        "high customer rating" in q
        or "high customer ratings" in q
    ):

        return {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "customer_rating": "High"
            }
        }


    # ========================================================
    # MEDIUM CUSTOMER RATINGS
    # ========================================================

    if (
        "medium customer rating" in q
        or "medium customer ratings" in q
    ):

        return {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "customer_rating": "Medium"
            }
        }


    # ========================================================
    # HIGH PRIORITY + UNRESOLVED
    # ========================================================

    if (
        "high priority" in q
        and (
            "unresolved" in q
            or "open" in q
        )
    ):

        return {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "priority": "High",
                "status": "Open"
            }
        }


    # ========================================================
    # CRITICAL + UNRESOLVED
    # ========================================================

    if (
        "critical" in q
        and (
            "unresolved" in q
            or "open" in q
        )
    ):

        return {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "priority": "Critical",
                "condition": {
                    "type": "unresolved"
                }
            }
        }


    # ========================================================
    # OPEN TICKETS
    # ========================================================

    if (
        "open tickets" in q
        or "tickets are currently open" in q
        or "tickets currently open" in q
        or "how many tickets are open" in q
    ):

        return {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "status": "Open"
            }
        }


    # ========================================================
    # RESOLVED TICKETS BY AGENT
    # ========================================================

    if (
        "which agent" in q
        and "resolved" in q
        and (
            "most" in q
            or "highest" in q
        )
    ):

        return {
            "operation": "group_count",
            "field": "ticket_id",
            "group_by": "agent_id",
            "filters": {
                "status": "Resolved"
            }
        }


    # ========================================================
    # AGENT TICKET COUNT
    # ========================================================

    if (
        "tickets by agent" in q
        or "tickets per agent" in q
    ):

        return {
            "operation": "group_count",
            "field": "ticket_id",
            "group_by": "agent_id",
            "filters": {}
        }


    # ========================================================
    # RESOLVED TICKETS
    # ========================================================

    if (
        "how many resolved tickets" in q
        or "number of resolved tickets" in q
    ):

        return {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "status": "Resolved"
            }
        }


    # ========================================================
    # CRITICAL TICKETS
    # ========================================================

    if (
        "critical tickets" in q
        or "critical ticket" in q
    ):

        return {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "priority": "Critical"
            }
        }


    # ========================================================
    # HIGH PRIORITY TICKETS
    # ========================================================

    if (
        "high priority tickets" in q
        or "high priority ticket" in q
    ):

        return {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "priority": "High"
            }
        }


    # ========================================================
    # MEDIUM PRIORITY TICKETS
    # ========================================================

    if (
        "medium priority tickets" in q
        or "medium priority ticket" in q
    ):

        return {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "priority": "Medium"
            }
        }


    # ========================================================
    # LOW PRIORITY TICKETS
    # ========================================================

    if (
        "low priority tickets" in q
        or "low priority ticket" in q
    ):

        return {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "priority": "Low"
            }
        }


    # ========================================================
    # FALLBACK
    # ========================================================

    return {
        "operation": "count",
        "field": "ticket_id",
        "filters": {}
    }


# ============================================================
# MAIN FUNCTION USED BY main.py
# ============================================================

def answer_question(question):
    try:
        # Use the local Ollama LLM to understand the question
        query = parse_with_llm(question)

    except Exception as exc:
        # Fallback to the existing rule-based parser
        print(f"LLM parser failed: {exc}")
        query = parse_question(question)

    answer = execute_query(query)

    return {
        "query": query,
        "answer": answer
    }




# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_questions = [

        "How many tickets are currently open?",

        "How many high priority tickets are unresolved?",

        "How many critical tickets are unresolved?",

        "How many tickets have low customer ratings?",

        "Which agent resolved the most tickets?",

    ]


    for question in test_questions:

        print("\n" + "=" * 60)

        print(
            "QUESTION:",
            question
        )

        result = answer_question(
            question
        )

        print(
            "QUERY:",
            result["query"]
        )

        print(
            "ANSWER:",
            result["answer"]
        )