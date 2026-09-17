import json
import re

import ollama
from pydantic import BaseModel, Field


MODEL_NAME = "llama3.2:3b"


class QueryRequest(BaseModel):
    operation: str = Field(default="count")
    field: str = Field(default="ticket_id")
    group_by: str | None = None
    filters: dict = Field(default_factory=dict)


SYSTEM_PROMPT = """
You are a query parser for a customer support ticket system.

Convert the user's question into ONLY one valid JSON object.

The JSON must have this structure:

{
  "operation": "count",
  "field": "ticket_id",
  "group_by": null,
  "filters": {}
}

Available fields:
ticket_id
created_at
category
priority
status
response_time_hrs
resolution_time_hrs
customer_rating
agent_id
issue_summary

Allowed operations:
count
list
group_count

Status mapping:
- open tickets -> {"status": "Open"}
- resolved tickets -> {"status": "Resolved"}
- pending tickets -> {"status": "Pending"}
- in progress tickets -> {"status": "In Progress"}

Priority mapping:
- low priority -> {"priority": "Low"}
- medium priority -> {"priority": "Medium"}
- high priority -> {"priority": "High"}
- critical priority -> {"priority": "Critical"}

Customer rating mapping:
- poor/low customer ratings -> {"customer_rating": "Low"}
- medium customer ratings -> {"customer_rating": "Medium"}
- high customer ratings -> {"customer_rating": "High"}

IMPORTANT DEFINITION:

"unresolved" means every ticket whose status is NOT "Resolved".

For unresolved tickets, use:

{
  "condition": {
    "type": "unresolved"
  }
}

Do NOT convert "unresolved" into "status": "Open".

GROUPING BY AGENT:

If the user asks for tickets grouped by agent, use:

{
  "operation": "group_count",
  "field": "ticket_id",
  "group_by": "agent_id",
  "filters": {}
}

This applies to questions such as:

- Give me tickets by agent
- Give me tickets per agent
- Show tickets by each agent
- Show tickets for each agent
- How many tickets does each agent have?
- Tickets by agent
- Tickets per agent

For:

"Which agent resolved the most tickets?"

use:

{
  "operation": "group_count",
  "field": "ticket_id",
  "group_by": "agent_id",
  "filters": {
    "status": "Resolved"
  }
}

Examples:

User: How many tickets are currently open?

Output:
{
  "operation": "count",
  "field": "ticket_id",
  "filters": {
    "status": "Open"
  }
}

User: How many resolved tickets are there?

Output:
{
  "operation": "count",
  "field": "ticket_id",
  "filters": {
    "status": "Resolved"
  }
}

User: How many high priority tickets are open?

Output:
{
  "operation": "count",
  "field": "ticket_id",
  "filters": {
    "priority": "High",
    "status": "Open"
  }
}

User: How many high priority tickets are unresolved?

Output:
{
  "operation": "count",
  "field": "ticket_id",
  "filters": {
    "priority": "High",
    "condition": {
      "type": "unresolved"
    }
  }
}

User: How many critical tickets are unresolved?

Output:
{
  "operation": "count",
  "field": "ticket_id",
  "filters": {
    "priority": "Critical",
    "condition": {
      "type": "unresolved"
    }
  }
}

User: Give me tickets by agent

Output:
{
  "operation": "group_count",
  "field": "ticket_id",
  "group_by": "agent_id",
  "filters": {}
}

IMPORTANT:
- Always extract filters explicitly mentioned in the question.
- "open" means status = "Open".
- "resolved" means status = "Resolved".
- "pending" means status = "Pending".
- "in progress" means status = "In Progress".
- "unresolved" means status != "Resolved".
- "high priority" means priority = "High".
- "critical" means priority = "Critical".
- Never ignore a filter mentioned by the user.
- Questions asking for tickets by agent must use group_count.
- Return ONLY valid JSON.
- Do not return SQL.
- Do not return explanations.
- Do not use markdown.
"""


def _normalize_query(question: str, query: dict) -> dict:
    """
    Apply deterministic normalization for known business-language
    patterns that the local LLM may occasionally interpret incorrectly.

    The LLM remains responsible for natural-language understanding.
    This function only protects important application-specific mappings.
    """

    question_lower = question.strip().lower()

    # ------------------------------------------------------------
    # TICKETS BY AGENT
    # ------------------------------------------------------------

    agent_group_phrases = (
        "tickets by agent",
        "tickets per agent",
        "tickets by each agent",
        "tickets for each agent",
        "how many tickets does each agent",
    )

    if any(
        phrase in question_lower
        for phrase in agent_group_phrases
    ):
        query["operation"] = "group_count"
        query["field"] = "ticket_id"
        query["group_by"] = "agent_id"

        # Preserve any meaningful filters returned by the LLM.
        if "filters" not in query:
            query["filters"] = {}

    # ------------------------------------------------------------
    # NORMALIZE UNRESOLVED
    # ------------------------------------------------------------

    if "unresolved" in question_lower:
        filters = query.setdefault("filters", {})

        filters.pop("status", None)

        filters["condition"] = {
            "type": "unresolved"
        }

    # ------------------------------------------------------------
    # NORMALIZE EXPLICIT OPEN STATUS
    # ------------------------------------------------------------

    if "open" in question_lower and "unresolved" not in question_lower:
        filters = query.setdefault("filters", {})
        filters["status"] = "Open"

    # ------------------------------------------------------------
    # NORMALIZE EXPLICIT RESOLVED STATUS
    # ------------------------------------------------------------

    if "resolved" in question_lower and "unresolved" not in question_lower:
        filters = query.setdefault("filters", {})
        filters["status"] = "Resolved"

    return query


def parse_with_llm(question: str) -> dict:
    """
    Convert a natural-language question into a structured query
    using the local Ollama LLM.
    """

    if not isinstance(question, str):
        raise TypeError("question must be a string")

    if not question.strip():
        raise ValueError("question cannot be empty")

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    content = response["message"]["content"].strip()

    # ------------------------------------------------------------
    # REMOVE MARKDOWN CODE FENCES
    # ------------------------------------------------------------

    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()

    # ------------------------------------------------------------
    # PARSE JSON
    # ------------------------------------------------------------

    try:
        query = json.loads(content)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {content}"
        ) from exc

    if not isinstance(query, dict):
        raise ValueError(
            "LLM response must be a JSON object"
        )

    # ------------------------------------------------------------
    # NORMALIZE LLM OUTPUT
    # ------------------------------------------------------------

    query = _normalize_query(
        question,
        query,
    )

    # ------------------------------------------------------------
    # VALIDATE STRUCTURE
    # ------------------------------------------------------------

    validated = QueryRequest.model_validate(query)

    return validated.model_dump(
        exclude_none=True
    )