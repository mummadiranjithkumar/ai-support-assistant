from unittest.mock import patch

from app.query_engine import answer_question


def mock_llm_response(content):
    return {
        "message": {
            "content": content
        }
    }


def test_open_query_case_insensitive():
    response = mock_llm_response(
        """
        {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "status": "Open"
            }
        }
        """
    )

    with patch(
        "app.llm_parser.ollama.chat",
        return_value=response
    ):
        result = answer_question(
            "How many OPEN tickets are there?"
        )

    assert result["query"]["operation"] == "count"
    assert result["query"]["filters"]["status"] == "Open"
    assert result["answer"]["result"] == 111


def test_low_rating_query():
    response = mock_llm_response(
        """
        {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "customer_rating": "Low"
            }
        }
        """
    )

    with patch(
        "app.llm_parser.ollama.chat",
        return_value=response
    ):
        result = answer_question(
            "How many tickets have poor customer ratings?"
        )

    assert result["query"]["operation"] == "count"
    assert result["query"]["filters"]["customer_rating"] == "Low"
    assert result["answer"]["result"] == 47


def test_high_priority_open_query():
    response = mock_llm_response(
        """
        {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "priority": "High",
                "status": "Open"
            }
        }
        """
    )

    with patch(
        "app.llm_parser.ollama.chat",
        return_value=response
    ):
        result = answer_question(
            "How many high priority tickets are open?"
        )

    assert result["query"]["operation"] == "count"
    assert result["query"]["filters"]["priority"] == "High"
    assert result["query"]["filters"]["status"] == "Open"
    assert result["answer"]["result"] == 31


def test_critical_unresolved_query():
    response = mock_llm_response(
        """
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
        """
    )

    with patch(
        "app.llm_parser.ollama.chat",
        return_value=response
    ):
        result = answer_question(
            "How many critical tickets are unresolved?"
        )

    assert result["query"]["operation"] == "count"
    assert result["query"]["filters"]["priority"] == "Critical"
    assert result["query"]["filters"]["condition"]["type"] == "unresolved"
    assert result["answer"]["result"] is not None


def test_agent_query():
    response = mock_llm_response(
        """
        {
            "operation": "group_count",
            "field": "ticket_id",
            "group_by": "agent_id",
            "filters": {
                "status": "Resolved"
            }
        }
        """
    )

    with patch(
        "app.llm_parser.ollama.chat",
        return_value=response
    ):
        result = answer_question(
            "Which agent resolved the most tickets?"
        )

    assert result["query"]["operation"] == "group_count"
    assert result["query"]["group_by"] == "agent_id"

    agent_results = result["answer"]["result"]

    assert isinstance(agent_results, list)
    assert len(agent_results) > 0


def test_tickets_by_agent_query():
    response = mock_llm_response(
        """
        {
            "operation": "group_count",
            "field": "ticket_id",
            "group_by": "agent_id",
            "filters": {}
        }
        """
    )

    with patch(
        "app.llm_parser.ollama.chat",
        return_value=response
    ):
        result = answer_question(
            "Give me tickets by agent"
        )

    assert result["query"]["operation"] == "group_count"
    assert result["query"]["group_by"] == "agent_id"

    assert isinstance(
        result["answer"]["result"],
        list
    )

    assert len(
        result["answer"]["result"]
    ) > 0


def test_resolved_ticket_query():
    response = mock_llm_response(
        """
        {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "status": "Resolved"
            }
        }
        """
    )

    with patch(
        "app.llm_parser.ollama.chat",
        return_value=response
    ):
        result = answer_question(
            "How many resolved tickets are there?"
        )

    assert result["query"]["operation"] == "count"
    assert result["query"]["filters"]["status"] == "Resolved"
    assert result["answer"]["result"] is not None


def test_critical_ticket_query():
    response = mock_llm_response(
        """
        {
            "operation": "count",
            "field": "ticket_id",
            "filters": {
                "priority": "Critical"
            }
        }
        """
    )

    with patch(
        "app.llm_parser.ollama.chat",
        return_value=response
    ):
        result = answer_question(
            "How many critical tickets are there?"
        )

    assert result["query"]["operation"] == "count"
    assert result["query"]["filters"]["priority"] == "Critical"
    assert result["answer"]["result"] is not None