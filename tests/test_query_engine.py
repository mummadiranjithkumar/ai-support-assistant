from unittest.mock import patch

from app.query_engine import answer_question


def mock_llm_response(content):
    return {
        "message": {
            "content": content
        }
    }


def test_open_ticket_count():
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
            "How many tickets are currently open?"
        )

    assert isinstance(result, dict)
    assert result["query"]["operation"] == "count"
    assert result["answer"]["result"] == 111


def test_high_priority_unresolved_count():
    response = mock_llm_response(
        """
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
        """
    )

    with patch(
        "app.llm_parser.ollama.chat",
        return_value=response
    ):
        result = answer_question(
            "How many high priority tickets are unresolved?"
        )

    assert isinstance(result, dict)
    assert result["query"]["operation"] == "count"
    assert result["query"]["filters"]["priority"] == "High"
    assert result["query"]["filters"]["condition"]["type"] == "unresolved"

    # Unresolved means status != Resolved.
    assert result["answer"]["result"] == 49


def test_low_customer_rating_count():
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
            "How many tickets have low customer ratings?"
        )

    assert isinstance(result, dict)
    assert result["query"]["operation"] == "count"
    assert result["query"]["filters"]["customer_rating"] == "Low"
    assert result["answer"]["result"] == 47


def test_agent_group_count():
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

    assert isinstance(result, dict)

    assert result["query"]["operation"] == "group_count"

    assert isinstance(
        result["answer"]["result"],
        list
    )

    assert len(
        result["answer"]["result"]
    ) > 0

    first_result = result["answer"]["result"][0]

    assert "agent_id" in first_result
    assert "ticket_count" in first_result


def test_matching_tickets_is_present():
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
            "How many tickets are currently open?"
        )

    assert "answer" in result
    assert "matching_tickets" in result["answer"]

    assert isinstance(
        result["answer"]["matching_tickets"],
        int
    )


def test_query_result_is_not_none():
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
            "How many tickets have low customer ratings?"
        )

    assert "answer" in result
    assert result["answer"]["result"] is not None