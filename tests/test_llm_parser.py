from unittest.mock import patch

from app.llm_parser import parse_with_llm


def test_llm_parser_returns_structured_query():
    mock_response = {
        "message": {
            "content": """
            {
                "operation": "count",
                "field": "ticket_id",
                "filters": {
                    "status": "Open"
                }
            }
            """
        }
    }

    with patch("app.llm_parser.ollama.chat", return_value=mock_response):
        result = parse_with_llm("How many tickets are currently open?")

    assert result["operation"] == "count"
    assert result["field"] == "ticket_id"
    assert result["filters"]["status"] == "Open"