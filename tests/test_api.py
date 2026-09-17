from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200


def test_ticket_count_endpoint():
    response = client.get("/tickets/count")

    assert response.status_code == 200

    data = response.json()

    assert "total_tickets" in data
    assert data["total_tickets"] == 500


def test_query_endpoint():
    response = client.post(
        "/query",
        json={
            "question": "How many tickets are currently open?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "query" in data


def test_query_endpoint_returns_open_ticket_count():
    response = client.post(
        "/query",
        json={
            "question": "How many tickets are currently open?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"]["result"] == 111


def test_anomalies_endpoint():
    response = client.get("/anomalies")

    assert response.status_code == 200

    data = response.json()

    assert "total_tickets" in data
    assert "total_anomalies" in data
    assert "thresholds" in data
    assert "anomalies" in data


def test_anomalies_endpoint_total_tickets():
    response = client.get("/anomalies")

    assert response.status_code == 200

    data = response.json()

    assert data["total_tickets"] == 500



def test_query_missing_question():
    response = client.post(
        "/query",
        json={}
    )

    assert response.status_code == 422


def test_query_invalid_question_type():
    response = client.post(
        "/query",
        json={
            "question": 123
        }
    )

    assert response.status_code == 422


def test_query_missing_request_body():
    response = client.post("/query")

    assert response.status_code == 422


def test_invalid_endpoint():
    response = client.get("/wrong-endpoint")

    assert response.status_code == 404