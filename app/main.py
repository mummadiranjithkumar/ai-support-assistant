from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.data_loader import load_tickets
from app.query_engine import answer_question
from app.anomaly_detector import detect_anomalies


app = FastAPI(
    title="AI Support Ticket Assistant",
    description="AI-powered support ticket analysis using Ollama",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    """
    Check whether the API is running.
    """

    return {
        "status": "healthy",
        "service": "AI Support Ticket Assistant"
    }


@app.get("/tickets/count")
def ticket_count():
    """
    Return the total number of support tickets.
    """

    df = load_tickets()

    return {
        "total_tickets": len(df)
    }


@app.post("/query")
def query_tickets(request: QueryRequest):
    """
    Ask a natural-language question about support tickets.
    """

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = answer_question(
            request.question
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


@app.get("/anomalies")
def get_anomalies():
    """
    Detect unusual support tickets.
    """

    try:
        return detect_anomalies()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )