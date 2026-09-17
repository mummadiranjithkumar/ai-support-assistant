import pandas as pd
from app.data_loader import load_tickets


def test_load_tickets_returns_dataframe():
    df = load_tickets()
    assert isinstance(df, pd.DataFrame)


def test_dataset_is_not_empty():
    df = load_tickets()
    assert not df.empty


def test_dataset_has_expected_columns():
    df = load_tickets()

    expected_columns = {
        "ticket_id",
        "created_at",
        "category",
        "priority",
        "status",
        "response_time_hrs",
        "resolution_time_hrs",
        "customer_rating",
        "agent_id",
        "issue_summary",
    }

    assert expected_columns.issubset(set(df.columns))


def test_dataset_has_500_tickets():
    df = load_tickets()
    assert len(df) == 500


def test_ticket_ids_are_unique():
    df = load_tickets()
    assert df["ticket_id"].is_unique