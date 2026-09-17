from pathlib import Path
import pandas as pd


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# CSV file location
CSV_PATH = BASE_DIR / "data" / "support_tickets.csv"


def load_tickets() -> pd.DataFrame:
    """
    Load support tickets from CSV.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found at: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Convert numeric columns
    numeric_columns = [
        "response_time_hrs",
        "resolution_time_hrs",
        "customer_rating"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def get_ticket_count() -> int:
    """
    Return total number of tickets.
    """
    df = load_tickets()
    return len(df)


if __name__ == "__main__":
    df = load_tickets()

    print("Total tickets:", len(df))
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 tickets:")
    print(df.head())