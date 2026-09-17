import pandas as pd

from app.data_loader import load_tickets


def calculate_iqr_bounds(series: pd.Series):
    """
    Calculate anomaly boundaries using IQR.
    """

    clean_series = series.dropna()

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    return lower_bound, upper_bound


def detect_anomalies() -> dict:
    """
    Detect unusual support tickets.

    Anomaly rules:

    1. Unusually high response time
    2. Unusually high resolution time
    3. Very low customer rating
    """

    df = load_tickets()

    # -----------------------------------------
    # Calculate thresholds
    # -----------------------------------------

    response_lower, response_upper = calculate_iqr_bounds(
        df["response_time_hrs"]
    )

    resolution_lower, resolution_upper = calculate_iqr_bounds(
        df["resolution_time_hrs"]
    )

    # -----------------------------------------
    # Detect anomalies
    # -----------------------------------------

    anomaly_mask = (
        (df["response_time_hrs"] > response_upper)
        |
        (df["resolution_time_hrs"] > resolution_upper)
        |
        (df["customer_rating"] <= 2)
    )

    anomalies = df[anomaly_mask].copy()

    # -----------------------------------------
    # Explain anomaly
    # -----------------------------------------

    def get_reason(row):

        reasons = []

        if (
            pd.notna(row["response_time_hrs"])
            and row["response_time_hrs"] > response_upper
        ):
            reasons.append(
                "Unusually high response time"
            )

        if (
            pd.notna(row["resolution_time_hrs"])
            and row["resolution_time_hrs"] > resolution_upper
        ):
            reasons.append(
                "Unusually high resolution time"
            )

        if (
            pd.notna(row["customer_rating"])
            and row["customer_rating"] <= 2
        ):
            reasons.append(
                "Low customer rating"
            )

        return ", ".join(reasons)

    anomalies["anomaly_reason"] = anomalies.apply(
        get_reason,
        axis=1
    )

    # -----------------------------------------
    # Output columns
    # -----------------------------------------

    output_columns = [
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
        "anomaly_reason"
    ]

    output_df = anomalies[output_columns].copy()

    # -----------------------------------------
    # Convert datetime to JSON-safe string
    # -----------------------------------------

    output_df["created_at"] = output_df["created_at"].astype(str)

    # -----------------------------------------
    # Convert NaN to None
    # -----------------------------------------

    output_df = output_df.astype(object)

    output_df = output_df.where(
        pd.notna(output_df),
        None
    )

    anomaly_records = output_df.to_dict(
        orient="records"
    )

    # -----------------------------------------
    # Return result
    # -----------------------------------------

    return {
        "total_tickets": int(len(df)),

        "total_anomalies": int(len(anomalies)),

        "thresholds": {
            "response_time_upper": round(
                float(response_upper),
                2
            ),

            "resolution_time_upper": round(
                float(resolution_upper),
                2
            ),

            "low_rating_threshold": 2
        },

        "anomalies": anomaly_records
    }


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    result = detect_anomalies()

    print(
        "Total tickets:",
        result["total_tickets"]
    )

    print(
        "Total anomalies:",
        result["total_anomalies"]
    )

    print("\nThresholds:")

    print(
        result["thresholds"]
    )

    print("\nFirst 5 anomalies:")

    for ticket in result["anomalies"][:5]:

        print(
            ticket["ticket_id"],
            "->",
            ticket["anomaly_reason"]
        )