from app.anomaly_detector import detect_anomalies


def test_anomaly_detection_returns_dictionary():
    result = detect_anomalies()

    assert isinstance(result, dict)


def test_total_tickets_is_500():
    result = detect_anomalies()

    assert result["total_tickets"] == 500


def test_total_anomalies_is_present():
    result = detect_anomalies()

    assert "total_anomalies" in result
    assert isinstance(result["total_anomalies"], int)


def test_anomaly_count_is_65():
    result = detect_anomalies()

    assert result["total_anomalies"] == 65


def test_thresholds_are_present():
    result = detect_anomalies()

    assert "thresholds" in result

    thresholds = result["thresholds"]

    assert "response_time_upper" in thresholds
    assert "resolution_time_upper" in thresholds
    assert "low_rating_threshold" in thresholds


def test_anomaly_list_is_present():
    result = detect_anomalies()

    assert "anomalies" in result
    assert isinstance(result["anomalies"], list)


def test_anomaly_records_have_required_fields():
    result = detect_anomalies()

    anomalies = result["anomalies"]

    assert len(anomalies) > 0

    first_anomaly = anomalies[0]

    required_fields = {
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
        "anomaly_reason",
    }

    assert required_fields.issubset(set(first_anomaly.keys()))