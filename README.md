# AI Support Ticket Assistant

An AI-powered application for analyzing customer support tickets using natural-language questions and anomaly detection.

The project uses a local Ollama LLM to understand user questions, FastAPI for REST APIs, Pandas for data processing, and Streamlit for the user interface.

## Features

- Load support tickets from CSV
- Ask questions using natural language
- Use Ollama LLM for query understanding
- Convert natural language into structured queries
- Filter tickets by status and priority
- Analyze customer ratings
- Group tickets by support agent
- Detect anomalous tickets
- FastAPI REST API
- Streamlit dashboard
- Automated testing with Pytest
- Local LLM without a paid API

## Technologies

- Python
- FastAPI
- Pandas
- Ollama
- Llama 3.2 3B
- Streamlit
- Pydantic
- Pytest
- Requests

## Project Structure

```text
ai-support-assistant/
│
├── app/
│   ├── anomaly_detector.py
│   ├── data_loader.py
│   ├── llm.py
│   ├── llm_parser.py
│   ├── main.py
│   └── query_engine.py
│
├── data/
│   └── support_tickets.csv
│
├── tests/
│   ├── test_ai_queries.py
│   ├── test_anomaly_detector.py
│   ├── test_api.py
│   ├── test_data_loader.py
│   ├── test_llm.py
│   ├── test_llm_parser.py
│   └── test_query_engine.py
│
├── ui/
│   └── app.py
│
├── README.md
└── requirements.txt