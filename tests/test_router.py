import json
import os
import sqlite3
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.router import SemanticRouter
from app.tools import (
    evaluate_calculator,
    query_revenue_db,
    lookup_faq_cache,
    query_small_llm,
    query_large_llm,
)
from app.cost_meter import calculate_cost, count_tokens


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def router_instance():
    return SemanticRouter()


def test_calculator_valid():
    assert evaluate_calculator("What is 10 + 10?") == "20"
    assert evaluate_calculator("24 * 193") == "4632"
    assert evaluate_calculator("100 - 45") == "55"
    assert evaluate_calculator("144 / 12") == "12"


def test_calculator_division_by_zero():
    res = evaluate_calculator("Calculate 10 divided by 0")
    assert "error" in res.lower() or "division by zero" in res.lower()


def test_calculator_invalid_expression():
    res = evaluate_calculator("What is the capital of France?")
    assert res.startswith("Error:")


def test_revenue_db_lookup():
    res = query_revenue_db("What was our Q3 2023 revenue?")
    assert "$4.2M" in res or "4.2" in res

    res2 = query_revenue_db("Q1 2024 revenue")
    assert "$4.7M" in res2 or "4.7" in res2


def test_faq_cache_lookup():
    res = lookup_faq_cache("What is your refund policy?")
    assert "30-day money-back guarantee" in res

    res_missing = lookup_faq_cache("Completely unknown question not in cache")
    assert "not found" in res_missing.lower()


def test_cost_metering():
    # Cache flat $0.0000
    assert calculate_cost("cache", "test query", "test answer") == 0.0

    # Calculator flat $0.0001
    assert calculate_cost("calculator", "10 + 10", "20") == 0.0001

    # SQL flat $0.0001
    assert calculate_cost("sql", "Q3 Revenue", "Revenue was $4.2M") == 0.0001

    # Small LLM formula: in/1000 * 0.001 + out/1000 * 0.002
    q = "What is the capital of France?"
    ans = "The capital of France is Paris."
    in_tok = count_tokens(q)
    out_tok = count_tokens(ans)
    expected_small = (in_tok / 1000.0) * 0.001 + (out_tok / 1000.0) * 0.002
    assert abs(calculate_cost("small_llm", q, ans) - expected_small) < 1e-7

    # Large LLM formula: in/1000 * 0.010 + out/1000 * 0.030
    expected_large = (in_tok / 1000.0) * 0.010 + (out_tok / 1000.0) * 0.030
    assert abs(calculate_cost("large_llm", q, ans) - expected_large) < 1e-7


def test_semantic_router_routing(router_instance):
    # High confidence math
    cat, path, sim = router_instance.route_details("What is 10 + 10?", 0.1)
    assert path == "calculator"
    assert sim >= 0.1

    # High confidence revenue
    cat, path, sim = router_instance.route_details("Q3 Revenue", 0.1)
    assert path == "sql"

    # Nonsense with high threshold falls back to large_llm
    cat, path, sim = router_instance.route_details("asdfghjklqwerty", 0.9)
    assert path == "large_llm"
    assert cat == "fallback"


def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_api_route_calculator(client):
    res = client.post("/api/v1/route", json={"query": "What is 10 + 10?", "threshold": 0.1})
    assert res.status_code == 200
    data = res.json()
    assert data["path_taken"] == "calculator"
    assert "20" in data["answer"]
    assert abs(data["cost"] - 0.0001) < 1e-7


def test_api_route_division_by_zero(client):
    res = client.post("/api/v1/route", json={"query": "Calculate 10 divided by 0", "threshold": 0.1})
    assert res.status_code in (200, 400)
    data = res.json()
    assert data["path_taken"] == "calculator"
    assert "error" in data["answer"].lower() or "division by zero" in data["answer"].lower()


def test_api_route_sql(client):
    res = client.post("/api/v1/route", json={"query": "Q3 Revenue", "threshold": 0.1})
    assert res.status_code == 200
    data = res.json()
    assert data["path_taken"] == "sql"
    assert abs(data["cost"] - 0.0001) < 1e-7


def test_api_route_fallback(client):
    res = client.post("/api/v1/route", json={"query": "asdfghjklqwerty", "threshold": 0.9})
    assert res.status_code == 200
    data = res.json()
    assert data["path_taken"] == "large_llm"


def test_dataset_integrity():
    assert os.path.exists("data/evaluation_dataset.json")
    with open("data/evaluation_dataset.json") as f:
        ds = json.load(f)
    assert isinstance(ds, list) and len(ds) >= 100
    allowed = {"arithmetic", "revenue", "faq", "general_simple", "complex_reasoning"}
    for item in ds:
        assert "query" in item and "expected_category" in item and "ground_truth" in item
        assert item["expected_category"] in allowed


def test_revenue_db_integrity():
    assert os.path.exists("data/revenue.db")
    conn = sqlite3.connect("data/revenue.db")
    c = conn.cursor()
    c.execute("SELECT count(*) FROM quarterly_revenue")
    count = c.fetchone()[0]
    conn.close()
    assert count >= 4


def test_pareto_curve_integrity():
    assert os.path.exists("output/pareto_curve.json")
    with open("output/pareto_curve.json") as f:
        curve = json.load(f)
    assert isinstance(curve, list) and len(curve) >= 5
    for pt in curve:
        assert "threshold" in pt and "accuracy" in pt and "avg_cost" in pt and "total_fallback_count" in pt
        assert isinstance(pt["threshold"], float)
        assert isinstance(pt["accuracy"], float)
        assert isinstance(pt["avg_cost"], float)


def test_metrics_report_integrity():
    assert os.path.exists("metrics_report.json")
    with open("metrics_report.json") as f:
        mr = json.load(f)
    assert "pareto_optimal_threshold" in mr and "justification" in mr and "misroute_example" in mr
    assert isinstance(mr["pareto_optimal_threshold"], float)
    assert 0.0 <= mr["pareto_optimal_threshold"] <= 1.0
    me = mr["misroute_example"]
    assert "query" in me and "expected_path" in me and "actual_path" in me and "reasoning" in me
    assert me["expected_path"] != me["actual_path"]
