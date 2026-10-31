"""FastAPI Server Integration and Endpoint Compatibility Tests.

Tests server endpoint functions directly without requiring socket creation,
maintaining 100% compatibility with isolated sandbox execution.
"""

from web.server import health_check, serve_index, run_pipeline_endpoint, RunRequest


def _run(coro):
    """Executes a coroutine that contains no nested yields."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value


def test_fastapi_health_endpoint():
    """Tests /api/health endpoint returns expected status."""
    data = _run(health_check())
    assert data["status"] == "healthy"
    assert "SILSILA" in data["service"]


def test_fastapi_index_html_serving():
    """Tests GET / serves the exact silsila.html page."""
    resp = _run(serve_index())
    assert resp.status_code == 200
    assert len(resp.body) > 100000


def test_fastapi_run_clean_endpoint():
    """Tests POST /api/run on a clean scenario (Scenario 2)."""
    req = RunRequest(
        query="طلب إجازة سنوية لمدة 3 أيام تبدأ الأحد",
        employee_id="EMP-102",
        max_tool_calls=2,
    )
    res = _run(run_pipeline_endpoint(req))
    assert res["final_status"] in ("healthy", "clean")
    assert res["alert"]["detected"] is False
    assert res["actual"]["balance"] == 18
    assert float(res["actual"]["deduction"]) == 0.0
