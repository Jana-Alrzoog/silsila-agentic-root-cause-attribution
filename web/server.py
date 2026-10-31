"""FastAPI Server for SILSILA Web Integration.

Serves the exact silsila.html interface and exposes POST /api/run
to execute the real Python multi-agent pipeline and Ollama-backed AI investigation.
"""

import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.runner import execute_live_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = os.path.join(BASE_DIR, "silsila.html")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

app = FastAPI(
    title="SILSILA AI Reliability API",
    description="HUMAIN Hackathon Challenge 1 — Autonomous Multi-Agent Failure Recovery",
    version="1.0.0",
)

# Enable CORS for cross-origin and file:// access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount assets directory if available
if os.path.exists(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


class RunRequest(BaseModel):
    query: str = Field(..., description="Arabic leave request text (e.g. 'أبي إجازة ٥ أيام')")
    employee_id: str = Field(default="EMP-101", description="Employee identifier")
    fault_injection: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional fault injection configuration, e.g. {'target_stage': 'intake', 'mutation': {'days': 15}}"
    )
    max_tool_calls: int = Field(default=5, description="Maximum bounded diagnostic tool calls for investigator")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the exact silsila.html interface."""
    if not os.path.exists(HTML_PATH):
        raise HTTPException(status_code=404, detail="silsila.html not found")
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def health_check():
    """Health check endpoint confirming backend readiness."""
    return {
        "status": "healthy",
        "service": "SILSILA Autonomous Agent Reliability Engine",
        "challenge": "HUMAIN Hackathon Challenge 1",
        "pipeline": ["IntakeAgent", "PolicyAgent", "PayrollAgent"],
        "investigator": "AgenticInvestigator (qwen2.5:3b via Ollama)",
        "recovery": "RecoveryEngine (Sandbox Replay) + RecoveryVerifier",
    }


@app.post("/api/run")
async def run_pipeline_endpoint(req: RunRequest):
    """
    Executes the REAL SILSILA pipeline end-to-end:
    - Multi-agent execution DAG
    - Silent failure detection via invariants
    - Autonomous investigation with Ollama tool calling
    - Isolated state recovery
    - Independent verification
    """
    try:
        result = execute_live_pipeline(
            query=req.query,
            employee_id=req.employee_id,
            fault_injection=req.fault_injection,
            max_tool_calls=req.max_tool_calls,
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=False)
