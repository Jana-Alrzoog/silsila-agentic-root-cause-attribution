"""Single-scenario live smoke test for SILSILA AgenticInvestigator using local Ollama.

Target scenario: IntakeAgent Step 0 fault on Arabic query 'أبي إجازة ٥ أيام'.
Acceptance criteria:
- Qwen2.5:3b returns valid diagnostic responses.
- Meaningful tool-supported diagnosis (test_counterfactual / audit / finish).
- Root cause attributed to IntakeAgent, Step 0.
- Counterfactual validation succeeds.
- Recovery verified by independent RecoveryVerifier.
- investigation_mode == 'agentic_llm'.
"""

import os
import sys
import time

# Ensure silsila_v2 takes precedence over any installed package
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pytest
from silsila.pipeline import WorkflowPipeline, FaultInjector
from silsila.investigation.detector import SilentFailureDetector
from silsila.investigation.investigator import AgenticInvestigator
from silsila.investigation.recovery import RecoveryEngine
from silsila.investigation.verifier import RecoveryVerifier
from silsila.llm_client import OllamaClient


@pytest.mark.llm
def test_live_smoke_intake_step0_arabic():
    """Live smoke test verifying genuine end-to-end agentic recovery with Qwen2.5:3b."""
    client = OllamaClient()
    if not client.is_available():
        pytest.skip("Local Ollama is not accessible in this environment. Mark: llm.")

    query = "أبي إجازة ٥ أيام"
    start_time = time.perf_counter()

    # 1. Forward execution with injected intake numeral mutation (٥ -> 15)
    pipeline = WorkflowPipeline()
    fault = FaultInjector(
        target_stage="intake",
        mutation={"days": 15},
        fault_description="Intake numeral distortion ٥ -> 15",
    )
    trace = pipeline.run(query=query, employee_id="EMP-101", fault_injector=fault)

    # 2. Silent failure detection via independent business invariants
    detector = SilentFailureDetector()
    alert = detector.detect(trace)
    assert alert.detected is True, "Silent failure must be detected."

    # 3. Autonomous agentic investigation via local Ollama
    investigator = AgenticInvestigator(client=client, max_tool_calls=4)
    report = investigator.investigate(trace, alert)

    # 4. Safe recovery in isolated sandbox
    recovery_engine = RecoveryEngine()
    recovery = recovery_engine.recover(trace, report)

    # 5. Independent verification using canonical employee profile
    verifier = RecoveryVerifier()
    verified = verifier.verify(trace, recovery)

    elapsed_s = time.perf_counter() - start_time

    # Diagnostic inspection
    tools_invoked = [s.tool_selected for s in report.investigation_timeline]

    print("\n" + "=" * 60)
    print("SILSILA LIVE SMOKE TEST METRICS (Qwen2.5:3b)")
    print("=" * 60)
    print(f"Elapsed Time:               {elapsed_s:.2f}s")
    print(f"Investigation Latency:      {report.investigation_latency_ms:.2f}ms")
    print(f"Investigation Mode:         {report.investigation_mode}")
    print(f"Patch Source:               {report.patch_source}")
    print(f"Attributed Faulty Agent:    {report.faulty_agent} (Step {report.faulty_step})")
    print(f"Suggested Patch:            {report.suggested_patch}")
    print(f"LLM Candidate Patch:        {report.llm_candidate_patch}")
    print(f"Tools Selected:             {tools_invoked}")
    print(f"Recovery Success:           {recovery.success}")
    print(f"Verification Passed:        {verified.verification_passed}")
    print("=" * 60)

    # Acceptance Assertions
    assert report.investigation_mode == "agentic_llm", (
        f"Expected investigation_mode 'agentic_llm', got '{report.investigation_mode}'"
    )
    assert report.faulty_agent == "IntakeAgent", (
        f"Expected faulty_agent 'IntakeAgent', got '{report.faulty_agent}'"
    )
    assert report.faulty_step == 0, (
        f"Expected faulty_step 0, got {report.faulty_step}"
    )
    assert report.suggested_patch.get("days") == 5, (
        f"Expected suggested_patch days=5, got {report.suggested_patch}"
    )
    assert report.patch_source == "llm_proposed", (
        f"Expected patch_source 'llm_proposed', got '{report.patch_source}'"
    )
    assert report.escalated is False, "Investigation should not escalate for grounded intake discrepancy."
    assert len(report.investigation_timeline) >= 2, (
        f"Expected at least 2 diagnostic timeline steps, got {len(report.investigation_timeline)}"
    )
    assert recovery.success is True, "Recovery replay must succeed."
    assert verified.verification_passed is True, "Verification must confirm statutory balance and deduction."


if __name__ == "__main__":
    test_live_smoke_intake_step0_arabic()
