"""End-to-End Test for the Primary SILSILA Demonstration Scenario.

Scenario Specification:
- Original Arabic input: "أبي إجازة ٥ أيام"
- Injected fault: Intake outputs 15 instead of 5.
- Expected:
  1. System detects the silent failure mismatch.
  2. Identifies IntakeAgent as responsible at Step 0 using execution evidence.
  3. Tests safe isolated correction to 5 days.
  4. Verifies downstream recovery (Policy remaining balance = 16, Payroll net = 15,000 SAR).
  5. The injected fault label is NEVER exposed directly to the Investigator.
"""

from silsila.pipeline import WorkflowPipeline, FaultInjector
from silsila.investigation.detector import SilentFailureDetector
from silsila.investigation.investigator import RootCauseInvestigator
from silsila.investigation.recovery import RecoveryEngine
from silsila.investigation.verifier import RecoveryVerifier


def test_first_demonstration_scenario():
    # 1. Setup workflow and fault injector (Intake mutation: 5 -> 15)
    query = "أبي إجازة ٥ أيام"
    injected_mutation = {"days": 15}
    fault_injector = FaultInjector(
        target_stage="intake",
        mutation=injected_mutation,
        fault_description="Distort extracted Arabic numeral from 5 to 15 days.",
    )

    pipeline = WorkflowPipeline()
    trace = pipeline.run(query=query, fault_injector=fault_injector)

    # 2. Verify that workflow reported "SUCCESS" despite silent failure
    assert len(trace.steps) == 3
    for step in trace.steps:
        assert step.status == "SUCCESS"

    # Step 0 emitted 15 days
    assert trace.get_step(0).output_state["days"] == 15
    # Downstream policy saw 15 days -> 14 paid, 1 unpaid (cap of 14)
    assert trace.get_step(1).output_state["unpaid_days"] == 1
    # Downstream payroll deducted 500 SAR
    assert trace.get_step(2).output_state["salary_deduction"] == 500.0
    assert trace.get_step(2).output_state["final_monthly_net"] == 14500.0

    # 3. Detect the silent failure against independent business invariants
    detector = SilentFailureDetector()
    alert = detector.detect(trace)

    assert alert.detected is True
    assert alert.invariant_violated == "Semantic Quantity Invariant"
    assert "Discrepancy in leave intake" in alert.description
    assert alert.divergent_fields["expected_quantity"] == 5
    assert alert.divergent_fields["extracted_quantity"] == 15

    # 4. Root Cause Attribution (Trace-evidence based, NO ground truth leakage)
    investigator = RootCauseInvestigator()
    attribution = investigator.investigate(trace, alert)

    assert attribution.faulty_agent == "IntakeAgent"
    assert attribution.faulty_step == 0
    assert "٥" in attribution.evidence or "5" in attribution.evidence
    assert attribution.suggested_patch == {"days": 5}

    from silsila.llm_client import OllamaClient
    if OllamaClient().is_available():
        assert attribution.investigation_mode == "agentic_llm"
        assert attribution.confidence >= 0.90
    else:
        # LLM may be unreachable (llm_unavailable_fallback) or reachable but returning
        # malformed output (llm_invalid_output_fallback). Both are valid offline modes.
        valid_offline_modes = {"llm_unavailable_fallback", "llm_invalid_output_fallback"}
        assert attribution.investigation_mode in valid_offline_modes, (
            f"Expected one of {valid_offline_modes}, got {attribution.investigation_mode!r}"
        )
        assert attribution.confidence == 0.80

    # 5. Safe Isolated State Correction & Replay
    recovery_engine = RecoveryEngine()
    recovery = recovery_engine.recover(trace, attribution)

    assert recovery.success is True
    assert recovery.forked_step == 0
    assert recovery.applied_patch == {"days": 5}
    assert len(recovery.replayed_steps) == 3

    # Check recovered output state
    rec_state = recovery.final_recovered_state
    assert rec_state["requested_days"] == 5
    assert rec_state["approved_paid_days"] == 5
    assert rec_state["unpaid_days"] == 0
    assert rec_state["remaining_balance"] == 16
    assert rec_state["salary_deduction"] == 0.0
    assert rec_state["final_monthly_net"] == 15000.0

    # 6. Independent Recovery Verification
    verifier = RecoveryVerifier()
    verified_recovery = verifier.verify(
        original_trace=trace,
        recovery_result=recovery,
        expected_outcome={
            "approved_paid_days": 5,
            "unpaid_days": 0,
            "remaining_balance": 16,
            "salary_deduction": 0.0,
            "final_monthly_net": 15000.0,
        },
    )

    assert verified_recovery.verification_passed is True
    assert "Verification PASSED" in verified_recovery.verification_notes
