"""Unit tests for SilentFailureDetector invariant audits."""

from silsila.pipeline import WorkflowPipeline, FaultInjector
from silsila.investigation.detector import SilentFailureDetector


def test_detector_passes_on_clean_run():
    pipeline = WorkflowPipeline()
    trace = pipeline.run("أبي إجازة 3 أيام")
    detector = SilentFailureDetector()

    alert = detector.detect(trace)
    assert alert.detected is False
    assert alert.severity == "NONE"


def test_detector_flags_intake_quantity_mismatch():
    pipeline = WorkflowPipeline()
    injector = FaultInjector(
        target_stage="intake",
        mutation={"days": 12},
        fault_description="Intake misparse",
    )
    trace = pipeline.run("أبي إجازة ٤ أيام", fault_injector=injector)
    detector = SilentFailureDetector()

    alert = detector.detect(trace)
    assert alert.detected is True
    assert alert.invariant_violated == "Semantic Quantity Invariant"
    assert alert.divergent_fields["expected_quantity"] == 4
    assert alert.divergent_fields["extracted_quantity"] == 12


def test_detector_flags_policy_conservation_violation():
    pipeline = WorkflowPipeline()
    injector = FaultInjector(
        target_stage="policy",
        mutation={"remaining_balance": 0},  # Balance dropped improperly
        fault_description="Corrupt balance",
    )
    trace = pipeline.run("أبي إجازة 2 يوم", fault_injector=injector)
    detector = SilentFailureDetector()

    alert = detector.detect(trace)
    assert alert.detected is True
    assert "Balance" in alert.invariant_violated


def test_detector_flags_payroll_deduction_arithmetic():
    pipeline = WorkflowPipeline()
    injector = FaultInjector(
        target_stage="payroll",
        mutation={"salary_deduction": 3000.0, "final_monthly_net": 12000.0},
        fault_description="Improper deduction",
    )
    trace = pipeline.run("أبي إجازة 3 أيام", fault_injector=injector)
    detector = SilentFailureDetector()

    alert = detector.detect(trace)
    assert alert.detected is True
    assert "Payroll Deduction Invariant" in alert.invariant_violated
