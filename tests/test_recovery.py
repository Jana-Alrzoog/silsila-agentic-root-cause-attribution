"""Unit tests for RecoveryEngine and RecoveryVerifier."""

from silsila.pipeline import WorkflowPipeline, FaultInjector
from silsila.investigation.detector import SilentFailureDetector
from silsila.investigation.investigator import RootCauseInvestigator
from silsila.investigation.recovery import RecoveryEngine
from silsila.investigation.verifier import RecoveryVerifier


def test_isolated_state_recovery_for_policy_fault():
    pipeline = WorkflowPipeline()
    injector = FaultInjector(
        target_stage="policy",
        mutation={"remaining_balance": 0},
        fault_description="Balance zeroed",
    )
    trace = pipeline.run("أبي إجازة 2 يوم", fault_injector=injector)

    detector = SilentFailureDetector()
    alert = detector.detect(trace)
    assert alert.detected is True

    investigator = RootCauseInvestigator()
    attribution = investigator.investigate(trace, alert)
    assert attribution.faulty_agent == "PolicyAgent"

    engine = RecoveryEngine()
    recovery = engine.recover(trace, attribution)
    assert recovery.success is True
    assert recovery.forked_step == 1
    assert recovery.final_recovered_state["remaining_balance"] == 19

    verifier = RecoveryVerifier()
    verified = verifier.verify(trace, recovery)
    assert verified.verification_passed is True


def test_isolated_state_recovery_for_payroll_fault():
    pipeline = WorkflowPipeline()
    injector = FaultInjector(
        target_stage="payroll",
        mutation={"salary_deduction": 4000.0, "final_monthly_net": 11000.0},
        fault_description="Erroneous deduction",
    )
    trace = pipeline.run("طلب إجازة بدون راتب لمدة 2 يوم", fault_injector=injector)

    detector = SilentFailureDetector()
    alert = detector.detect(trace)
    assert alert.detected is True

    investigator = RootCauseInvestigator()
    attribution = investigator.investigate(trace, alert)
    assert attribution.faulty_agent == "PayrollAgent"

    engine = RecoveryEngine()
    recovery = engine.recover(trace, attribution)
    assert recovery.success is True
    assert recovery.forked_step == 2
    assert recovery.final_recovered_state["salary_deduction"] == 1000.0
    assert recovery.final_recovered_state["final_monthly_net"] == 14000.0

    verifier = RecoveryVerifier()
    verified = verifier.verify(trace, recovery)
    assert verified.verification_passed is True
