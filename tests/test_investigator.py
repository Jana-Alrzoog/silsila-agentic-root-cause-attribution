"""Unit tests for RootCauseInvestigator causal attribution.

These tests exercise the deterministic rule-based mode to assert exact
contract behaviour (agent identity, step, patch structure).  The agentic
mode is tested separately in test_agentic_investigator.py.
"""

from silsila.pipeline import WorkflowPipeline, FaultInjector
from silsila.investigation.detector import SilentFailureDetector
from silsila.investigation.investigator import RootCauseInvestigator


def test_investigate_attributes_intake_fault():
    pipeline = WorkflowPipeline()
    injector = FaultInjector(
        target_stage="intake",
        mutation={"days": 14},
        fault_description="Intake inflation",
    )
    trace = pipeline.run("أبي إجازة ٤ أيام", fault_injector=injector)
    detector = SilentFailureDetector()
    alert = detector.detect(trace)

    # Use rule_based mode for deterministic contract assertions
    investigator = RootCauseInvestigator(mode="rule_based")
    report = investigator.investigate(trace, alert)

    assert report.faulty_agent == "IntakeAgent"
    assert report.faulty_step == 0
    assert report.suggested_patch == {"days": 4}
    assert report.confidence >= 0.90


def test_investigate_attributes_policy_fault():
    pipeline = WorkflowPipeline()
    injector = FaultInjector(
        target_stage="policy",
        mutation={"remaining_balance": 0},
        fault_description="Policy calculation bug",
    )
    trace = pipeline.run("أبي إجازة 2 يوم", fault_injector=injector)
    detector = SilentFailureDetector()
    alert = detector.detect(trace)

    # Use rule_based mode for deterministic contract assertions
    investigator = RootCauseInvestigator(mode="rule_based")
    report = investigator.investigate(trace, alert)

    assert report.faulty_agent == "PolicyAgent"
    assert report.faulty_step == 1
    assert "remaining_balance" in report.suggested_patch


def test_investigate_attributes_payroll_fault():
    pipeline = WorkflowPipeline()
    injector = FaultInjector(
        target_stage="payroll",
        mutation={"salary_deduction": 3500.0, "final_monthly_net": 11500.0},
        fault_description="Payroll calculation bug",
    )
    trace = pipeline.run("طلب إجازة بدون راتب لمدة 2 يوم", fault_injector=injector)
    detector = SilentFailureDetector()
    alert = detector.detect(trace)

    # Use rule_based mode for deterministic contract assertions
    investigator = RootCauseInvestigator(mode="rule_based")
    report = investigator.investigate(trace, alert)

    assert report.faulty_agent == "PayrollAgent"
    assert report.faulty_step == 2
    assert "salary_deduction" in report.suggested_patch
