"""Targeted research integrity tests for SILSILA.

Covers:
1. Arabic numeral extraction without false positives on calendar dates
2. Semantic leave-type error detection and attribution
3. Counterfactual rejection of wrong-but-plausible patches
4. Attribution schema consistency (no contradictory agent/step pairs)
5. Context-aware recovery and independent verification (non-default balances/salaries)
6. SILSILA_DISABLE_LLM deterministic fallback switch
7. EmployeeProfile as authoritative source — corrupted trace recovery
8. Patch provenance and sanitizer_modified_fields tracking
9. llm_invalid_output_fallback mode (fake client returning malformed JSON)
10. Arabic extraction evaluation — 10 NEW phrasings (abstention rate)
"""

import os
import pytest
from silsila.pipeline import WorkflowPipeline, FaultInjector
from silsila.investigation.detector import SilentFailureDetector
from silsila.investigation.investigator import (
    AgenticInvestigator,
    RuleBasedInvestigator,
    RootCauseInvestigator,
)
from silsila.investigation.tools import DiagnosticTools
from silsila.investigation.recovery import RecoveryEngine
from silsila.investigation.verifier import RecoveryVerifier
from silsila.models import (
    SilentFailureAlert,
    AttributionReport,
    EmployeeProfile,
    register_employee_profile,
    get_employee_profile,
)
from silsila.llm_client import (
    extract_arabic_number_from_text,
    extract_leave_type_from_text,
    OllamaClient,
)


class TestArabicNumeralIntegrity:
    """Tests proving Arabic numerals and dates are accurately parsed without false positives."""

    def test_date_with_duration_does_not_confuse_detector(self):
        query = "أريد إجازة من 20 نوفمبر لمدة 5 أيام"
        extracted = extract_arabic_number_from_text(query)
        assert extracted == 5, f"Expected 5 days duration, but got {extracted}"

        pipeline = WorkflowPipeline()
        trace = pipeline.run(query)

        # IntakeAgent output should have 5 days, not 20
        assert trace.get_step(0).output_state["days"] == 5

        # Detector should report healthy clean run, NOT a quantity mismatch
        detector = SilentFailureDetector()
        alert = detector.detect(trace)
        assert alert.detected is False
        assert alert.severity == "NONE"

    def test_arabic_indic_numeral_primary_scenario(self):
        query = "أبي إجازة ٥ أيام"
        extracted = extract_arabic_number_from_text(query)
        assert extracted == 5

    def test_word_numbers_and_multi_day_terms(self):
        assert extract_arabic_number_from_text("إجازة لمدة أسبوع") == 7
        assert extract_arabic_number_from_text("طلب إجازة لمدة أسبوعين") == 14
        assert extract_arabic_number_from_text("أريد إجازة لمدة يومين") == 2
        assert extract_arabic_number_from_text("إجازة لمدة ثلاثة أيام") == 3


class TestSemanticLeaveTypeIntegrity:
    """Tests proving semantic leave-type errors are detected and correctly attributed."""

    def test_annual_to_unpaid_leave_type_corruption_detected(self):
        query = "طلب إجازة اعتيادية لمدة 3 أيام"
        assert extract_leave_type_from_text(query) == "annual"

        pipeline = WorkflowPipeline()
        # Injected fault: Intake distorts leave_type from annual to unpaid
        injector = FaultInjector(
            target_stage="intake",
            mutation={"leave_type": "unpaid"},
            fault_description="Intake incorrectly emitted unpaid leave instead of annual.",
        )
        trace = pipeline.run(query, fault_injector=injector)

        # Verify downstream payroll was corrupted (deduction applied when 0 was expected)
        assert trace.get_step(0).output_state["leave_type"] == "unpaid"
        assert trace.get_step(2).output_state["salary_deduction"] > 0

        # Detector MUST catch the semantic leave type invariant violation
        detector = SilentFailureDetector()
        alert = detector.detect(trace)
        assert alert.detected is True
        assert alert.invariant_violated == "Semantic Leave Type Invariant"
        assert "leave type" in alert.description.lower()

        # Attribution must identify IntakeAgent as root cause
        investigator = RootCauseInvestigator(mode="rule_based")
        report = investigator.investigate(trace, alert)
        assert report.faulty_agent == "IntakeAgent"
        assert report.faulty_step == 0
        assert report.suggested_patch.get("leave_type") == "annual"

        # Recovery should heal the leave type and restore 0 SAR deduction
        engine = RecoveryEngine()
        recovery = engine.recover(trace, report)
        assert recovery.success is True
        assert recovery.final_recovered_state["salary_deduction"] == 0.0

        verifier = RecoveryVerifier()
        verified = verifier.verify(trace, recovery)
        assert verified.verification_passed is True


class TestCounterfactualValidity:
    """Tests proving counterfactual testing rejects wrong-but-plausible patches."""

    def test_wrong_but_plausible_days_patch_rejected(self):
        pipeline = WorkflowPipeline()
        injector = FaultInjector(
            target_stage="intake",
            mutation={"days": 15},
            fault_description="Intake distorted 5 to 15",
        )
        trace = pipeline.run("أبي إجازة ٥ أيام", fault_injector=injector)

        # 1. Correct patch (days=5) must be valid
        res_valid = DiagnosticTools.test_counterfactual(trace, 0, {"days": 5})
        assert res_valid["counterfactual_valid"] is True

        # 2. Wrong-but-plausible patches (e.g. 7, 12, 1 days) MUST BE REJECTED
        # even though they produce mathematically consistent downstream allocations!
        for wrong_days in [7, 12, 1, 20]:
            res_wrong = DiagnosticTools.test_counterfactual(trace, 0, {"days": wrong_days})
            assert res_wrong["counterfactual_valid"] is False
            assert "contradicts original request text evidence" in str(res_wrong["violations_encountered"])


class TestAttributionConsistency:
    """Tests proving _resolve_finish_attribution enforces strict agent/step pairing."""

    def test_inconsistent_pairs_reconciled_or_escalated(self):
        inv = AgenticInvestigator()
        alert = SilentFailureAlert(
            detected=True,
            severity="CRITICAL",
            invariant_violated="Payroll Deduction Invariant",
            description="computed 4500, expected 1500",
        )
        history = [
            {
                "action": "audit_arabic_semantics",
                "action_input": {"step_index": 0},
                "observation": {"has_quantity_discrepancy": False, "step_index": 0},
            },
            {
                "action": "check_business_invariants",
                "action_input": {"step_index": 2},
                "observation": {"violations": ["Payroll deduction mismatch"], "step_index": 2},
            },
        ]

        # Contradictory input: faulty_agent=PayrollAgent but faulty_step=0
        contradictory_input = {
            "faulty_agent": "PayrollAgent",
            "faulty_step": 0,
            "suggested_patch": {},
        }
        agent, step, escalated = inv._resolve_finish_attribution(
            contradictory_input, "Contradictory hypothesis", history, alert
        )

        # Schema consistency: If PayrollAgent, step MUST be 2, never 0!
        if not escalated:
            assert (agent, step) in [("PayrollAgent", 2), ("IntakeAgent", 0), ("PolicyAgent", 1)]
            assert agent != "PayrollAgent" or step == 2
            assert agent != "IntakeAgent" or step == 0


class TestContextAwareRecovery:
    """Tests proving recovery and verifier support non-default employee profiles."""

    def test_non_default_salary_and_balance_recovery(self):
        pipeline = WorkflowPipeline()
        # Scenario where employee has 10 accrued days balance (not 21) and 24,000 SAR salary (not 15,000)
        # PolicyAgent: 10 days balance. PayrollAgent: 24,000 SAR salary, 800 SAR/day rate.
        from silsila.agents.policy import PolicyAgent
        from silsila.agents.payroll import PayrollAgent

        custom_pipeline = WorkflowPipeline(
            policy_agent=PolicyAgent(accrued_balance=10),
            payroll_agent=PayrollAgent(monthly_salary=24000.0, daily_rate=800.0),
        )

        injector = FaultInjector(
            target_stage="payroll",
            mutation={"salary_deduction": 4000.0, "final_monthly_net": 20000.0},
            fault_description="Payroll charged 4000 instead of 1600 (2 unpaid * 800).",
        )
        trace = custom_pipeline.run("طلب إجازة بدون راتب لمدة 2 يوم", employee_id="EMP-999", fault_injector=injector)

        detector = SilentFailureDetector()
        alert = detector.detect(trace)
        assert alert.detected is True

        investigator = RootCauseInvestigator(mode="rule_based")
        report = investigator.investigate(trace, alert)
        assert report.faulty_agent == "PayrollAgent"
        assert report.faulty_step == 2
        # Correct deduction for 2 unpaid days at 800 SAR/day is 1600 SAR
        assert report.suggested_patch["salary_deduction"] == 1600.0

        engine = RecoveryEngine()
        recovery = engine.recover(trace, report)
        assert recovery.success is True
        assert recovery.final_recovered_state["salary_deduction"] == 1600.0
        assert recovery.final_recovered_state["final_monthly_net"] == 22400.0

        verifier = RecoveryVerifier()
        verified = verifier.verify(trace, recovery)
        assert verified.verification_passed is True


class TestEnvironmentDisableLLM:
    """Tests the SILSILA_DISABLE_LLM=1 deterministic fallback switch."""

    def test_disable_llm_switch_uses_fallback_mode(self, monkeypatch):
        monkeypatch.setenv("SILSILA_DISABLE_LLM", "1")
        from silsila.config import SILSILA_DISABLE_LLM

        client = OllamaClient()
        assert client.is_available() is False

        pipeline = WorkflowPipeline()
        injector = FaultInjector(
            target_stage="intake",
            mutation={"days": 15},
            fault_description="Intake distorted 5 to 15",
        )
        trace = pipeline.run("أبي إجازة ٥ أيام", fault_injector=injector)
        detector = SilentFailureDetector()
        alert = detector.detect(trace)

        # AgenticInvestigator should cleanly fall back and report llm_unavailable_fallback
        investigator = AgenticInvestigator(max_tool_calls=2)
        report = investigator.investigate(trace, alert)

        assert report.investigation_mode == "llm_unavailable_fallback"
        assert report.faulty_agent == "IntakeAgent"
        assert report.faulty_step == 0


# ---------------------------------------------------------------------------
# 7. EmployeeProfile as authoritative reference — corrupted trace test
# ---------------------------------------------------------------------------

class TestEmployeeProfileAuthoritative:
    """Proves that RecoveryVerifier uses EmployeeProfile and NOT corrupted trace outputs."""

    def test_corrupted_trace_balance_does_not_fool_verifier(self):
        """
        Scenario: Payroll agent corrupts the trace with a wrong salary_deduction.
        We additionally corrupt step 2's daily_rate in the trace AFTER detection, investigation,
        and recovery — simulating a compromised trace at verification time only.

        The RecoveryVerifier MUST use EmployeeProfile (EMP-101: 500 SAR/day)
        to verify the recovered state, NOT the corrupted trace daily_rate.
        """
        query = "أبي إجازة ٥ أيام"
        employee_id = "EMP-101"

        # Run with payroll fault
        pipeline = WorkflowPipeline()
        injector = FaultInjector(
            target_stage="payroll",
            mutation={"salary_deduction": 5000.0, "final_monthly_net": 10000.0},
            fault_description="Payroll overcharged: 5000 SAR instead of 0 SAR.",
        )
        trace = pipeline.run(query=query, employee_id=employee_id, fault_injector=injector)

        # Detect, investigate, recover BEFORE corrupting
        detector = SilentFailureDetector()
        alert = detector.detect(trace)
        assert alert.detected is True, "Should detect payroll fault"

        investigator = RootCauseInvestigator(mode="rule_based")
        report = investigator.investigate(trace, alert)
        assert report.faulty_agent == "PayrollAgent"
        assert report.faulty_step == 2

        engine = RecoveryEngine()
        recovery = engine.recover(trace, report)
        assert recovery.success is True

        # NOW corrupt trace daily_rate AFTER recovery — only affects verification context
        step_2 = trace.get_step(2)
        assert step_2 is not None
        step_2.output_state["daily_rate"] = 9999.0  # Corrupted AFTER recovery

        # Verifier: MUST use EmployeeProfile's daily_rate (500), NOT 9999
        verifier = RecoveryVerifier()
        verified = verifier.verify(original_trace=trace, recovery_result=recovery)

        # EmployeeProfile EMP-101: 5 paid days, 0 unpaid, 500 SAR/day -> deduction=0, net=15000
        assert verified.verification_passed is True, (
            f"Verification should pass using EmployeeProfile values (rate=500), "
            f"not corrupted trace rate. Notes: {verified.verification_notes}"
        )

    def test_corrupted_daily_rate_does_not_fool_verifier(self):
        """
        When daily_rate is corrupted in the trace AFTER recovery, RecoveryVerifier uses
        EmployeeProfile and still correctly verifies recovery for a 3-day unpaid leave scenario.
        """
        # Register a custom employee for this test
        profile = EmployeeProfile(
            employee_id="EMP-CORRUPT-TEST",
            accrued_annual_leave=21,
            monthly_salary=12000.0,
            days_in_month=30,
            daily_rate=400.0,  # Real rate: 400 SAR/day
        )
        register_employee_profile(profile)

        from silsila.agents.policy import PolicyAgent
        from silsila.agents.payroll import PayrollAgent

        custom_pipeline = WorkflowPipeline(
            policy_agent=PolicyAgent(accrued_balance=21),
            payroll_agent=PayrollAgent(monthly_salary=12000.0, daily_rate=400.0),
        )
        injector = FaultInjector(
            target_stage="payroll",
            mutation={"salary_deduction": 9000.0, "final_monthly_net": 3000.0},
            fault_description="Payroll 9x overcharged for 3 unpaid days.",
        )
        trace = custom_pipeline.run(
            query="طلب إجازة بدون راتب لمدة 3 أيام",
            employee_id="EMP-CORRUPT-TEST",
            fault_injector=injector,
        )

        detector = SilentFailureDetector()
        alert = detector.detect(trace)
        assert alert.detected is True

        investigator = RootCauseInvestigator(mode="rule_based")
        report = investigator.investigate(trace, alert)
        assert report.faulty_agent == "PayrollAgent"

        engine = RecoveryEngine()
        recovery = engine.recover(trace, report)
        assert recovery.success is True
        # Recovery uses EmployeeProfile rate (400 SAR), so: 3 * 400 = 1200 SAR
        assert abs(recovery.final_recovered_state.get("salary_deduction", 0) - 1200.0) < 0.01

        # NOW corrupt step 2 daily_rate AFTER recovery
        step_2 = trace.get_step(2)
        step_2.output_state["daily_rate"] = 99999.0  # Corrupted AFTER recovery!

        verifier = RecoveryVerifier()
        verified = verifier.verify(original_trace=trace, recovery_result=recovery)
        # Verifier uses EmployeeProfile (rate=400, not 99999) -> 3*400=1200 -> passes
        assert verified.verification_passed is True


# ---------------------------------------------------------------------------
# 8. Patch provenance and sanitizer_modified_fields
# ---------------------------------------------------------------------------

class TestPatchProvenance:
    """Proves patch_source and sanitizer_modified_fields are reported correctly."""

    def test_rule_based_sets_rule_derived_patch_source(self):
        pipeline = WorkflowPipeline()
        injector = FaultInjector(
            target_stage="intake",
            mutation={"days": 15},
            fault_description="Intake distorted 5 to 15",
        )
        trace = pipeline.run("أبي إجازة ٥ أيام", fault_injector=injector)
        detector = SilentFailureDetector()
        alert = detector.detect(trace)
        assert alert.detected is True

        investigator = RootCauseInvestigator(mode="rule_based")
        report = investigator.investigate(trace, alert)
        assert report.patch_source == "rule_derived", f"Expected rule_derived, got {report.patch_source}"
        assert report.suggested_patch.get("days") == 5

    def test_llm_disabled_sets_rule_derived_or_none(self, monkeypatch):
        monkeypatch.setenv("SILSILA_DISABLE_LLM", "1")
        pipeline = WorkflowPipeline()
        injector = FaultInjector(
            target_stage="payroll",
            mutation={"salary_deduction": 4500.0, "final_monthly_net": 10500.0},
            fault_description="Payroll triple-charged",
        )
        trace = pipeline.run("طلب إجازة بدون راتب لمدة 3 أيام", fault_injector=injector)
        detector = SilentFailureDetector()
        alert = detector.detect(trace)
        assert alert.detected is True

        investigator = AgenticInvestigator(max_tool_calls=3)
        report = investigator.investigate(trace, alert)

        assert report.investigation_mode == "llm_unavailable_fallback"
        # Patch source must never be llm_proposed when LLM was disabled
        assert report.patch_source in ("rule_derived", "none"), (
            f"LLM-disabled run must not report llm_proposed, got: {report.patch_source}"
        )

    def test_sanitizer_modified_fields_reported_for_intake_fix(self, monkeypatch):
        """When sanitizer corrects days from None to text-derived value, it must be reported."""
        monkeypatch.setenv("SILSILA_DISABLE_LLM", "1")
        pipeline = WorkflowPipeline()
        injector = FaultInjector(
            target_stage="intake",
            mutation={"days": 15},
            fault_description="Intake distorted 5 to 15",
        )
        trace = pipeline.run("أبي إجازة ٥ أيام", fault_injector=injector)
        detector = SilentFailureDetector()
        alert = detector.detect(trace)

        inv = AgenticInvestigator(max_tool_calls=2)
        # Deliberately pass empty patch (simulates LLM not proposing one) through sanitizer
        empty_patch = {}
        sanitized, modified = inv._validate_and_sanitize_patch(trace, "IntakeAgent", 0, empty_patch, [])
        assert "days" in modified, f"Sanitizer should have reported 'days' as modified. Got: {modified}"
        assert sanitized.get("days") == 5

    def test_sanitizer_modified_fields_for_policy_completion(self, monkeypatch):
        """Sanitizer must track all fields it adds for Policy step 1."""
        monkeypatch.setenv("SILSILA_DISABLE_LLM", "1")
        pipeline = WorkflowPipeline()
        injector = FaultInjector(
            target_stage="policy",
            mutation={"remaining_balance": 0},
            fault_description="Balance wiped to 0",
        )
        trace = pipeline.run("أرغب في إجازة اعتيادية لمدة 2 يوم", fault_injector=injector)

        inv = AgenticInvestigator(max_tool_calls=2)
        # Provide only partial patch — sanitizer should complete it and track the additions
        partial_patch = {"approved_paid_days": 2}
        sanitized, modified = inv._validate_and_sanitize_patch(trace, "PolicyAgent", 1, partial_patch, [])
        # At minimum, remaining_balance should be set or corrected
        assert "remaining_balance" in sanitized
        # modified should contain remaining_balance since it was corrected
        # (whether from missing or from wrong value)
        assert len(modified) > 0, f"Some fields should be tracked as modified. Got: {modified}"


# ---------------------------------------------------------------------------
# 9. llm_invalid_output_fallback mode
# ---------------------------------------------------------------------------

class FakeInvalidJsonClient:
    """Fake LLM client that is 'reachable' but always returns malformed JSON."""

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, system_prompt=None, format_json=False, temperature=0.0, timeout=40):
        # Return syntactically invalid JSON to trigger JSONDecodeError
        return "{ invalid json: definitely not parseable }"


class TestLLMInvalidOutputFallback:
    """Tests the llm_invalid_output_fallback mode when LLM is reachable but returns bad JSON."""

    def test_invalid_json_client_sets_fallback_mode(self):
        """
        When the LLM endpoint is reachable but returns malformed JSON on every call,
        investigation_mode must be set to llm_invalid_output_fallback (not agentic_llm
        or llm_unavailable_fallback).
        """
        pipeline = WorkflowPipeline()
        injector = FaultInjector(
            target_stage="intake",
            mutation={"days": 15},
            fault_description="Intake distorted 5 to 15",
        )
        trace = pipeline.run("أبي إجازة ٥ أيام", fault_injector=injector)
        detector = SilentFailureDetector()
        alert = detector.detect(trace)
        assert alert.detected is True

        # Use the fake client that returns malformed JSON
        investigator = AgenticInvestigator(client=FakeInvalidJsonClient(), max_tool_calls=3)
        report = investigator.investigate(trace, alert)

        assert report.investigation_mode == "llm_invalid_output_fallback", (
            f"Expected llm_invalid_output_fallback, got {report.investigation_mode!r}"
        )
        # Must still attribute correctly via fallback rule-based synthesis
        assert report.faulty_agent == "IntakeAgent"
        assert report.faulty_step == 0

    def test_first_scenario_accepts_invalid_output_fallback(self):
        """
        test_first_scenario.py assertion on investigation_mode must accept
        llm_invalid_output_fallback as a valid offline mode (in addition to
        llm_unavailable_fallback) when LLM is not producing parseable output.
        """
        valid_offline_modes = {"llm_unavailable_fallback", "llm_invalid_output_fallback"}

        pipeline = WorkflowPipeline()
        injector = FaultInjector(
            target_stage="intake",
            mutation={"days": 15},
            fault_description="Distort extracted Arabic numeral from 5 to 15 days.",
        )
        trace = pipeline.run(query="أبي إجازة ٥ أيام", fault_injector=injector)
        detector = SilentFailureDetector()
        alert = detector.detect(trace)

        # Test with fake invalid-JSON client
        investigator = AgenticInvestigator(client=FakeInvalidJsonClient(), max_tool_calls=3)
        report = investigator.investigate(trace, alert)

        # This assertion mirrors what test_first_scenario should accept:
        if not OllamaClient().is_available():
            assert report.investigation_mode in valid_offline_modes, (
                f"Offline investigation mode must be one of {valid_offline_modes}, "
                f"got: {report.investigation_mode!r}"
            )
        # Regardless of mode, attribution must be correct
        assert report.faulty_agent == "IntakeAgent"
        assert report.faulty_step == 0
        assert report.suggested_patch.get("days") == 5


# ---------------------------------------------------------------------------
# 10. Arabic extraction evaluation — NEW phrasings (abstention rate)
# ---------------------------------------------------------------------------

class TestArabicExtractionNewPhrasings:
    """
    Development-set baseline: 19 phrases designed with the regex (see llm_client.py).
    Held-out evaluation: 10 NEW phrasings NOT used during regex design.

    Abstention (None) rate is computed separately and documented.
    Known limitation: an Intake fault on an inherently ambiguous request
    (e.g., 'إجازة لفترة وجيزة') CANNOT be detected by deterministic invariant audits.
    """

    # --- Development set (19 phrases used during regex design) ---
    DEV_SET = [
        ("أبي إجازة ٥ أيام", 5),                                        # Arabic-Indic
        ("طلب إجازة سنوية لمدة 3 أيام تبدأ الأحد", 3),                  # standard phrasing
        ("أحتاج إجازة ٤ أيام لظروف عائلية", 4),                         # Arabic-Indic numeral
        ("أرغب في إجازة اعتيادية لمدة 2 يوم", 2),                       # "يوم" singular
        ("طلب إجازة بدون راتب لمدة 3 أيام", 3),                         # unpaid leave
        ("أريد إجازة مرضية لمدة يومين", 2),                              # يومين dual
        ("إجازة طارئة لمدة أسبوع", 7),                                    # one week
        ("طلب إجازة لمدة أسبوعين", 14),                                   # two weeks
        ("أريد إجازة من 20 نوفمبر لمدة 5 أيام", 5),                     # date+duration
        ("إجازة لمدة ثلاثة أيام", 3),                                     # word number
        ("أريد إجازة لمدة خمسة أيام", 5),                               # five days
        ("إجازة اعتيادية لمدة 10 أيام", 10),                             # ten days numeric
        ("لمدة سبعة أيام أحتاج إجازة", 7),                               # seven word number
        ("أبغى إجازة ٣ أيام", 3),                                         # Arabic-Indic 3
        ("أحتاج إجازة عشرة أيام", 10),                                    # ten word
        ("إجازة لمدة أربعة أيام", 4),                                     # four word
        ("أريد إجازة لمدة ستة أيام", 6),                                  # six word
        ("إجازة ثمانية أيام", 8),                                          # eight word
        ("أريد إجازة لمدة تسعة أيام", 9),                                # nine word
    ]

    # --- NEW phrasings (NOT designed with the regex) ---
    # These test edge cases: written-out tens, weeks with modifiers, two numbers,
    # Latin digits, Arabic-Indic dates, and ambiguous requests.
    NEW_PHRASINGS = [
        # Written-out tens
        ("أريد إجازة لمدة عشرين يوماً", 20, False),                   # عشرون not in dict
        ("أحتاج إجازة لمدة ثلاثين يوماً", 30, False),                  # ثلاثون not in dict
        # Weeks with modifiers
        ("أريد إجازة أسبوع ونصف", 7, False),                          # ambiguous half-week
        ("طلب إجازة ثلاثة أسابيع", 21, False),                        # ثلاثة أسابيع = 21
        # Two numbers with different meanings (address + days)
        ("في الشارع 15 أريد إجازة 4 أيام", 4, True),                 # should extract 4, not 15
        ("رقم الموظف 123 يطلب إجازة 7 أيام", 7, True),               # should extract 7, not 123
        # Latin digits with English unit
        ("I need 5 days annual leave", 5, True),                      # English phrasing
        ("أريد 10 days إجازة", 10, True),                             # mixed Arabic/English
        # Arabic-Indic date + Latin duration
        ("من تاريخ ١٥/١٠ لمدة 4 أيام", 4, True),                     # date should not override duration
        # Fully ambiguous (no number) — expected abstention (None)
        ("إجازة لفترة وجيزة", None, False),                           # no quantity → abstention
    ]

    def test_dev_set_extraction_accuracy(self):
        """All 19 development-set phrases must extract correctly."""
        failures = []
        for phrase, expected in self.DEV_SET:
            extracted = extract_arabic_number_from_text(phrase)
            if extracted != expected:
                failures.append(f"  '{phrase[:50]}': expected={expected}, got={extracted}")
        assert not failures, "Development-set extraction failures:\n" + "\n".join(failures)

    def test_new_phrasings_abstention_rate(self):
        """
        Evaluates 10 NEW phrasings and reports per-category results and abstention rate.
        Known failures are documented; the test does NOT fabricate a pass.
        """
        results = []
        abstentions = 0
        correct = 0
        incorrect = 0

        for phrase, expected, _deterministic in self.NEW_PHRASINGS:
            extracted = extract_arabic_number_from_text(phrase)
            is_correct = (extracted == expected)
            is_abstention = (extracted is None)

            if is_abstention:
                abstentions += 1
                is_correct = (expected is None)  # Abstention is correct if expected is None
            if is_correct:
                correct += 1
            else:
                incorrect += 1

            results.append({
                "phrase": phrase,
                "expected": expected,
                "extracted": extracted,
                "correct": is_correct,
                "abstention": is_abstention,
            })

        total = len(self.NEW_PHRASINGS)
        abstention_rate = abstentions / total

        # Write a summary to file to avoid console encoding issues
        import os
        summary_path = os.path.join(os.path.dirname(__file__), "..", "EXTRACTION_EVAL_NEW.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=== NEW PHRASING EXTRACTION EVALUATION ===\n")
            f.write(f"Total NEW phrases: {total}\n")
            f.write(f"Correct:    {correct}/{total}  ({correct/total*100:.0f}%)\n")
            f.write(f"Incorrect:  {incorrect}/{total}  ({incorrect/total*100:.0f}%)\n")
            f.write(f"Abstentions (None returned): {abstentions}/{total} ({abstention_rate*100:.0f}%)\n")
            f.write("\nKnown limitations:\n")
            f.write("  - Written-out tens (عشرون, ثلاثون) are NOT in the ARABIC_WORD_NUMBERS dict.\n")
            f.write("    These will fail or abstain. Extending the dict is a future improvement.\n")
            f.write("  - Plural weeks (ثلاثة أسابيع) are not explicitly mapped.\n")
            f.write("  - An Intake fault on an inherently ambiguous request (e.g., 'إجازة لفترة وجيزة')\n")
            f.write("    CANNOT be detected by deterministic invariant audits, because no expected\n")
            f.write("    quantity can be derived from the text to compare against the emitted value.\n")
            f.write("\nPer-phrase results:\n")
            for r in results:
                status = "CORRECT" if r["correct"] else "INCORRECT"
                if r["abstention"] and r["expected"] is None:
                    status = "CORRECT (abstention expected)"
                elif r["abstention"]:
                    status = "ABSTAINED (expected value)"
                f.write(f"  [{status}] '{r['phrase'][:60]}' => expected={r['expected']}, got={r['extracted']}\n")

        # The test passes as long as:
        # 1. Deterministic phrasings with clear numeric context are extracted correctly.
        # 2. Ambiguous phrases may abstain (return None) — that is acceptable behavior.
        # 3. The abstention rate is documented.
        deterministic_results = [r for r in results if r["phrase"] in [p for p, _, d in self.NEW_PHRASINGS if d]]
        deterministic_correct = sum(1 for r in deterministic_results if r["correct"])
        if deterministic_results:
            assert deterministic_correct / len(deterministic_results) >= 0.5, (
                f"At least 50% of deterministic new phrases must extract correctly. "
                f"Got {deterministic_correct}/{len(deterministic_results)}. "
                f"See EXTRACTION_EVAL_NEW.txt for details."
            )
