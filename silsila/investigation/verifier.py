"""Recovery Verifier: Independently audits recovered state to guarantee full system healing.

Design note — authoritative reference values:
  RecoveryVerifier MUST source expected_balance, monthly_salary, and daily_rate from
  EmployeeProfile (get_employee_profile) — NOT from upstream trace step outputs.
  A corrupted trace may contain wrong previous_balance or daily_rate values.
  Reading them from the trace would cause the verifier to validate against corrupt
  data and falsely accept (or reject) the recovery.
"""

from typing import Any, Dict, Optional
from silsila.models import (
    ExecutionTrace,
    RecoveryResult,
    TraceStep,
    get_employee_profile,
)
from silsila.investigation.detector import SilentFailureDetector
from silsila.llm_client import extract_arabic_number_from_text


class RecoveryVerifier:
    """Independently verifies that the recovered workflow state is valid and satisfies all invariants."""

    def __init__(self, detector: Optional[SilentFailureDetector] = None):
        self.detector = detector or SilentFailureDetector()

    def verify(
        self,
        original_trace: ExecutionTrace,
        recovery_result: RecoveryResult,
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> RecoveryResult:
        """
        Validates that the recovered state fixes the failure and causes no secondary violations.
        Independently recomputes expected outcomes using EmployeeProfile as the authoritative source
        and does not trust Investigator or counterfactual flags.
        Does NOT read salary/balance from potentially corrupted trace step outputs.
        """
        if not recovery_result.success:
            recovery_result.verification_passed = False
            recovery_result.verification_notes = f"RECOVERY_FAILED: {recovery_result.verification_notes}"
            return recovery_result

        # Build synthetic trace representing the healed execution
        healed_steps = []
        if recovery_result.forked_step == 0:
            healed_steps = list(recovery_result.replayed_steps)
        elif recovery_result.forked_step == 1:
            step_0 = original_trace.get_step(0)
            if step_0:
                healed_steps.append(step_0)
            healed_steps.extend(recovery_result.replayed_steps)
        elif recovery_result.forked_step == 2:
            step_0 = original_trace.get_step(0)
            step_1 = original_trace.get_step(1)
            if step_0:
                healed_steps.append(step_0)
            if step_1:
                healed_steps.append(step_1)
            healed_steps.extend(recovery_result.replayed_steps)

        healed_trace = ExecutionTrace(
            workflow_id=f"healed-{original_trace.workflow_id}",
            input_query=original_trace.input_query,
            steps=healed_steps,
            final_output=recovery_result.final_recovered_state,
        )

        # 1. Audit healed trace against all business invariants
        alert = self.detector.detect(healed_trace, expected_outcome=expected_outcome)
        if alert.detected:
            recovery_result.verification_passed = False
            recovery_result.verification_notes = (
                f"Verification REJECTED: Invariant '{alert.invariant_violated}' violated in healed state. "
                f"Details: {alert.description}"
            )
            return recovery_result

        # 2. Independently compute expected ground truth outcome using EmployeeProfile
        #    Never read previous_balance / daily_rate from trace step outputs —
        #    those may be corrupted (that is the exact scenario we are guarding against).
        exp_outcome = dict(expected_outcome or {})
        if not exp_outcome and original_trace.input_query:
            # Derive employee_id: prefer step 0 input (which is the original query state)
            step_0 = original_trace.get_step(0)
            employee_id = "EMP-101"
            if step_0:
                employee_id = step_0.input_state.get("employee_id", employee_id)
                if not employee_id:
                    employee_id = step_0.output_state.get("employee_id", "EMP-101")

            profile = get_employee_profile(employee_id)
            init_balance = profile.accrued_annual_leave
            base_sal = profile.monthly_salary
            daily_rate = profile.effective_daily_rate()

            exp_days = extract_arabic_number_from_text(original_trace.input_query)
            if exp_days is not None:
                # Check if unpaid leave requested in query
                is_unpaid = "بدون راتب" in original_trace.input_query or "غير مدفوع" in original_trace.input_query
                if is_unpaid:
                    exp_approved = 0
                    exp_unpaid = exp_days
                    exp_remaining = init_balance
                else:
                    exp_approved = min(exp_days, 14, init_balance)
                    exp_unpaid = max(0, exp_days - exp_approved)
                    exp_remaining = init_balance - exp_approved

                exp_deduction = float(exp_unpaid * daily_rate)
                exp_net = float(base_sal - exp_deduction)
                exp_outcome["approved_paid_days"] = exp_approved
                exp_outcome["unpaid_days"] = exp_unpaid
                exp_outcome["remaining_balance"] = exp_remaining
                exp_outcome["salary_deduction"] = exp_deduction
                exp_outcome["final_monthly_net"] = exp_net

        # 3. Verify healed final state matches independently recomputed outcome
        if exp_outcome:
            mismatches = []
            for k, exp_val in exp_outcome.items():
                act_val = recovery_result.final_recovered_state.get(k)
                if act_val is not None:
                    if isinstance(exp_val, float):
                        if abs(float(act_val) - exp_val) > 0.01:
                            mismatches.append(f"{k}: expected {exp_val}, got {act_val}")
                    elif act_val != exp_val:
                        mismatches.append(f"{k}: expected {exp_val}, got {act_val}")

            if mismatches:
                recovery_result.verification_passed = False
                recovery_result.verification_notes = (
                    f"Verification REJECTED: Divergence from independently verified business outcome: {', '.join(mismatches)}"
                )
                return recovery_result

        # Verification successful
        recovery_result.verification_passed = True
        recovery_result.verification_notes = (
            "Verification PASSED: All invariants satisfied. "
            "Isolated patch resolved silent failure and restored end-to-end business integrity."
        )
        return recovery_result
