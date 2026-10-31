"""Silent Failure Detector: Evaluates workflow traces against independent invariants and expectations."""

from typing import Any, Dict, Optional
from silsila.models import ExecutionTrace, SilentFailureAlert
from silsila.llm_client import extract_arabic_number_from_text, extract_leave_type_from_text


class SilentFailureDetector:
    """Detects silent execution failures where agents report SUCCESS but business outcome is invalid."""

    def detect(
        self,
        trace: ExecutionTrace,
        expected_outcome: Optional[Dict[str, Any]] = None,
    ) -> SilentFailureAlert:
        """
        Audits execution trace against business invariants and optional ground truth expectation.
        Does NOT rely on internal injected fault flags!
        """
        intake_step = trace.get_agent_step("IntakeAgent")
        policy_step = trace.get_agent_step("PolicyAgent")
        payroll_step = trace.get_agent_step("PayrollAgent")

        # ----------------------------------------------------
        # Invariant 1: Semantic Quantity Invariant
        # Does the extracted leave days match the numeric intent in the original query?
        # ----------------------------------------------------
        if intake_step:
            query = trace.input_query
            extracted_days = intake_step.output_state.get("days")
            expected_days_from_text = extract_arabic_number_from_text(query)

            if expected_days_from_text is not None and extracted_days is not None:
                if int(extracted_days) != int(expected_days_from_text):
                    return SilentFailureAlert(
                        detected=True,
                        severity="CRITICAL",
                        invariant_violated="Semantic Quantity Invariant",
                        description=(
                            f"Discrepancy in leave intake: Original text specifies '{expected_days_from_text}' days, "
                            f"but IntakeAgent emitted days={extracted_days}."
                        ),
                        divergent_fields={
                            "query": query,
                            "expected_quantity": expected_days_from_text,
                            "extracted_quantity": extracted_days,
                        },
                    )

            # Invariant 1b: Semantic Leave Type Invariant
            # Does the extracted leave type match explicit textual intent?
            expected_leave_type = extract_leave_type_from_text(query)
            extracted_leave_type = intake_step.output_state.get("leave_type")
            if expected_leave_type is not None and extracted_leave_type is not None:
                if extracted_leave_type.lower() != expected_leave_type.lower():
                    return SilentFailureAlert(
                        detected=True,
                        severity="CRITICAL",
                        invariant_violated="Semantic Leave Type Invariant",
                        description=(
                            f"Discrepancy in leave type intake: Original text specifies '{expected_leave_type}' leave, "
                            f"but IntakeAgent emitted leave_type='{extracted_leave_type}'."
                        ),
                        divergent_fields={
                            "query": query,
                            "expected_leave_type": expected_leave_type,
                            "extracted_leave_type": extracted_leave_type,
                        },
                    )

        # ----------------------------------------------------
        # Invariant 2: Policy Conservation Invariant
        # approved_paid_days + unpaid_days must equal requested_days
        # and balance deduction must match approved_paid_days
        # ----------------------------------------------------
        if policy_step:
            requested = policy_step.output_state.get("requested_days", 0)
            approved = policy_step.output_state.get("approved_paid_days", 0)
            unpaid = policy_step.output_state.get("unpaid_days", 0)
            prev_bal = policy_step.output_state.get("previous_balance", 21)
            rem_bal = policy_step.output_state.get("remaining_balance", 21)

            if approved + unpaid != requested:
                return SilentFailureAlert(
                    detected=True,
                    severity="HIGH",
                    invariant_violated="Policy Total Days Conservation",
                    description=(
                        f"Policy allocation mismatch: approved ({approved}) + unpaid ({unpaid}) "
                        f"!= requested ({requested})."
                    ),
                    divergent_fields={
                        "requested": requested,
                        "approved": approved,
                        "unpaid": unpaid,
                    },
                )

            if (prev_bal - rem_bal) != approved:
                return SilentFailureAlert(
                    detected=True,
                    severity="HIGH",
                    invariant_violated="Policy Leave Balance Conservation",
                    description=(
                        f"Balance deduction mismatch: previous ({prev_bal}) - remaining ({rem_bal}) "
                        f"!= approved paid days ({approved})."
                    ),
                    divergent_fields={
                        "previous_balance": prev_bal,
                        "remaining_balance": rem_bal,
                        "approved": approved,
                    },
                )

        # ----------------------------------------------------
        # Invariant 3: Payroll Compensation Invariant
        # salary_deduction must equal unpaid_days * daily_rate
        # final_monthly_net must equal base_monthly_salary - salary_deduction
        # ----------------------------------------------------
        if payroll_step:
            base_sal = payroll_step.output_state.get("base_monthly_salary", 15000.0)
            daily_rate = payroll_step.output_state.get("daily_rate", 500.0)
            unpaid = payroll_step.output_state.get("unpaid_days", 0)
            deduction = payroll_step.output_state.get("salary_deduction", 0.0)
            net = payroll_step.output_state.get("final_monthly_net", base_sal)

            expected_deduction = unpaid * daily_rate
            if abs(deduction - expected_deduction) > 0.01:
                return SilentFailureAlert(
                    detected=True,
                    severity="CRITICAL",
                    invariant_violated="Payroll Deduction Invariant",
                    description=(
                        f"Payroll deduction mismatch: computed {deduction:.2f} SAR, "
                        f"expected {expected_deduction:.2f} SAR (unpaid days {unpaid} * rate {daily_rate})."
                    ),
                    divergent_fields={
                        "actual_deduction": deduction,
                        "expected_deduction": expected_deduction,
                    },
                )

            expected_net = base_sal - deduction
            if abs(net - expected_net) > 0.01:
                return SilentFailureAlert(
                    detected=True,
                    severity="CRITICAL",
                    invariant_violated="Payroll Net Balance Invariant",
                    description=(
                        f"Net salary mismatch: net {net:.2f} SAR != base {base_sal:.2f} - deduction {deduction:.2f} SAR."
                    ),
                    divergent_fields={
                        "actual_net": net,
                        "expected_net": expected_net,
                    },
                )

        # ----------------------------------------------------
        # Invariant 4: Independent Outcome Oracle (if provided for evaluation)
        # ----------------------------------------------------
        if expected_outcome:
            final_out = trace.final_output
            divergences = {}
            for key, exp_val in expected_outcome.items():
                act_val = final_out.get(key)
                if act_val is not None and act_val != exp_val:
                    divergences[key] = {"expected": exp_val, "actual": act_val}

            if divergences:
                return SilentFailureAlert(
                    detected=True,
                    severity="CRITICAL",
                    invariant_violated="Independent Outcome Mismatch",
                    description=f"Workflow final state diverged from expected business outcome on {list(divergences.keys())}.",
                    divergent_fields=divergences,
                )

        # No invariant violations detected
        return SilentFailureAlert(
            detected=False,
            severity="NONE",
            invariant_violated="",
            description="All business invariants passed successfully. No silent failure detected.",
            divergent_fields={},
        )
