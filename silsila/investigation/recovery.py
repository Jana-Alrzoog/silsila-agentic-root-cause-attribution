"""Recovery Engine: Isolates execution state, applies safe patch, and re-executes downstream agents."""

import copy
from typing import Any, Dict, Optional
from silsila.models import (
    ExecutionTrace,
    AttributionReport,
    RecoveryResult,
    TraceStep,
    get_employee_profile,
)
from silsila.agents.policy import PolicyAgent
from silsila.agents.payroll import PayrollAgent


class RecoveryEngine:
    """Safe state rewinding, patching, and isolated replay."""

    ALLOWLISTED_FIELDS = {
        0: {"days", "leave_type", "employee_id"},
        1: {"status", "approved_paid_days", "unpaid_days", "remaining_balance", "previous_balance"},
        2: {"salary_deduction", "final_monthly_net", "net_payout_adjustment", "paid_leave_allowance"},
    }

    def __init__(
        self,
        policy_agent: Optional[PolicyAgent] = None,
        payroll_agent: Optional[PayrollAgent] = None,
    ):
        self.policy_agent = policy_agent or PolicyAgent()
        self.payroll_agent = payroll_agent or PayrollAgent()

    def recover(
        self,
        trace: ExecutionTrace,
        attribution: AttributionReport,
    ) -> RecoveryResult:
        """
        Executes isolated recovery:
        1. Validates allowlisted fields in suggested_patch.
        2. Forks execution state at attribution.faulty_step.
        3. Applies patch and updates dependent values.
        4. Re-executes downstream pipeline stages in isolated sandbox.
        """
        forked_step_idx = attribution.faulty_step
        replayed_steps = []

        # Find the faulty step
        faulty_step = trace.get_step(forked_step_idx)
        if not faulty_step:
            return RecoveryResult(
                success=False,
                original_failure=attribution.evidence,
                applied_patch={},
                forked_step=forked_step_idx,
                replayed_steps=[],
                final_recovered_state={},
                verification_passed=False,
                verification_notes=f"Step index {forked_step_idx} not found in trace.",
            )

        # 1. Enforce allowlisted fields in suggested_patch
        allowed = self.ALLOWLISTED_FIELDS.get(forked_step_idx, set())
        unknown = set(attribution.suggested_patch.keys()) - allowed
        if unknown:
            return RecoveryResult(
                success=False,
                original_failure=attribution.evidence,
                applied_patch=attribution.suggested_patch,
                forked_step=forked_step_idx,
                replayed_steps=[],
                final_recovered_state={},
                verification_passed=False,
                verification_notes=f"Recovery REJECTED: unknown or unallowlisted field(s) in patch: {sorted(list(unknown))}",
            )

        # Clone and patch state in isolated environment
        patched_state = copy.deepcopy(faulty_step.output_state)
        for key, val in attribution.suggested_patch.items():
            patched_state[key] = val

        # Source authoritative context from EmployeeProfile — NOT from trace outputs.
        # A corrupted trace may carry wrong previous_balance or daily_rate; we must
        # not propagate those into the recovery replay.
        step_0_orig = trace.get_step(0)
        employee_id = "EMP-101"
        if step_0_orig:
            employee_id = step_0_orig.input_state.get("employee_id") or step_0_orig.output_state.get("employee_id", "EMP-101")

        profile = get_employee_profile(employee_id)
        prev_bal_ctx = profile.accrued_annual_leave
        sal_ctx = profile.monthly_salary
        rate_ctx = profile.effective_daily_rate()

        replayed_policy_agent = PolicyAgent(accrued_balance=prev_bal_ctx)
        replayed_payroll_agent = PayrollAgent(monthly_salary=sal_ctx, daily_rate=rate_ctx)

        # Update dependent fields on patched state
        if forked_step_idx == 2:
            base_sal = float(patched_state.get("base_monthly_salary", sal_ctx))
            daily_r = float(patched_state.get("daily_rate", rate_ctx))
            unpaid_d = int(faulty_step.input_state.get("unpaid_days", patched_state.get("unpaid_days", 0)))
            ded = float(patched_state.get("salary_deduction", unpaid_d * daily_r))
            patched_state["salary_deduction"] = ded
            patched_state["final_monthly_net"] = base_sal - ded
            patched_state["net_payout_adjustment"] = -ded
        elif forked_step_idx == 1:
            req_days = int(faulty_step.input_state.get("days", 0))
            prev_b = int(patched_state.get("previous_balance", prev_bal_ctx))
            appr_b = int(patched_state.get("approved_paid_days", 0))
            if appr_b + int(patched_state.get("unpaid_days", 0)) != req_days:
                patched_state["unpaid_days"] = max(0, req_days - appr_b)
            if "remaining_balance" not in attribution.suggested_patch or (prev_b - int(patched_state.get("remaining_balance", prev_bal_ctx))) != appr_b:
                patched_state["remaining_balance"] = prev_b - appr_b

        # Record patched step as replay root
        replay_step_0 = TraceStep(
            step_index=forked_step_idx,
            agent_name=faulty_step.agent_name,
            input_state=copy.deepcopy(faulty_step.input_state),
            output_state=copy.deepcopy(patched_state),
            status="RECOVERED_PATCH",
            metadata={"forked_from": faulty_step.step_index, "patch": attribution.suggested_patch},
        )
        replayed_steps.append(replay_step_0)

        # Downstream replay
        current_state = copy.deepcopy(patched_state)
        final_state: Dict[str, Any] = {}

        if forked_step_idx == 0:
            # Step 0 (Intake) was patched -> Re-run Policy (Step 1) and Payroll (Step 2)
            policy_output = replayed_policy_agent.execute(current_state)
            replayed_policy_step = TraceStep(
                step_index=1,
                agent_name=replayed_policy_agent.name,
                input_state=copy.deepcopy(current_state),
                output_state=copy.deepcopy(policy_output),
                status="REPLAYED_SUCCESS",
                metadata={"recovered_execution": True},
            )
            replayed_steps.append(replayed_policy_step)

            # Re-run Payroll (Step 2)
            payroll_output = replayed_payroll_agent.execute(policy_output)
            replayed_payroll_step = TraceStep(
                step_index=2,
                agent_name=replayed_payroll_agent.name,
                input_state=copy.deepcopy(policy_output),
                output_state=copy.deepcopy(payroll_output),
                status="REPLAYED_SUCCESS",
                metadata={"recovered_execution": True},
            )
            replayed_steps.append(replayed_payroll_step)

            final_state = {
                "requested_days": patched_state.get("days"),
                "leave_type": patched_state.get("leave_type"),
                "policy_status": policy_output.get("status"),
                "approved_paid_days": policy_output.get("approved_paid_days"),
                "unpaid_days": policy_output.get("unpaid_days"),
                "remaining_balance": policy_output.get("remaining_balance"),
                "salary_deduction": payroll_output.get("salary_deduction"),
                "final_monthly_net": payroll_output.get("final_monthly_net"),
            }

        elif forked_step_idx == 1:
            # Step 1 (Policy) was patched -> Re-run Payroll (Step 2)
            payroll_output = replayed_payroll_agent.execute(current_state)
            replayed_payroll_step = TraceStep(
                step_index=2,
                agent_name=replayed_payroll_agent.name,
                input_state=copy.deepcopy(current_state),
                output_state=copy.deepcopy(payroll_output),
                status="REPLAYED_SUCCESS",
                metadata={"recovered_execution": True},
            )
            replayed_steps.append(replayed_payroll_step)

            final_state = {
                "requested_days": trace.get_step(0).output_state.get("days") if trace.get_step(0) else None,
                "leave_type": trace.get_step(0).output_state.get("leave_type") if trace.get_step(0) else "annual",
                "policy_status": patched_state.get("status"),
                "approved_paid_days": patched_state.get("approved_paid_days"),
                "unpaid_days": patched_state.get("unpaid_days"),
                "remaining_balance": patched_state.get("remaining_balance"),
                "salary_deduction": payroll_output.get("salary_deduction"),
                "final_monthly_net": payroll_output.get("final_monthly_net"),
            }

        elif forked_step_idx == 2:
            # Step 2 (Payroll) was patched directly
            final_state = {
                "requested_days": trace.get_step(0).output_state.get("days") if trace.get_step(0) else None,
                "leave_type": trace.get_step(0).output_state.get("leave_type") if trace.get_step(0) else "annual",
                "policy_status": trace.get_step(1).output_state.get("status") if trace.get_step(1) else "APPROVED",
                "approved_paid_days": trace.get_step(1).output_state.get("approved_paid_days") if trace.get_step(1) else None,
                "unpaid_days": trace.get_step(1).output_state.get("unpaid_days") if trace.get_step(1) else 0,
                "remaining_balance": trace.get_step(1).output_state.get("remaining_balance") if trace.get_step(1) else 16,
                "salary_deduction": patched_state.get("salary_deduction"),
                "final_monthly_net": patched_state.get("final_monthly_net"),
            }

        return RecoveryResult(
            success=True,
            original_failure=attribution.evidence,
            applied_patch=attribution.suggested_patch,
            forked_step=forked_step_idx,
            replayed_steps=replayed_steps,
            final_recovered_state=final_state,
            verification_passed=False,
            verification_notes="Replay completed. Awaiting independent verification.",
        )
