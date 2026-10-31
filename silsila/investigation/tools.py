"""Validated Diagnostic Tool Registry for Autonomous Agentic Failure Investigation."""

import copy
from typing import Any, Dict, List, Optional
from silsila.models import ExecutionTrace
from silsila.llm_client import (
    extract_arabic_number_from_text,
    extract_leave_type_from_text,
    normalize_arabic_numerals,
)
from silsila.agents.policy import PolicyAgent
from silsila.agents.payroll import PayrollAgent


class DiagnosticTools:
    """Suite of deterministic diagnostic tools invoked autonomously by the Investigator Agent."""

    @staticmethod
    def inspect_trace_step(trace: ExecutionTrace, step_index: int) -> Dict[str, Any]:
        """Inspects the input, output, and execution metadata of a single workflow step."""
        try:
            step_idx = int(step_index)
        except (ValueError, TypeError):
            return {"error": f"Invalid step_index: {step_index}. Must be an integer (0, 1, or 2)."}

        step = trace.get_step(step_idx)
        if not step:
            return {
                "error": f"Step index {step_idx} does not exist in trace. Available steps: 0 to {len(trace.steps)-1}."
            }

        return {
            "step_index": step.step_index,
            "agent_name": step.agent_name,
            "reported_status": step.status,
            "input_state": copy.deepcopy(step.input_state),
            "output_state": copy.deepcopy(step.output_state),
            "metadata": step.metadata,
        }

    @staticmethod
    def compare_agent_handoffs(trace: ExecutionTrace, step_index: int) -> Dict[str, Any]:
        """Compares the output of step (step_index - 1) against the input received at step_index."""
        try:
            step_idx = int(step_index)
        except (ValueError, TypeError):
            return {"error": f"Invalid step_index: {step_index}. Must be an integer."}

        if step_idx <= 0:
            return {
                "step_index": step_idx,
                "note": "Step 0 is the initial intake step; it has no preceding agent handoff.",
                "handoff_consistent": True,
            }

        prev_step = trace.get_step(step_idx - 1)
        curr_step = trace.get_step(step_idx)

        if not prev_step or not curr_step:
            return {"error": f"Cannot compare handoff for step {step_idx}; adjacent step not found."}

        mismatches = {}
        for key, val in curr_step.input_state.items():
            if key in prev_step.output_state and prev_step.output_state[key] != val:
                mismatches[key] = {
                    "previous_emitted": prev_step.output_state[key],
                    "current_received": val,
                }

        is_consistent = len(mismatches) == 0
        return {
            "step_index": step_idx,
            "previous_agent": prev_step.agent_name,
            "current_agent": curr_step.agent_name,
            "handoff_consistent": is_consistent,
            "mismatches": mismatches,
            "diagnosis": (
                "Handoff data was faithfully transferred from upstream agent."
                if is_consistent
                else f"Corruption occurred in handoff transmission on keys: {list(mismatches.keys())}."
            ),
        }

    @staticmethod
    def audit_arabic_semantics(trace: ExecutionTrace, step_index: int = 0) -> Dict[str, Any]:
        """Audits Arabic textual input against numerical and categorical extractions in step output."""
        try:
            step_idx = int(step_index)
        except (ValueError, TypeError):
            step_idx = 0

        step = trace.get_step(step_idx)
        if not step:
            return {"error": f"Step {step_idx} not found."}

        query = trace.input_query or step.input_state.get("query", "")
        if not query:
            return {
                "step_index": step_idx,
                "audit_applicable": False,
                "note": "No raw Arabic query found in this step's inputs.",
            }

        expected_qty = extract_arabic_number_from_text(query)
        normalized_q = normalize_arabic_numerals(query)
        output_qty = step.output_state.get("days")

        has_qty_discrepancy = False
        if expected_qty is not None and output_qty is not None:
            has_qty_discrepancy = int(expected_qty) != int(output_qty)

        expected_leave_type = extract_leave_type_from_text(query)
        output_leave_type = step.output_state.get("leave_type")
        has_type_discrepancy = False
        if expected_leave_type is not None and output_leave_type is not None:
            has_type_discrepancy = output_leave_type.lower() != expected_leave_type.lower()

        discrepancies = []
        if has_qty_discrepancy:
            discrepancies.append(f"quantity mismatch (text={expected_qty}, emitted={output_qty})")
        if has_type_discrepancy:
            discrepancies.append(f"leave_type mismatch (text='{expected_leave_type}', emitted='{output_leave_type}')")

        return {
            "step_index": step_idx,
            "raw_arabic_query": query,
            "normalized_query": normalized_q,
            "expected_quantity_from_text": expected_qty,
            "extracted_quantity_in_output": output_qty,
            "has_quantity_discrepancy": has_qty_discrepancy,
            "expected_leave_type_from_text": expected_leave_type,
            "extracted_leave_type_in_output": output_leave_type,
            "has_leave_type_discrepancy": has_type_discrepancy,
            "audit_verdict": (
                f"DISCREPANCY DETECTED: {'; '.join(discrepancies)}."
                if discrepancies
                else "CONGRUENT: Arabic text and extracted quantities and leave type match."
            ),
        }

    @staticmethod
    def check_business_invariants(trace: ExecutionTrace, step_index: int) -> Dict[str, Any]:
        """Audits mathematical and business conservation laws at a specific workflow step."""
        try:
            step_idx = int(step_index)
        except (ValueError, TypeError):
            return {"error": f"Invalid step_index: {step_index}."}

        step = trace.get_step(step_idx)
        if not step:
            return {"error": f"Step {step_idx} not found."}

        violations: List[str] = []

        if step_idx == 0:
            days = step.output_state.get("days")
            if days is None or days <= 0:
                violations.append(f"Invalid leave days extracted: {days}")
        elif step_idx == 1:
            req = step.input_state.get("days", 0)
            approved = step.output_state.get("approved_paid_days", 0)
            unpaid = step.output_state.get("unpaid_days", 0)
            prev_b = step.output_state.get("previous_balance", 21)
            rem_b = step.output_state.get("remaining_balance", 21)

            if approved + unpaid != req:
                violations.append(f"Quota conservation violated: approved ({approved}) + unpaid ({unpaid}) != requested ({req})")
            if (prev_b - rem_b) != approved:
                expected_rem = prev_b - approved
                violations.append(
                    f"Balance deduction violated: prev ({prev_b}) - rem ({rem_b}) != approved ({approved}). "
                    f"Expected remaining_balance is {expected_rem}."
                )
        elif step_idx == 2:
            base_sal = float(step.input_state.get("base_monthly_salary", 15000.0))
            daily_rate = float(step.input_state.get("daily_rate", 500.0))
            unpaid = int(step.input_state.get("unpaid_days", 0))
            deduction = float(step.output_state.get("salary_deduction", 0.0))
            net = float(step.output_state.get("final_monthly_net", base_sal))

            expected_deduction = unpaid * daily_rate
            if abs(deduction - expected_deduction) > 0.01:
                violations.append(f"Deduction arithmetic violated: computed {deduction}, expected {expected_deduction}")
            if abs(net - (base_sal - deduction)) > 0.01:
                violations.append(f"Net payout violated: computed {net}, expected {base_sal - deduction}")

        is_valid = len(violations) == 0
        return {
            "step_index": step_idx,
            "agent_name": step.agent_name,
            "invariants_satisfied": is_valid,
            "violations": violations,
            "audit_verdict": (
                "All step invariants satisfied." if is_valid else f"Violations detected: {'; '.join(violations)}"
            ),
        }

    ALLOWLISTED_FIELDS = {
        0: {"days", "leave_type", "employee_id"},
        1: {"status", "approved_paid_days", "unpaid_days", "remaining_balance", "previous_balance"},
        2: {"salary_deduction", "final_monthly_net", "net_payout_adjustment", "paid_leave_allowance"},
    }

    @staticmethod
    def test_counterfactual(
        trace: ExecutionTrace,
        step_index: int,
        proposed_patch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Safely simulates isolated downstream re-execution with a proposed patch in a sandbox.
        Rejects unknown fields, updates dependent values, and validates against all invariants.
        """
        try:
            step_idx = int(step_index)
        except (ValueError, TypeError):
            return {"error": f"Invalid step_index: {step_index}."}

        target_step = trace.get_step(step_idx)
        if not target_step:
            return {"error": f"Step {step_idx} not found."}

        if not isinstance(proposed_patch, dict) or not proposed_patch:
            return {
                "step_index": step_idx,
                "tested_patch": proposed_patch if isinstance(proposed_patch, dict) else {},
                "counterfactual_valid": False,
                "simulated_outcome": {},
                "violations_encountered": ["proposed_patch must be a non-empty dictionary of allowlisted field replacements."],
                "verdict": "Counterfactual failed: proposed_patch is empty or invalid.",
            }

        # 1. Reject unknown or unallowlisted fields
        allowed = DiagnosticTools.ALLOWLISTED_FIELDS.get(step_idx, set())
        unknown_keys = set(proposed_patch.keys()) - allowed
        if unknown_keys:
            violations = [
                f"Rejected: unknown or unallowlisted field(s) in proposed_patch: {sorted(list(unknown_keys))}. "
                f"Allowlisted fields for step {step_idx}: {sorted(list(allowed))}."
            ]
            return {
                "step_index": step_idx,
                "tested_patch": proposed_patch,
                "counterfactual_valid": False,
                "simulated_outcome": {},
                "violations_encountered": violations,
                "verdict": f"Counterfactual failed: unknown or unallowlisted field(s) {sorted(list(unknown_keys))}.",
            }

        # Reject no-op patches explicitly
        is_noop = all(target_step.output_state.get(k) == v for k, v in proposed_patch.items())
        if is_noop:
            return {
                "step_index": step_idx,
                "tested_patch": proposed_patch,
                "counterfactual_valid": False,
                "simulated_outcome": {},
                "violations_encountered": [
                    f"Rejected no-op patch: proposed values {proposed_patch} are already identical to the current output of step {step_idx}. "
                    f"A valid repair patch must modify the erroneous field(s)."
                ],
                "verdict": f"Counterfactual failed: proposed_patch is a no-op (no values modified).",
            }

        # Sandbox simulation
        sim_state = copy.deepcopy(target_step.output_state)
        for k, v in proposed_patch.items():
            sim_state[k] = v

        # Context-aware policy and payroll agent configuration
        step_1_orig = trace.get_step(1)
        step_2_orig = trace.get_step(2)

        prev_bal_ctx = int(step_1_orig.output_state.get("previous_balance", 21)) if step_1_orig else 21
        sal_ctx = float(step_2_orig.output_state.get("base_monthly_salary", 15000.0)) if step_2_orig else 15000.0
        rate_ctx = float(step_2_orig.output_state.get("daily_rate", 500.0)) if step_2_orig else 500.0

        policy_agent = PolicyAgent(accrued_balance=prev_bal_ctx)
        payroll_agent = PayrollAgent(monthly_salary=sal_ctx, daily_rate=rate_ctx)

        violations: List[str] = []
        simulated_final: Dict[str, Any] = {}

        if step_idx == 0:
            # Step 0: Intake
            days = sim_state.get("days")
            if days is None or not isinstance(days, int) or days <= 0:
                violations.append(f"Invalid leave days in patch: {days}")
            else:
                # Evidence consistency check: patch must align with original request text
                raw_query = trace.input_query or target_step.input_state.get("query", "")
                if raw_query:
                    exp_qty = extract_arabic_number_from_text(raw_query)
                    if exp_qty is not None and days != exp_qty:
                        violations.append(
                            f"Counterfactual rejected: proposed patch days={days} contradicts original request text evidence (expected {exp_qty} days)."
                        )
                    exp_type = extract_leave_type_from_text(raw_query)
                    patched_type = sim_state.get("leave_type")
                    if exp_type is not None and patched_type is not None and patched_type.lower() != exp_type.lower():
                        violations.append(
                            f"Counterfactual rejected: proposed patch leave_type='{patched_type}' contradicts original request text ('{exp_type}')."
                        )

                policy_out = policy_agent.execute(sim_state)
                payroll_out = payroll_agent.execute(policy_out)

                if policy_out["approved_paid_days"] + policy_out["unpaid_days"] != days:
                    violations.append("Simulated policy allocation mismatch.")
                if policy_out["previous_balance"] - policy_out["remaining_balance"] != policy_out["approved_paid_days"]:
                    violations.append("Simulated policy balance mismatch.")

                simulated_final = {
                    "requested_days": days,
                    "approved_paid_days": policy_out["approved_paid_days"],
                    "unpaid_days": policy_out["unpaid_days"],
                    "remaining_balance": policy_out["remaining_balance"],
                    "salary_deduction": payroll_out["salary_deduction"],
                    "final_monthly_net": payroll_out["final_monthly_net"],
                }

        elif step_idx == 1:
            # Step 1: Policy
            req_days = int(target_step.input_state.get("days", sim_state.get("requested_days", 0)))
            appr = int(sim_state.get("approved_paid_days", 0))
            unpaid = int(sim_state.get("unpaid_days", 0))
            prev_b = int(sim_state.get("previous_balance", prev_bal_ctx))
            rem_b = int(sim_state.get("remaining_balance", prev_bal_ctx))

            if appr + unpaid != req_days:
                violations.append(f"Quota conservation violated: approved ({appr}) + unpaid ({unpaid}) != requested ({req_days})")
            if (prev_b - rem_b) != appr:
                violations.append(f"Balance deduction violated: prev ({prev_b}) - rem ({rem_b}) != approved ({appr})")

            # Replay Payroll with simulated policy output
            payroll_out = payroll_agent.execute(sim_state)
            exp_ded = unpaid * payroll_out["daily_rate"]
            if abs(payroll_out["salary_deduction"] - exp_ded) > 0.01:
                violations.append(f"Payroll deduction violated: computed {payroll_out['salary_deduction']}, expected {exp_ded}")

            simulated_final = {
                "approved_paid_days": appr,
                "unpaid_days": unpaid,
                "remaining_balance": rem_b,
                "salary_deduction": payroll_out["salary_deduction"],
                "final_monthly_net": payroll_out["final_monthly_net"],
            }

        elif step_idx == 2:
            # Step 2: Payroll
            unpaid = int(target_step.input_state.get("unpaid_days", sim_state.get("unpaid_days", 0)))
            daily_rate = float(sim_state.get("daily_rate", rate_ctx))
            base_salary = float(sim_state.get("base_monthly_salary", sal_ctx))

            # If salary_deduction was patched but final_monthly_net was not, recompute dependent values
            if "salary_deduction" in proposed_patch and "final_monthly_net" not in proposed_patch:
                sim_state["final_monthly_net"] = base_salary - float(sim_state["salary_deduction"])
                sim_state["net_payout_adjustment"] = -float(sim_state["salary_deduction"])

            ded = float(sim_state.get("salary_deduction", 0.0))
            net = float(sim_state.get("final_monthly_net", base_salary))
            expected_ded = unpaid * daily_rate
            expected_net = base_salary - ded

            if abs(ded - expected_ded) > 0.01:
                violations.append(f"Deduction arithmetic violated: computed {ded:.2f}, expected {expected_ded:.2f} (unpaid {unpaid} * rate {daily_rate})")
            if abs(net - expected_net) > 0.01:
                violations.append(f"Net payout violated: computed {net:.2f}, expected {expected_net:.2f}")

            simulated_final = copy.deepcopy(sim_state)

        counterfactual_valid = len(violations) == 0
        return {
            "step_index": step_idx,
            "tested_patch": proposed_patch,
            "counterfactual_valid": counterfactual_valid,
            "simulated_outcome": simulated_final,
            "violations_encountered": violations,
            "verdict": (
                "Counterfactual simulation succeeded: proposed patch restores all downstream invariants."
                if counterfactual_valid
                else f"Counterfactual failed: {'; '.join(violations)}"
            ),
        }
