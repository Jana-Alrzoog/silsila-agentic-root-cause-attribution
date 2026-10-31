"""SILSILA Live Pipeline Runner for Web Integration.

Executes the REAL SILSILA multi-agent pipeline and formats results
into structured responses and replayable timeline events for silsila.html.
"""

import copy
import json
import time
from typing import Any, Dict, List, Optional

from silsila.pipeline import WorkflowPipeline, FaultInjector
from silsila.investigation.detector import SilentFailureDetector
from silsila.investigation.investigator import AgenticInvestigator
from silsila.investigation.recovery import RecoveryEngine
from silsila.investigation.verifier import RecoveryVerifier


def execute_live_pipeline(
    query: str,
    employee_id: str = "EMP-101",
    fault_injection: Optional[Dict[str, Any]] = None,
    max_tool_calls: int = 5,
) -> Dict[str, Any]:
    """
    Executes the real SILSILA pipeline end-to-end:
    1. WorkflowPipeline (Intake -> Policy -> Payroll)
    2. SilentFailureDetector (Audit invariants)
    3. AgenticInvestigator (Real Ollama qwen2.5:3b bounded ReAct loop)
    4. RecoveryEngine (Isolated sandbox state fork & replay)
    5. RecoveryVerifier (Independent invariant verification)

    Returns structured execution results and formatted timeline events.
    """
    start_total = time.perf_counter()

    # 1. Setup real fault injector if specified
    fault_injector = None
    if fault_injection and fault_injection.get("target_stage") and fault_injection.get("target_stage") != "none":
        fault_injector = FaultInjector(
            target_stage=fault_injection.get("target_stage"),
            mutation=fault_injection.get("mutation", {}),
            fault_description=fault_injection.get("fault_description", "Injected silent failure"),
        )

    # 2. Execute real multi-agent workflow
    pipeline = WorkflowPipeline()
    trace = pipeline.run(
        query=query,
        employee_id=employee_id,
        fault_injector=fault_injector,
    )

    # 3. Detect silent failure using independent invariants
    detector = SilentFailureDetector()
    alert = detector.detect(trace)

    # 4. Investigate root cause with AgenticInvestigator
    report = None
    recovery = None
    verified = None
    inv_duration_ms = 0.0

    if alert.detected:
        inv_start = time.perf_counter()
        investigator = AgenticInvestigator(max_tool_calls=max_tool_calls)
        report = investigator.investigate(trace, alert)
        inv_duration_ms = (time.perf_counter() - inv_start) * 1000

        # 5. Safe recovery in isolated sandbox replay
        if not report.escalated:
            recovery_engine = RecoveryEngine()
            recovery = recovery_engine.recover(trace, report)

            # 6. Independent verification
            verifier = RecoveryVerifier()
            verified = verifier.verify(trace, recovery)

    total_duration_ms = (time.perf_counter() - start_total) * 1000

    # Determine decisive final status
    if not alert.detected:
        final_status = "healthy"
    elif report and report.escalated:
        final_status = "escalated"
    elif recovery and recovery.success and verified and verified.verification_passed:
        final_status = "verified_recovery"
    else:
        final_status = "failed_recovery"

    # Extract step states
    step_0 = trace.get_step(0)
    step_1 = trace.get_step(1)
    step_2 = trace.get_step(2)

    intake_out = step_0.output_state if step_0 else {}
    policy_out = step_1.output_state if step_1 else {}
    payroll_out = step_2.output_state if step_2 else {}

    # Extract balances and deductions accurately from actual execution
    from silsila.llm_client import extract_arabic_number_from_text
    exp_qty = extract_arabic_number_from_text(query)
    if exp_qty is None:
        exp_qty = int(intake_out.get("days", 0))

    initial_balance = int(policy_out.get("previous_balance", 21))
    corrupted_balance = int(policy_out.get("remaining_balance", 0))
    corrupted_deduction = float(payroll_out.get("salary_deduction", 0.0))
    daily_rate = float(payroll_out.get("daily_rate", 500.0 if "15000" in str(payroll_out) else 400.0))

    leave_type = intake_out.get("leave_type", "annual")
    if leave_type == "unpaid":
        exp_approved = 0
        exp_unpaid = exp_qty
        exp_balance = initial_balance
        expected_deduction = exp_unpaid * daily_rate
    else:
        exp_approved = min(exp_qty, 14, initial_balance)
        exp_unpaid = max(0, exp_qty - exp_approved)
        exp_balance = initial_balance - exp_approved
        expected_deduction = exp_unpaid * daily_rate

    # Verified recovered values
    rec_balance = None
    rec_deduction = None
    if recovery and recovery.success and recovery.final_recovered_state:
        rec_balance = int(recovery.final_recovered_state.get("remaining_balance", exp_balance))
        rec_deduction = float(recovery.final_recovered_state.get("salary_deduction", expected_deduction))

    # Build timeline events compatible with silsila.html
    events = _build_html_events(
        query=query,
        employee_id=employee_id,
        trace=trace,
        alert=alert,
        report=report,
        recovery=recovery,
        verified=verified,
        final_status=final_status,
        expected_deduction=expected_deduction,
        expected_balance=exp_balance,
        corrupted_deduction=corrupted_deduction,
        corrupted_balance=corrupted_balance,
        rec_deduction=rec_deduction,
        rec_balance=rec_balance,
    )

    # Build comparison data
    faulty_name = report.faulty_agent.lower().replace("agent", "") if report else "none"
    comparison = {
        "baseline": {
            "blamed": "payroll",
            "status": "recovered" if alert.detected else "nominal",
            "truly_recovered": False if alert.detected else True,
        },
        "silsila": {
            "blamed": faulty_name if alert.detected else "none",
            "status": "recovered" if final_status == "verified_recovery" else ("escalated" if final_status == "escalated" else "nominal"),
            "truly_recovered": True if final_status in ("verified_recovery", "healthy") else False,
        },
    }

    # Format trace steps for JSON serialization
    serialized_steps = []
    for s in trace.steps:
        serialized_steps.append({
            "step_index": s.step_index,
            "agent_name": s.agent_name,
            "status": s.status,
            "timestamp": s.timestamp,
            "input_state": s.input_state,
            "output_state": s.output_state,
            "metadata": s.metadata,
        })

    # Format investigation timeline
    serialized_inv_timeline = []
    if report and report.investigation_timeline:
        for t in report.investigation_timeline:
            serialized_inv_timeline.append({
                "step_number": t.step_number,
                "tool_selected": t.tool_selected,
                "thought": t.thought,
                "tool_arguments": t.tool_arguments,
                "observation": t.observation,
                "updated_hypothesis": t.updated_hypothesis,
            })

    return {
        "success": True,
        "original_request": query,
        "employee_id": employee_id,
        "final_status": final_status,
        "duration_ms": round(total_duration_ms, 2),
        "investigation_duration_ms": round(inv_duration_ms, 2),
        "expected": {
            "deduction": f"{expected_deduction:.2f}",
            "balance": exp_balance,
        },
        "corrupted": {
            "deduction": f"{corrupted_deduction:.2f}",
            "balance": corrupted_balance,
        },
        "verified_after": {
            "deduction": f"{rec_deduction:.2f}" if rec_deduction is not None else None,
            "balance": rec_balance,
        },
        "actual": {
            "deduction": f"{corrupted_deduction:.2f}",
            "balance": corrupted_balance,
        },
        "trace": {
            "workflow_id": trace.workflow_id,
            "steps": serialized_steps,
            "final_output": trace.final_output,
        },
        "alert": {
            "detected": alert.detected,
            "invariant_violated": alert.invariant_violated if alert.detected else None,
            "description": alert.description if alert.detected else "All invariants passed.",
            "divergent_fields": alert.divergent_fields if alert.detected else {},
        },
        "investigation": {
            "faulty_agent": report.faulty_agent if report else None,
            "faulty_step": report.faulty_step if report else None,
            "confidence": report.confidence if report else 1.0,
            "evidence": report.evidence if report else "Nominal execution.",
            "suggested_patch": report.suggested_patch if report else {},
            "escalated": report.escalated if report else False,
            "investigation_mode": report.investigation_mode if report else "none",
            "patch_source": getattr(report, "patch_source", None) if report else None,
            "llm_candidate_patch": getattr(report, "llm_candidate_patch", None) if report else None,
            "rule_candidate_patch": getattr(report, "rule_candidate_patch", None) if report else None,
            "sanitizer_modified_fields": getattr(report, "sanitizer_modified_fields", None) if report else None,
            "timeline": serialized_inv_timeline,
        } if report else None,
        "recovery": {
            "success": recovery.success if recovery else False,
            "forked_step": recovery.forked_step if recovery else None,
            "applied_patch": recovery.applied_patch if recovery else {},
            "final_recovered_state": recovery.final_recovered_state if recovery else {},
            "verification_notes": recovery.verification_notes if recovery else None,
        } if recovery else None,
        "verification": {
            "verification_passed": verified.verification_passed if verified else (True if not alert.detected else False),
            "verification_notes": verified.verification_notes if verified else ("All invariants passed nominal audit." if not alert.detected else "Unverified recovery state."),
        } if verified or not alert.detected else None,
        "comparison": comparison,
        "events": events,
    }


def _build_html_events(
    query: str,
    employee_id: str,
    trace: Any,
    alert: Any,
    report: Optional[Any],
    recovery: Optional[Any],
    verified: Optional[Any],
    final_status: str,
    expected_deduction: float,
    expected_balance: int,
    corrupted_deduction: float,
    corrupted_balance: int,
    rec_deduction: Optional[float],
    rec_balance: Optional[int],
) -> List[Dict[str, Any]]:
    """Generates sequential playback events compatible with silsila.html."""
    events = []
    t_idx = 0

    step_0 = trace.get_step(0)
    step_1 = trace.get_step(1)
    step_2 = trace.get_step(2)

    intake_out = step_0.output_state if step_0 else {}
    policy_out = step_1.output_state if step_1 else {}
    payroll_out = step_2.output_state if step_2 else {}

    # Event 0: Request Received
    events.append({
        "t": t_idx,
        "kind": "request",
        "title": "Request received",
        "body": query,
        "status": "info",
        "nodes": {"intake": "idle", "policy": "idle", "payroll": "idle"},
        "agent": None,
        "data": {
            "salary": 12000,
            "leave_balance": 21,
            "employee_id": employee_id,
        },
    })
    t_idx += 1

    # Event 1: IntakeAgent Ran
    events.append({
        "t": t_idx,
        "kind": "agent_run",
        "title": "Intake ran",
        "body": f"days: {intake_out.get('days')}, leave_type: {intake_out.get('leave_type')}",
        "status": "neutral",
        "nodes": {"intake": "ok", "policy": "idle", "payroll": "idle"},
        "agent": "intake",
        "data": intake_out,
    })
    t_idx += 1

    # Event 2: PolicyAgent Ran
    events.append({
        "t": t_idx,
        "kind": "agent_run",
        "title": "Policy ran",
        "body": f"approved_paid: {policy_out.get('approved_paid_days')}, unpaid: {policy_out.get('unpaid_days')}, remaining_balance: {policy_out.get('remaining_balance')}",
        "status": "neutral",
        "nodes": {"intake": "ok", "policy": "ok", "payroll": "idle"},
        "agent": "policy",
        "data": policy_out,
    })
    t_idx += 1

    # Event 3: PayrollAgent Ran
    events.append({
        "t": t_idx,
        "kind": "agent_run",
        "title": "Payroll ran",
        "body": f"salary_deduction: {corrupted_deduction:,.2f} SAR, final_monthly_net: {payroll_out.get('final_monthly_net', 12000):,.2f} SAR",
        "status": "neutral",
        "nodes": {"intake": "ok", "policy": "ok", "payroll": "ok"},
        "agent": "payroll",
        "data": payroll_out,
    })
    t_idx += 1

    if not alert.detected:
        # Clean execution: no invariant violated
        events.append({
            "t": t_idx,
            "kind": "resolved",
            "title": "Clean execution verified",
            "body": f"All invariants nominal: {intake_out.get('days')} days approved, deduction is {corrupted_deduction:,.2f} SAR, balance is {corrupted_balance} days.",
            "status": "success",
            "nodes": {"intake": "ok", "policy": "ok", "payroll": "ok"},
            "agent": None,
            "data": {
                "result": {"deduction": f"{corrupted_deduction:.2f}", "balance": corrupted_balance},
                "expected": {"deduction": f"{expected_deduction:.2f}", "balance": expected_balance},
                "is_clean": True,
            },
        })
        return events

    # Event 4: Silent Failure Detected by Invariant Audit
    events.append({
        "t": t_idx,
        "kind": "outcome_alert",
        "title": "Outcome violates business invariant",
        "body": f"{alert.invariant_violated}: {alert.description}",
        "status": "alert",
        "nodes": {"intake": "ok", "policy": "ok", "payroll": "surfaced"},
        "agent": "payroll",
        "data": {
            "actual": {"deduction": f"{corrupted_deduction:.2f}", "balance": corrupted_balance},
            "expected": {"deduction": f"{expected_deduction:.2f}", "balance": expected_balance},
            "alert": {
                "invariant": alert.invariant_violated,
                "description": alert.description,
            },
        },
    })
    t_idx += 1

    faulty_tag = "intake"
    if report and report.faulty_step == 1:
        faulty_tag = "policy"
    elif report and report.faulty_step == 2:
        faulty_tag = "payroll"

    # Event 5..N: Investigation Tool Calls
    if report and report.investigation_timeline:
        for t_step in report.investigation_timeline:
            events.append({
                "t": t_idx,
                "kind": "contract_violation" if "contract" in t_step.tool_selected else "investigation_step",
                "title": f"Investigator called: {t_step.tool_selected}",
                "body": f"Hypothesis: {t_step.thought[:140]}... Observation: {json.dumps(t_step.observation)[:120]}",
                "status": "alert" if "contract" in t_step.tool_selected else "info",
                "nodes": {
                    "intake": "root" if faulty_tag == "intake" else "ok",
                    "policy": "root" if faulty_tag == "policy" else "ok",
                    "payroll": "surfaced",
                },
                "agent": faulty_tag,
                "data": {
                    "tool": t_step.tool_selected,
                    "arguments": t_step.tool_arguments,
                    "observation": t_step.observation,
                },
            })
            t_idx += 1

    # Event N+1: Causal Attribution Verdict
    if report:
        events.append({
            "t": t_idx,
            "kind": "attribution",
            "title": f"Root cause attributed to {report.faulty_agent}",
            "body": f"Decisive failure origin: Step {report.faulty_step}. Evidence: {report.evidence}",
            "status": "alert" if not report.escalated else "warning",
            "nodes": {
                "intake": "root" if faulty_tag == "intake" else "ok",
                "policy": "root" if faulty_tag == "policy" else "ok",
                "payroll": "root" if faulty_tag == "payroll" else "surfaced",
            },
            "agent": faulty_tag,
            "data": {
                "agent": report.faulty_agent,
                "step": report.faulty_step,
                "evidence": report.evidence,
                "patch": report.suggested_patch,
            },
        })
        t_idx += 1

    # Event N+2: Sandbox Patch Replay
    if recovery and recovery.success:
        events.append({
            "t": t_idx,
            "kind": "fix_attempt",
            "title": f"Sandbox replay with minimal patch at Step {recovery.forked_step}",
            "body": f"Applied patch {json.dumps(recovery.applied_patch)}. Replayed downstream stages in isolation.",
            "status": "passed_replay",
            "nodes": {
                "intake": "root" if faulty_tag == "intake" else "ok",
                "policy": "ok",
                "payroll": "ok",
            },
            "agent": faulty_tag,
            "data": {
                "patch": recovery.applied_patch,
                "replayed_output": recovery.final_recovered_state,
            },
        })
        t_idx += 1

    # Event N+3: Resolution / Verification
    if verified and verified.verification_passed:
        final_ded = float(recovery.final_recovered_state.get("salary_deduction", expected_deduction))
        final_bal = int(recovery.final_recovered_state.get("remaining_balance", expected_balance))
        events.append({
            "t": t_idx,
            "kind": "resolved",
            "title": "System recovered & verified",
            "body": f"Patch validated by independent verifier: {verified.verification_notes}. Final deduction is {final_ded:,.2f} SAR, remaining balance is {final_bal} days.",
            "status": "success",
            "nodes": {"intake": "ok", "policy": "ok", "payroll": "ok"},
            "agent": faulty_tag,
            "data": {
                "result": {"deduction": f"{final_ded:.2f}", "balance": final_bal},
                "corrupted": {"deduction": f"{corrupted_deduction:.2f}", "balance": corrupted_balance},
                "expected": {"deduction": f"{expected_deduction:.2f}", "balance": expected_balance},
                "notes": verified.verification_notes,
            },
        })
    else:
        # Distinguish individual agent status from overall incident outcome (Never mark all agents escalated)
        suspected_step = report.faulty_step if report else -1
        esc_nodes = {
            "intake": "root" if suspected_step == 0 else "ok",
            "policy": "root" if suspected_step == 1 else "ok",
            "payroll": "root" if suspected_step == 2 else "ok",
        }
        title_text = "Escalated to human supervisor"
        if report and "suspected" in str(report.faulty_agent).lower():
            title_text = str(report.faulty_agent)

        events.append({
            "t": t_idx,
            "kind": "escalated",
            "title": title_text,
            "body": verified.verification_notes if verified else (report.evidence if report else "Unrecoverable failure."),
            "status": "warning",
            "nodes": esc_nodes,
            "agent": "human",
            "data": {
                "reason": "unverified_recovery",
                "suspected": report.faulty_agent if report else "Unknown",
                "actual": {"deduction": f"{corrupted_deduction:.2f}", "balance": corrupted_balance},
                "expected": {"deduction": f"{expected_deduction:.2f}", "balance": expected_balance},
            },
        })

    return events
