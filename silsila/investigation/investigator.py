"""Autonomous Agentic Failure Investigator with Bounded Tool-Calling Loop.

Powered by local Ollama LLM (qwen2.5:3b) and deterministic diagnostic tools.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
from silsila.models import (
    ExecutionTrace,
    SilentFailureAlert,
    AttributionReport,
    InvestigationStep,
)
from silsila.llm_client import OllamaClient
from silsila.investigation.tools import DiagnosticTools

logger = logging.getLogger(__name__)

INVESTIGATOR_AGENT_SYSTEM_PROMPT = """You are an Autonomous AI Systems Investigator diagnosing a silent failure in a multi-agent workflow.

Workflow Architecture:
- Step 0 (IntakeAgent): Responsible for extracting the leave details from the Arabic user query into JSON.
- Step 1 (PolicyAgent): Applies organizational leave policy quotas and consecutive leave limits.
- Step 2 (PayrollAgent): Calculates downstream salary deductions based on unpaid days from the policy decision.

Available Diagnostic Tools:
1. "inspect_trace_step": {"step_index": int} -> Inspects input state, output state, and execution metadata of a step.
2. "compare_agent_handoffs": {"step_index": int} -> Checks if step received input matching output of previous step.
3. "audit_arabic_semantics": {"step_index": int} -> Audits raw Arabic query text against numerical and categorical extractions in step output.
4. "check_business_invariants": {"step_index": int} -> Audits quota, balance conservation, and payroll formulas for that step.
5. "test_counterfactual": {"step_index": int, "proposed_patch": dict} -> Safely simulates downstream re-execution in isolated sandbox to see if proposed patch clears all violations.
   - Step 0 allowlisted fields: ["days"].
   - Step 1 allowlisted fields: ["remaining_balance", "approved_paid_days", "unpaid_days"]. Note: For PolicyAgent balance deduction violation (prev - rem != approved), patch remaining_balance with (previous_balance - approved_paid_days).
   - Step 2 allowlisted fields: ["salary_deduction", "final_monthly_net"].
   - Note: No-op patches identical to current step output are rejected.
6. "finish": {"faulty_agent": str, "faulty_step": int, "error_category": str, "evidence": str, "causal_reasoning": str, "suggested_patch": dict} -> Terminate investigation with final attribution.
7. "escalate": {"reason": str} -> Escalate to human operator if evidence is ambiguous or contradictory.

Instructions:
- Formulate an initial hypothesis. Keep your "thought" concise (1-2 sentences).
- Autonomously select which diagnostic tool to invoke.
- Examine returned tool observations.
- You can test a fix with "test_counterfactual" or conclude directly with "finish" once confident.
- If the case is genuinely ambiguous and evidence is contradictory, choose "escalate".

You MUST respond strictly in valid JSON format:
{
  "thought": "Your current hypothesis and diagnostic reasoning based on evidence gathered so far",
  "action": "inspect_trace_step | compare_agent_handoffs | audit_arabic_semantics | check_business_invariants | test_counterfactual | finish | escalate",
  "action_input": {"step_index": 0}
}
Note for action 'finish': set action_input to {"faulty_agent": "IntakeAgent|PolicyAgent|PayrollAgent", "faulty_step": 0|1|2, "suggested_patch": {...}}"""


class AgenticInvestigator:
    """Genuine autonomous tool-calling AI agent for multi-agent root cause attribution."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        max_tool_calls: int = 5,
    ):
        self.client = client or OllamaClient()
        self.max_tool_calls = max_tool_calls

    def _call_client_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        format_json: bool = True,
        temperature: float = 0.0,
        timeout: int = 60,
        num_predict: int = 220,
    ) -> str:
        import inspect
        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "format_json": format_json,
            "temperature": temperature,
        }
        sig = inspect.signature(self.client.generate)
        if "timeout" in sig.parameters:
            kwargs["timeout"] = timeout
        if "num_predict" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            kwargs["num_predict"] = num_predict
        return self.client.generate(**kwargs)

    def investigate(
        self,
        trace: ExecutionTrace,
        alert: SilentFailureAlert,
    ) -> AttributionReport:
        """
        Executes an autonomous ReAct diagnostic loop:
        Failure Alert -> LLM Hypothesis -> Tool Call -> Observation -> Updated Hypothesis -> Final Attribution

        Pre-seeding: Before the LLM loop begins, run two deterministic tool calls to establish
        grounding context so the LLM does not waste iterations or hallucinate blame:
        1. audit_arabic_semantics(step_index=0) — always run; tells LLM if intake is clean or faulty.
        2. check_business_invariants on the step hinted by the alert invariant name.
        """
        start_time = time.perf_counter()
        timeline: List[InvestigationStep] = []
        query = trace.input_query
        step_count = len(trace.steps)
        llm_invocations_succeeded = 0
        llm_reached_but_invalid = False  # True if LLM responded with unparseable JSON

        # Context history provided to the LLM across tool-calling iterations
        diagnostic_history: List[Dict[str, Any]] = []

        # --- Pre-seeding: run deterministic grounding tools before LLM loop ---
        semantic_obs = DiagnosticTools.audit_arabic_semantics(trace, 0)
        diagnostic_history.append({
            "action": "audit_arabic_semantics",
            "action_input": {"step_index": 0},
            "observation": semantic_obs,
        })
        timeline.append(InvestigationStep(
            step_number=0,
            thought="Pre-seeding: auditing Arabic semantic quantity invariant at intake step.",
            tool_selected="audit_arabic_semantics",
            tool_arguments={"step_index": 0},
            observation=semantic_obs,
            updated_hypothesis="Grounding intake semantics before LLM hypothesis formulation.",
        ))

        # If intake looks clean, also pre-seed the invariant check on the layer the alert names
        if not semantic_obs.get("has_quantity_discrepancy", False):
            hint_step = self._alert_hint_step(alert)
            if hint_step is not None and hint_step != 0:
                inv_obs = DiagnosticTools.check_business_invariants(trace, hint_step)
                diagnostic_history.append({
                    "action": "check_business_invariants",
                    "action_input": {"step_index": hint_step},
                    "observation": inv_obs,
                })
                timeline.append(InvestigationStep(
                    step_number=0,
                    thought=f"Pre-seeding: checking business invariants at step {hint_step} (alert-hinted layer).",
                    tool_selected="check_business_invariants",
                    tool_arguments={"step_index": hint_step},
                    observation=inv_obs,
                    updated_hypothesis="Invariant violations at hinted layer recorded.",
                ))

        for call_idx in range(self.max_tool_calls):
            # Construct iterative prompt containing observations gathered so far
            history_text = ""
            if diagnostic_history:
                history_text = "\nPrior Diagnostic Actions & Observations:\n"
                for i, h in enumerate(diagnostic_history, 1):
                    history_text += (
                        f"Iteration {i}:\n"
                        f"  Tool: {h['action']}({json.dumps(h['action_input'])})\n"
                        f"  Observation: {json.dumps(h['observation'])}\n"
                    )

            prompt = f"""Silent Failure Alert:
- Violated Invariant: {alert.invariant_violated}
- Severity: {alert.severity}
- Description: {alert.description}

Trace Context:
- User Arabic Query: "{query}"
- Total Workflow Steps: {step_count} (Step 0: IntakeAgent, Step 1: PolicyAgent, Step 2: PayrollAgent)
{history_text}
Current Iteration: {call_idx + 1} of {self.max_tool_calls}.
Choose your next diagnostic tool or conclude with 'finish' or 'escalate'."""

            llm_response_raw = ""
            try:
                llm_response_raw = self._call_client_generate(
                    prompt=prompt,
                    system_prompt=INVESTIGATOR_AGENT_SYSTEM_PROMPT,
                    format_json=True,
                    temperature=0.0,
                    timeout=60,
                    num_predict=220,
                )
                decision = json.loads(llm_response_raw)
                llm_invocations_succeeded += 1
            except json.JSONDecodeError as e:
                # LLM was reachable but returned malformed / unparseable JSON
                logger.warning(f"Investigator LLM returned invalid JSON at iteration {call_idx}: {e!r}. Raw: {llm_response_raw[:200]!r}")
                llm_reached_but_invalid = True
                decision = {
                    "thought": f"LLM returned unparseable JSON: {e}. Falling back to rule-based trace audit.",
                    "action": "audit_arabic_semantics",
                    "action_input": {"step_index": 0},
                }
            except Exception as e:
                logger.warning(f"Investigator LLM connection error at iteration {call_idx}: {e}")
                decision = {
                    "thought": f"LLM connection error: {e}. Falling back to trace step audit.",
                    "action": "audit_arabic_semantics",
                    "action_input": {"step_index": 0},
                }

            thought = decision.get("thought", "Evaluating trace evidence.")
            action = decision.get("action", "inspect_trace_step")
            action_input = decision.get("action_input", {})
            if not isinstance(action_input, dict):
                action_input = {"step_index": 0}

            # Handle Escalation
            if action == "escalate":
                reason = action_input.get("reason", thought)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Check if evidence supports a suspected agent
                suspected_agent = "Unknown"
                suspected_step = -1
                for h in diagnostic_history:
                    obs = h.get("observation", {})
                    if obs.get("step_index") == 1 and obs.get("violations"):
                        suspected_agent = "PolicyAgent"
                        suspected_step = 1
                        break
                    elif obs.get("step_index") == 2 and obs.get("violations"):
                        suspected_agent = "PayrollAgent"
                        suspected_step = 2
                        break
                    elif obs.get("has_quantity_discrepancy"):
                        suspected_agent = "IntakeAgent"
                        suspected_step = 0
                        break

                display_agent = (
                    f"Root cause unresolved — suspected {suspected_agent} (Step {suspected_step})"
                    if suspected_step >= 0
                    else "Unknown"
                )

                timeline.append(
                    InvestigationStep(
                        step_number=call_idx + 1,
                        thought=thought,
                        tool_selected="escalate",
                        tool_arguments=action_input,
                        observation={"verdict": "ESCALATED", "reason": reason, "suspected": display_agent},
                        updated_hypothesis="Evidence is ambiguous or repair is unsupported. Escalating to human auditor.",
                    )
                )
                return AttributionReport(
                    faulty_agent=display_agent,
                    faulty_step=suspected_step,
                    error_category="Ambiguous / Unresolved",
                    evidence=reason,
                    causal_reasoning=thought,
                    confidence=0.3,
                    suggested_patch={},
                    investigation_timeline=timeline,
                    investigation_mode="escalated",
                    investigation_latency_ms=elapsed_ms,
                    escalated=True,
                    escalation_reason=reason,
                )

            # Handle Finish
            if action == "finish":
                faulty_agent, faulty_step, is_escalated = self._resolve_finish_attribution(
                    action_input, thought, diagnostic_history, alert
                )

                if is_escalated:
                    reason = action_input.get("reason", thought or "Insufficient diagnostic evidence to attribute root cause.")
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    timeline.append(
                        InvestigationStep(
                            step_number=call_idx + 1,
                            thought=thought,
                            tool_selected="escalate",
                            tool_arguments=action_input,
                            observation={"verdict": "ESCALATED", "reason": reason},
                            updated_hypothesis="Evidence is ambiguous or insufficient. Escalating to human auditor.",
                        )
                    )
                    return AttributionReport(
                        faulty_agent="Unknown",
                        faulty_step=-1,
                        error_category="Ambiguous / Unresolved",
                        evidence=reason,
                        causal_reasoning=thought,
                        confidence=0.3,
                        suggested_patch={},
                        investigation_timeline=timeline,
                        investigation_mode="escalated",
                        investigation_latency_ms=elapsed_ms,
                        escalated=True,
                        escalation_reason=reason,
                    )

                error_category = action_input.get("error_category", "Identified by Agentic Investigation")
                evidence = action_input.get("evidence", thought)
                causal_reasoning = action_input.get("causal_reasoning", thought)
                raw_patch = action_input.get("suggested_patch", {})
                llm_candidate_patch = dict(raw_patch) if raw_patch else None

                sanitized_patch, modified_fields = self._validate_and_sanitize_patch(
                    trace, faulty_agent, faulty_step, raw_patch, diagnostic_history
                )

                # Determine patch source
                if llm_invocations_succeeded > 0 and raw_patch:
                    patch_source = "llm_proposed"
                    rule_candidate_patch = None
                elif sanitized_patch:
                    patch_source = "rule_derived"
                    rule_candidate_patch = dict(sanitized_patch)
                    llm_candidate_patch = None
                else:
                    patch_source = "none"
                    rule_candidate_patch = None

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                timeline.append(
                    InvestigationStep(
                        step_number=call_idx + 1,
                        thought=thought,
                        tool_selected="finish",
                        tool_arguments=action_input,
                        observation={"status": "DIAGNOSIS_CONCLUDED"},
                        updated_hypothesis=causal_reasoning,
                    )
                )
                if llm_invocations_succeeded > 0:
                    active_mode = "agentic_llm"
                elif llm_reached_but_invalid:
                    active_mode = "llm_invalid_output_fallback"
                else:
                    active_mode = "llm_unavailable_fallback"
                return AttributionReport(
                    faulty_agent=faulty_agent,
                    faulty_step=faulty_step,
                    error_category=error_category,
                    evidence=evidence,
                    causal_reasoning=causal_reasoning,
                    confidence=0.95 if llm_invocations_succeeded > 0 else 0.80,
                    suggested_patch=sanitized_patch,
                    investigation_timeline=timeline,
                    investigation_mode=active_mode,
                    investigation_latency_ms=elapsed_ms,
                    escalated=False,
                    patch_source=patch_source,
                    llm_candidate_patch=llm_candidate_patch,
                    rule_candidate_patch=rule_candidate_patch,
                    sanitizer_modified_fields=modified_fields,
                )

            # Execute Selected Diagnostic Tool
            observation = self._execute_tool(trace, action, action_input)

            step_record = InvestigationStep(
                step_number=call_idx + 1,
                thought=thought,
                tool_selected=action,
                tool_arguments=action_input,
                observation=observation,
                updated_hypothesis=f"Tool {action} returned observation. Updating diagnostic state.",
            )
            timeline.append(step_record)

            diagnostic_history.append({
                "action": action,
                "action_input": action_input,
                "observation": observation,
            })

        # If loop reached max iterations without explicit finish, synthesize verdict from collected observations
        return self._synthesize_final_verdict(trace, alert, timeline, diagnostic_history, start_time, llm_invocations_succeeded, llm_reached_but_invalid)

    def _alert_hint_step(self, alert: SilentFailureAlert) -> Optional[int]:
        """Returns the step index most likely to be faulty based on the alert invariant name."""
        inv = alert.invariant_violated.lower()
        if "payroll" in inv or "deduction" in inv or "net" in inv or "salary" in inv:
            return 2
        if "policy" in inv or "balance" in inv or "conservation" in inv:
            return 1
        if "semantic" in inv or "quantity" in inv or "intake" in inv:
            return 0
        return None

    def _derive_patch_from_history(
        self,
        trace: ExecutionTrace,
        faulty_agent: str,
        faulty_step: int,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Derives a suggested patch from tool observations and trace data when the LLM omits one."""
        from silsila.llm_client import extract_arabic_number_from_text

        # --- Intake fault: correct days from Arabic query ---
        if faulty_agent == "IntakeAgent" or faulty_step == 0:
            # First check observations
            for h in history:
                obs = h.get("observation", {})
                exp = obs.get("expected_quantity_from_text")
                if exp is not None:
                    return {"days": int(exp)}
            # Fallback: derive directly from the original query
            expected = extract_arabic_number_from_text(trace.input_query)
            if expected is not None:
                return {"days": int(expected)}

        # --- Policy fault: recompute correct allocation from step 1 inputs ---
        if faulty_agent == "PolicyAgent" or faulty_step == 1:
            step_1 = trace.get_step(1)
            if step_1:
                req = int(step_1.input_state.get("days", 0))
                prev_bal = int(step_1.output_state.get("previous_balance", 21))
                from silsila.config import MAX_CONSECUTIVE_LEAVE_WITHOUT_VP_APPROVAL
                max_consec = MAX_CONSECUTIVE_LEAVE_WITHOUT_VP_APPROVAL
                correct_approved = min(req, max_consec, prev_bal)
                correct_unpaid = max(0, req - correct_approved)
                correct_rem = prev_bal - correct_approved
                return {
                    "approved_paid_days": correct_approved,
                    "unpaid_days": correct_unpaid,
                    "remaining_balance": correct_rem,
                }

        # --- Payroll fault: recompute from step 2 inputs ---
        if faulty_agent == "PayrollAgent" or faulty_step == 2:
            step_2 = trace.get_step(2)
            if step_2:
                daily_rate = float(step_2.output_state.get("daily_rate", 500.0))
                base_salary = float(step_2.output_state.get("base_monthly_salary", 15000.0))
                unpaid = int(step_2.input_state.get("unpaid_days", step_2.output_state.get("unpaid_days", 0)))
                correct_deduction = unpaid * daily_rate
                correct_net = base_salary - correct_deduction
                return {
                    "salary_deduction": correct_deduction,
                    "final_monthly_net": correct_net,
                }

        return {}

    def _execute_tool(
        self,
        trace: ExecutionTrace,
        action: str,
        action_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dispatches autonomous tool invocation to the deterministic diagnostic suite."""
        step_idx = action_input.get("step_index", 0)

        if action == "inspect_trace_step":
            return DiagnosticTools.inspect_trace_step(trace, step_idx)
        elif action == "compare_agent_handoffs":
            return DiagnosticTools.compare_agent_handoffs(trace, step_idx)
        elif action == "audit_arabic_semantics":
            return DiagnosticTools.audit_arabic_semantics(trace, step_idx)
        elif action == "check_business_invariants":
            return DiagnosticTools.check_business_invariants(trace, step_idx)
        elif action == "test_counterfactual":
            patch = action_input.get("proposed_patch", {})
            return DiagnosticTools.test_counterfactual(trace, step_idx, patch)
        else:
            return {"error": f"Unknown diagnostic tool: {action}. Please select a valid tool from registry."}

    def _resolve_finish_attribution(
        self,
        action_input: Dict[str, Any],
        thought: str,
        diagnostic_history: List[Dict[str, Any]],
        alert: SilentFailureAlert,
    ) -> tuple:
        """Resolves (faulty_agent, faulty_step, is_escalated) from action_input, thought, and evidence.
        Never defaults to IntakeAgent without supporting evidence.
        """
        agent_by_step = {0: "IntakeAgent", 1: "PolicyAgent", 2: "PayrollAgent"}
        step_by_agent = {"IntakeAgent": 0, "PolicyAgent": 1, "PayrollAgent": 2}

        # 1. Extract candidate step index from action_input
        step_val = action_input.get("faulty_step")
        if step_val is None:
            step_val = action_input.get("step_index")
        if step_val is None:
            step_val = action_input.get("step")

        step_int = None
        if step_val is not None:
            try:
                cand = int(step_val)
                if cand in agent_by_step:
                    step_int = cand
            except (ValueError, TypeError):
                step_int = None

        # 2. Extract candidate agent from action_input
        agent_val = action_input.get("faulty_agent") or action_input.get("agent")
        if agent_val and isinstance(agent_val, str):
            agent_clean = agent_val.strip()
            for k in step_by_agent:
                if agent_clean.lower() == k.lower():
                    agent_val = k
                    break

        # 3. Reconcile step and agent strictly
        if agent_val in step_by_agent and step_int is None:
            step_int = step_by_agent[agent_val]
        elif step_int in agent_by_step and agent_val is None:
            agent_val = agent_by_step[step_int]
        elif step_int in agent_by_step and agent_val in step_by_agent:
            # If agent and step match, we have a consistent attribution candidate
            if step_by_agent[agent_val] == step_int:
                return agent_val, step_int, False

            # Discrepancy between agent and step (e.g. PayrollAgent, 0): resolve based on diagnostic evidence
            intake_corrupted = any(
                h.get("observation", {}).get("has_quantity_discrepancy") is True or
                h.get("observation", {}).get("has_leave_type_discrepancy") is True
                for h in diagnostic_history
            )
            step1_corrupted = any(
                h.get("observation", {}).get("step_index") == 1 and bool(h.get("observation", {}).get("violations"))
                for h in diagnostic_history
            )
            step2_corrupted = any(
                h.get("observation", {}).get("step_index") == 2 and bool(h.get("observation", {}).get("violations"))
                for h in diagnostic_history
            )

            # Reconcile to the evidence-supported agent and its exact step
            if intake_corrupted:
                return "IntakeAgent", 0, False
            elif step1_corrupted:
                return "PolicyAgent", 1, False
            elif step2_corrupted:
                return "PayrollAgent", 2, False
            else:
                # Contradictory without clear evidence -> escalate
                return "Unknown", -1, True

        if step_int is not None and agent_val in step_by_agent and step_by_agent[agent_val] == step_int:
            return agent_val, step_int, False

        # 4. Check thought text for explicit agent diagnosis
        thought_lower = (thought or "").lower()
        if "payrollagent" in thought_lower or "payroll agent" in thought_lower:
            return "PayrollAgent", 2, False
        elif "policyagent" in thought_lower or "policy agent" in thought_lower:
            return "PolicyAgent", 1, False
        elif "intakeagent" in thought_lower or "intake agent" in thought_lower:
            return "IntakeAgent", 0, False

        # 5. Check tool observations for earliest supported causal violation
        for h in diagnostic_history:
            obs = h.get("observation", {})
            if obs.get("has_quantity_discrepancy") is True or obs.get("has_leave_type_discrepancy") is True:
                return "IntakeAgent", 0, False

        for h in diagnostic_history:
            obs = h.get("observation", {})
            if obs.get("step_index") == 1 and obs.get("violations"):
                return "PolicyAgent", 1, False

        for h in diagnostic_history:
            obs = h.get("observation", {})
            if obs.get("step_index") == 2 and obs.get("violations"):
                return "PayrollAgent", 2, False

        # 6. Fall back to alert invariant hint if unambiguous
        inv = alert.invariant_violated.lower()
        if "payroll" in inv or "deduction" in inv or "net" in inv or "salary" in inv:
            return "PayrollAgent", 2, False
        elif "policy" in inv or "balance" in inv or "conservation" in inv:
            return "PolicyAgent", 1, False
        elif "semantic" in inv or "quantity" in inv or "intake" in inv or "leave type" in inv:
            return "IntakeAgent", 0, False

        # 7. Unsupported or ambiguous evidence -> escalate
        return "Unknown", -1, True

    def _validate_and_sanitize_patch(
        self,
        trace: ExecutionTrace,
        faulty_agent: str,
        faulty_step: int,
        suggested_patch: Dict[str, Any],
        diagnostic_history: List[Dict[str, Any]],
    ) -> tuple:
        """Validates and completes proposed patch against business invariants before recovery.

        Returns:
            (sanitized_patch: Dict, modified_fields: List[str])

        modified_fields lists every field that was added, filled in, or changed by
        the sanitizer — including completions for Policy step 1.  This ensures that
        no LLM patch is silently altered without an audit trail.
        """
        if not isinstance(suggested_patch, dict):
            suggested_patch = {}
        patch = dict(suggested_patch)
        modified_fields: List[str] = []

        def _record_if_changed(key: str, old_val, new_val) -> None:
            """Records key to modified_fields if value changed or was added."""
            if old_val is None or old_val != new_val:
                modified_fields.append(key)

        if faulty_step == 0 or faulty_agent == "IntakeAgent":
            from silsila.llm_client import extract_arabic_number_from_text
            exp_days = extract_arabic_number_from_text(trace.input_query)
            days = patch.get("days")
            if days is None or not isinstance(days, int) or days <= 0 or (exp_days is not None and days != exp_days):
                new_days = exp_days if exp_days is not None else 5
                _record_if_changed("days", days, new_days)
                patch["days"] = new_days

        elif faulty_step == 1 or faulty_agent == "PolicyAgent":
            derived = self._derive_patch_from_history(trace, "PolicyAgent", 1, diagnostic_history)
            for k, v in derived.items():
                if k not in patch:
                    _record_if_changed(k, None, v)
                    patch[k] = v

            step_1 = trace.get_step(1)
            req_days = int(step_1.input_state.get("days", 0)) if step_1 else 0
            prev_bal = int(step_1.output_state.get("previous_balance", 21)) if step_1 else 21

            appr = int(patch.get("approved_paid_days", derived.get("approved_paid_days", 0)))
            if appr > req_days:
                new_appr = min(req_days, 14, prev_bal)
                _record_if_changed("approved_paid_days", appr, new_appr)
                appr = new_appr
                patch["approved_paid_days"] = appr

            # Enforce quota conservation: approved + unpaid == requested_days
            unpaid = int(patch.get("unpaid_days", derived.get("unpaid_days", 0)))
            if appr + unpaid != req_days:
                new_unpaid = max(0, req_days - appr)
                _record_if_changed("unpaid_days", unpaid, new_unpaid)
                patch["unpaid_days"] = new_unpaid

            # Enforce balance deduction invariant: prev_balance - remaining_balance == approved_paid_days
            rem = int(patch.get("remaining_balance", derived.get("remaining_balance", prev_bal - appr)))
            if (prev_bal - rem) != appr:
                new_rem = prev_bal - appr
                _record_if_changed("remaining_balance", rem, new_rem)
                patch["remaining_balance"] = new_rem

        elif faulty_step == 2 or faulty_agent == "PayrollAgent":
            derived = self._derive_patch_from_history(trace, "PayrollAgent", 2, diagnostic_history)
            for k, v in derived.items():
                if k not in patch:
                    _record_if_changed(k, None, v)
                    patch[k] = v

            step_2 = trace.get_step(2)
            base_sal = float(step_2.output_state.get("base_monthly_salary", 15000.0)) if step_2 else 15000.0
            daily_r = float(step_2.output_state.get("daily_rate", 500.0)) if step_2 else 500.0
            unpaid_d = int(step_2.input_state.get("unpaid_days", step_2.output_state.get("unpaid_days", 0))) if step_2 else 0

            expected_ded = unpaid_d * daily_r
            old_ded = patch.get("salary_deduction")
            ded = float(patch.get("salary_deduction", expected_ded))
            _record_if_changed("salary_deduction", old_ded, ded)
            patch["salary_deduction"] = ded

            old_net = patch.get("final_monthly_net")
            new_net = base_sal - ded
            _record_if_changed("final_monthly_net", old_net, new_net)
            patch["final_monthly_net"] = new_net

            old_adj = patch.get("net_payout_adjustment")
            new_adj = -ded
            _record_if_changed("net_payout_adjustment", old_adj, new_adj)
            patch["net_payout_adjustment"] = new_adj

        # Deduplicate modified_fields while preserving insertion order
        seen = set()
        unique_modified = [f for f in modified_fields if not (f in seen or seen.add(f))]

        return patch, unique_modified

    def _synthesize_final_verdict(
        self,
        trace: ExecutionTrace,
        alert: SilentFailureAlert,
        timeline: List[InvestigationStep],
        history: List[Dict[str, Any]],
        start_time: float,
        llm_invocations_succeeded: int = 0,
        llm_reached_but_invalid: bool = False,
    ) -> AttributionReport:
        """Synthesizes final attribution when max tool calls are reached."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        summary_prompt = f"""You have completed your diagnostic tool iterations.
Collected Observations:
{json.dumps(history, indent=2)}

Original Failure Alert:
{alert.description}

Invariant Violated: {alert.invariant_violated}

Based ONLY on the observations above, formulate your final root cause verdict in strict JSON.
Choose faulty_agent from: IntakeAgent (step 0), PolicyAgent (step 1), PayrollAgent (step 2).
The suggested_patch must contain the corrected field values derived from the observations.
{{
  "faulty_agent": "IntakeAgent | PolicyAgent | PayrollAgent",
  "faulty_step": 0,
  "error_category": "string",
  "evidence": "string explaining exact observations that prove the fault",
  "causal_reasoning": "string explaining how this caused downstream failure",
  "suggested_patch": {{}}
}}"""

        data = {}
        llm_success = False
        synth_invalid = False

        try:
            resp = self._call_client_generate(
                prompt=summary_prompt,
                system_prompt=INVESTIGATOR_AGENT_SYSTEM_PROMPT,
                format_json=True,
                temperature=0.0,
                timeout=60,
                num_predict=220,
            )
            data = json.loads(resp)
            llm_success = True
        except json.JSONDecodeError as e:
            logger.warning(f"Synthesize LLM returned invalid JSON: {e}. Deriving from observations.")
            synth_invalid = True
        except Exception as e:
            logger.warning(f"Synthesize LLM call failed: {e}. Deriving from observations.")

        thought = data.get("causal_reasoning", alert.description)
        llm_agent, llm_step, is_escalated = self._resolve_finish_attribution(
            data, thought, history, alert
        )

        if is_escalated:
            suspected_agent = "Unknown"
            suspected_step = -1
            for h in history:
                obs = h.get("observation", {})
                if obs.get("step_index") == 1 and obs.get("violations"):
                    suspected_agent = "PolicyAgent"
                    suspected_step = 1
                    break
                elif obs.get("step_index") == 2 and obs.get("violations"):
                    suspected_agent = "PayrollAgent"
                    suspected_step = 2
                    break
                elif obs.get("has_quantity_discrepancy"):
                    suspected_agent = "IntakeAgent"
                    suspected_step = 0
                    break

            display_agent = (
                f"Root cause unresolved — suspected {suspected_agent} (Step {suspected_step})"
                if suspected_step >= 0
                else "Unknown"
            )
            return AttributionReport(
                faulty_agent=display_agent,
                faulty_step=suspected_step,
                error_category="Ambiguous / Unresolved",
                evidence="Insufficient diagnostic evidence to attribute root cause.",
                causal_reasoning=thought,
                confidence=0.3,
                suggested_patch={},
                investigation_timeline=timeline,
                investigation_mode="escalated",
                investigation_latency_ms=elapsed_ms,
                escalated=True,
                escalation_reason="Evidence is ambiguous or insufficient.",
                patch_source="none",
            )

        raw_patch = data.get("suggested_patch", {})
        llm_candidate_patch = dict(raw_patch) if raw_patch else None
        sanitized_patch, modified_fields = self._validate_and_sanitize_patch(
            trace, llm_agent, llm_step, raw_patch, history
        )

        if llm_success and raw_patch:
            patch_source = "llm_proposed"
            rule_candidate_patch = None
        elif sanitized_patch:
            patch_source = "rule_derived"
            rule_candidate_patch = dict(sanitized_patch)
            llm_candidate_patch = None
        else:
            patch_source = "none"
            rule_candidate_patch = None

        if llm_success or llm_invocations_succeeded > 0:
            active_mode = "agentic_llm"
        elif llm_reached_but_invalid or synth_invalid:
            active_mode = "llm_invalid_output_fallback"
        else:
            active_mode = "llm_unavailable_fallback"

        return AttributionReport(
            faulty_agent=llm_agent,
            faulty_step=llm_step,
            error_category=data.get("error_category", "Agentic Causal Diagnosis" if llm_success else "Fallback Causal Synthesis"),
            evidence=data.get("evidence", alert.description),
            causal_reasoning=thought,
            confidence=0.90 if llm_success else 0.80,
            suggested_patch=sanitized_patch,
            investigation_timeline=timeline,
            investigation_mode=active_mode,
            investigation_latency_ms=elapsed_ms,
            escalated=False,
            patch_source=patch_source,
            llm_candidate_patch=llm_candidate_patch,
            rule_candidate_patch=rule_candidate_patch,
            sanitizer_modified_fields=modified_fields,
        )

class RuleBasedInvestigator:
    """Deterministic rule-based baseline comparator for the Evaluation Lab."""

    def investigate(self, trace: ExecutionTrace, alert: SilentFailureAlert) -> AttributionReport:
        start_time = time.perf_counter()
        step_0 = trace.get_step(0)
        step_1 = trace.get_step(1)
        step_2 = trace.get_step(2)

        from silsila.llm_client import extract_arabic_number_from_text, extract_leave_type_from_text, normalize_arabic_numerals

        query = trace.input_query
        expected_days = extract_arabic_number_from_text(query)
        expected_type = extract_leave_type_from_text(query)

        if step_0:
            emitted_days = step_0.output_state.get("days")
            emitted_type = step_0.output_state.get("leave_type")

            has_qty_err = expected_days is not None and emitted_days is not None and int(emitted_days) != int(expected_days)
            has_type_err = expected_type is not None and emitted_type is not None and emitted_type.lower() != expected_type.lower()

            if has_qty_err or has_type_err:
                patch = {}
                ev_list = []
                if has_qty_err:
                    patch["days"] = int(expected_days)
                    ev_list.append(f"days emitted {emitted_days} != expected {expected_days}")
                if has_type_err:
                    patch["leave_type"] = expected_type
                    ev_list.append(f"leave_type emitted '{emitted_type}' != expected '{expected_type}'")

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return AttributionReport(
                    faulty_agent="IntakeAgent",
                    faulty_step=0,
                    error_category="Rule-Based Invariant Check",
                    evidence=f"Intake semantic discrepancy: {'; '.join(ev_list)}.",
                    causal_reasoning="Procedural rule matched intake quantity or leave type divergence.",
                    confidence=1.0,
                    suggested_patch=patch,
                    investigation_timeline=[],
                    investigation_mode="rule_based",
                    investigation_latency_ms=elapsed_ms,
                    patch_source="rule_derived",
                    rule_candidate_patch=dict(patch),
                )

        if step_1:
            req_in = int(step_1.input_state.get("days", 0))
            approved = int(step_1.output_state.get("approved_paid_days", 0))
            unpaid = int(step_1.output_state.get("unpaid_days", 0))
            prev_bal = int(step_1.output_state.get("previous_balance", 21))
            rem_bal = int(step_1.output_state.get("remaining_balance", 21))

            if (approved + unpaid != req_in) or ((prev_bal - rem_bal) != approved):
                correct_approved = min(req_in, 14, prev_bal)
                correct_unpaid = max(0, req_in - correct_approved)
                correct_rem = prev_bal - correct_approved
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                policy_patch = {
                    "approved_paid_days": correct_approved,
                    "unpaid_days": correct_unpaid,
                    "remaining_balance": correct_rem,
                }
                return AttributionReport(
                    faulty_agent="PolicyAgent",
                    faulty_step=1,
                    error_category="Policy Balance Rule",
                    evidence="Procedural check found policy balance mismatch.",
                    causal_reasoning="Conservation rule violated in step 1.",
                    confidence=1.0,
                    suggested_patch=policy_patch,
                    investigation_timeline=[],
                    investigation_mode="rule_based",
                    investigation_latency_ms=elapsed_ms,
                    patch_source="rule_derived",
                    rule_candidate_patch=dict(policy_patch),
                )

        # Compute expected payroll values from step 2 inputs
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        payroll_patch: Dict[str, Any] = {}
        if step_2:
            daily_rate = float(step_2.output_state.get("daily_rate", 500.0))
            base_salary = float(step_2.output_state.get("base_monthly_salary", 15000.0))
            unpaid = int(step_2.input_state.get("unpaid_days", step_2.output_state.get("unpaid_days", 0)))
            expected_deduction = unpaid * daily_rate
            expected_net = base_salary - expected_deduction
            payroll_patch = {
                "salary_deduction": expected_deduction,
                "final_monthly_net": expected_net,
            }
        return AttributionReport(
            faulty_agent="PayrollAgent",
            faulty_step=2,
            error_category="Payroll Arithmetic Rule",
            evidence="Step 2 output salary_deduction/final_monthly_net violates payroll formula.",
            causal_reasoning="Payroll arithmetic discrepancy detected by rule-based invariant check.",
            confidence=1.0,
            suggested_patch=payroll_patch,
            investigation_timeline=[],
            investigation_mode="rule_based",
            investigation_latency_ms=elapsed_ms,
            patch_source="rule_derived",
            rule_candidate_patch=dict(payroll_patch) if payroll_patch else None,
        )


class RootCauseInvestigator:
    """Unified investigator interface supporting both Agentic LLM and Rule-based baseline modes."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        mode: str = "agentic",  # "agentic" or "rule_based"
        max_tool_calls: int = 5,
    ):
        self.mode = mode
        self.agentic_investigator = AgenticInvestigator(client=client, max_tool_calls=max_tool_calls)
        self.rule_based_investigator = RuleBasedInvestigator()

    def investigate(
        self,
        trace: ExecutionTrace,
        alert: SilentFailureAlert,
    ) -> AttributionReport:
        if self.mode == "rule_based":
            return self.rule_based_investigator.investigate(trace, alert)
        return self.agentic_investigator.investigate(trace, alert)
