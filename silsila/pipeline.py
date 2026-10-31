"""Workflow Pipeline: Orchestrates agents with execution tracing and fault injection."""

import copy
import uuid
from typing import Any, Dict, Optional
from silsila.agents.intake import IntakeAgent
from silsila.agents.policy import PolicyAgent
from silsila.agents.payroll import PayrollAgent
from silsila.models import ExecutionTrace, TraceStep


class FaultInjector:
    """Configures and applies simulated silent faults at specified workflow stages."""

    def __init__(
        self,
        target_stage: Optional[str] = None,  # "intake", "policy", "payroll"
        mutation: Optional[Dict[str, Any]] = None,
        fault_description: str = "",
    ):
        self.target_stage = target_stage.lower() if target_stage else None
        self.mutation = mutation or {}
        self.fault_description = fault_description

    def should_inject(self, stage_name: str) -> bool:
        if not self.target_stage:
            return False
        return self.target_stage in stage_name.lower()

    def apply(self, stage_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mutation silently to state."""
        if not self.should_inject(stage_name):
            return state

        mutated = copy.deepcopy(state)
        for key, val in self.mutation.items():
            mutated[key] = val
        return mutated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_stage": self.target_stage,
            "mutation": self.mutation,
            "fault_description": self.fault_description,
        }


class WorkflowPipeline:
    """End-to-end multi-agent workflow runner with full execution tracing."""

    def __init__(
        self,
        intake_agent: Optional[IntakeAgent] = None,
        policy_agent: Optional[PolicyAgent] = None,
        payroll_agent: Optional[PayrollAgent] = None,
    ):
        self.intake_agent = intake_agent or IntakeAgent()
        self.policy_agent = policy_agent or PolicyAgent()
        self.payroll_agent = payroll_agent or PayrollAgent()

    def run(
        self,
        query: str,
        employee_id: str = "EMP-101",
        fault_injector: Optional[FaultInjector] = None,
    ) -> ExecutionTrace:
        """
        Executes the full pipeline:
        Step 0: IntakeAgent (Arabic parsing)
        Step 1: PolicyAgent (simulated policy evaluation)
        Step 2: PayrollAgent (downstream compensation calculation)
        """
        workflow_id = f"wf-{uuid.uuid4().hex[:8]}"
        trace = ExecutionTrace(
            workflow_id=workflow_id,
            input_query=query,
            injected_fault=fault_injector.to_dict() if fault_injector and fault_injector.target_stage else None,
        )

        # ----------------------------------------------------
        # Step 0: Intake Agent
        # ----------------------------------------------------
        intake_input = {"query": query, "employee_id": employee_id}
        intake_output = self.intake_agent.execute(intake_input)

        if fault_injector and fault_injector.should_inject("intake"):
            intake_output = fault_injector.apply("intake", intake_output)

        step_0 = TraceStep(
            step_index=0,
            agent_name=self.intake_agent.name,
            input_state=copy.deepcopy(intake_input),
            output_state=copy.deepcopy(intake_output),
            status="SUCCESS",  # Silent failure reports SUCCESS
            metadata={"stage": "intake"},
        )
        trace.steps.append(step_0)

        # ----------------------------------------------------
        # Step 1: Policy Agent
        # ----------------------------------------------------
        policy_input = copy.deepcopy(intake_output)
        policy_output = self.policy_agent.execute(policy_input)

        if fault_injector and fault_injector.should_inject("policy"):
            policy_output = fault_injector.apply("policy", policy_output)

        step_1 = TraceStep(
            step_index=1,
            agent_name=self.policy_agent.name,
            input_state=copy.deepcopy(policy_input),
            output_state=copy.deepcopy(policy_output),
            status="SUCCESS",
            metadata={"stage": "policy"},
        )
        trace.steps.append(step_1)

        # ----------------------------------------------------
        # Step 2: Payroll Agent
        # ----------------------------------------------------
        payroll_input = copy.deepcopy(policy_output)
        payroll_output = self.payroll_agent.execute(payroll_input)

        if fault_injector and fault_injector.should_inject("payroll"):
            payroll_output = fault_injector.apply("payroll", payroll_output)

        step_2 = TraceStep(
            step_index=2,
            agent_name=self.payroll_agent.name,
            input_state=copy.deepcopy(payroll_input),
            output_state=copy.deepcopy(payroll_output),
            status="SUCCESS",
            metadata={"stage": "payroll"},
        )
        trace.steps.append(step_2)

        # Final output of the entire workflow is the payroll & policy summary
        trace.final_output = {
            "requested_days": intake_output.get("days"),
            "leave_type": intake_output.get("leave_type"),
            "policy_status": policy_output.get("status"),
            "approved_paid_days": policy_output.get("approved_paid_days"),
            "unpaid_days": policy_output.get("unpaid_days"),
            "remaining_balance": policy_output.get("remaining_balance"),
            "salary_deduction": payroll_output.get("salary_deduction"),
            "final_monthly_net": payroll_output.get("final_monthly_net"),
        }

        return trace
