"""Unit tests for WorkflowPipeline and Execution Tracing."""

from silsila.pipeline import WorkflowPipeline, FaultInjector


def test_pipeline_execution_trace_structure():
    pipeline = WorkflowPipeline()
    trace = pipeline.run("طلب إجازة سنوية لمدة 3 أيام", employee_id="EMP-200")

    assert trace.workflow_id.startswith("wf-")
    assert trace.input_query == "طلب إجازة سنوية لمدة 3 أيام"
    assert len(trace.steps) == 3

    # Step 0: IntakeAgent
    step_0 = trace.get_step(0)
    assert step_0.agent_name == "IntakeAgent"
    assert step_0.status == "SUCCESS"
    assert step_0.output_state["days"] == 3
    assert step_0.output_state["employee_id"] == "EMP-200"

    # Step 1: PolicyAgent
    step_1 = trace.get_step(1)
    assert step_1.agent_name == "PolicyAgent"
    assert step_1.status == "SUCCESS"
    assert step_1.output_state["approved_paid_days"] == 3
    assert step_1.output_state["remaining_balance"] == 18

    # Step 2: PayrollAgent
    step_2 = trace.get_step(2)
    assert step_2.agent_name == "PayrollAgent"
    assert step_2.status == "SUCCESS"
    assert step_2.output_state["salary_deduction"] == 0.0
    assert step_2.output_state["final_monthly_net"] == 15000.0


def test_pipeline_fault_injection_intake():
    pipeline = WorkflowPipeline()
    injector = FaultInjector(
        target_stage="intake",
        mutation={"days": 20},
        fault_description="Inject 20 days instead of actual",
    )
    trace = pipeline.run("أبي إجازة 2 يوم", fault_injector=injector)

    # Step 0 should have mutated days=20
    assert trace.get_step(0).output_state["days"] == 20
    # Policy step sees days=20 -> max consecutive cap 14
    assert trace.get_step(1).output_state["approved_paid_days"] == 14
    assert trace.get_step(1).output_state["unpaid_days"] == 6


def test_pipeline_fault_injection_policy():
    pipeline = WorkflowPipeline()
    injector = FaultInjector(
        target_stage="policy",
        mutation={"remaining_balance": 0},
        fault_description="Zero out remaining balance",
    )
    trace = pipeline.run("أبي إجازة 3 أيام", fault_injector=injector)

    assert trace.get_step(1).output_state["remaining_balance"] == 0


def test_pipeline_fault_injection_payroll():
    pipeline = WorkflowPipeline()
    injector = FaultInjector(
        target_stage="payroll",
        mutation={"salary_deduction": 5000.0, "final_monthly_net": 10000.0},
        fault_description="Corrupt salary deduction",
    )
    trace = pipeline.run("أبي إجازة 3 أيام", fault_injector=injector)

    assert trace.get_step(2).output_state["salary_deduction"] == 5000.0
    assert trace.get_step(2).output_state["final_monthly_net"] == 10000.0
