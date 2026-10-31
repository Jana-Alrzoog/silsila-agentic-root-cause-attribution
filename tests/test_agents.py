"""Unit tests for individual SILSILA agents."""

from silsila.agents.intake import IntakeAgent
from silsila.agents.policy import PolicyAgent
from silsila.agents.payroll import PayrollAgent
from silsila.models import LeaveType


class TestIntakeAgent:
    def test_intake_arabic_digits(self):
        agent = IntakeAgent()
        result = agent.execute({"query": "أبي إجازة ٥ أيام", "employee_id": "EMP-101"})
        assert result["days"] == 5
        assert result["employee_id"] == "EMP-101"
        assert result["leave_type"] == LeaveType.ANNUAL.value

    def test_intake_ascii_digits(self):
        agent = IntakeAgent()
        result = agent.execute({"query": "طلب إجازة 3 أيام", "employee_id": "EMP-102"})
        assert result["days"] == 3
        assert result["employee_id"] == "EMP-102"

    def test_intake_sick_leave(self):
        agent = IntakeAgent()
        result = agent.execute({"query": "إجازة مرضية 4 أيام", "employee_id": "EMP-103"})
        assert result["days"] == 4
        assert result["leave_type"] == LeaveType.SICK.value


class TestPolicyAgent:
    def test_standard_approved_leave(self):
        agent = PolicyAgent(accrued_balance=21, max_consecutive=14)
        input_data = {"employee_id": "EMP-101", "leave_type": "annual", "days": 5}
        decision = agent.execute(input_data)

        assert decision["status"] == "APPROVED"
        assert decision["approved_paid_days"] == 5
        assert decision["unpaid_days"] == 0
        assert decision["remaining_balance"] == 16

    def test_consecutive_cap_enforcement(self):
        agent = PolicyAgent(accrued_balance=21, max_consecutive=14)
        input_data = {"employee_id": "EMP-101", "leave_type": "annual", "days": 15}
        decision = agent.execute(input_data)

        # 14 days paid, 1 day unpaid
        assert decision["status"] == "PARTIALLY_APPROVED"
        assert decision["approved_paid_days"] == 14
        assert decision["unpaid_days"] == 1
        assert decision["remaining_balance"] == 7

    def test_exceeds_total_balance(self):
        agent = PolicyAgent(accrued_balance=10, max_consecutive=14)
        input_data = {"employee_id": "EMP-101", "leave_type": "annual", "days": 12}
        decision = agent.execute(input_data)

        assert decision["status"] == "PARTIALLY_APPROVED"
        assert decision["approved_paid_days"] == 10
        assert decision["unpaid_days"] == 2
        assert decision["remaining_balance"] == 0


class TestPayrollAgent:
    def test_full_paid_leave_no_deduction(self):
        agent = PayrollAgent(monthly_salary=15000.0, daily_rate=500.0)
        policy_data = {"approved_paid_days": 5, "unpaid_days": 0, "status": "APPROVED"}
        payroll = agent.execute(policy_data)

        assert payroll["salary_deduction"] == 0.0
        assert payroll["final_monthly_net"] == 15000.0
        assert payroll["paid_leave_allowance"] == 2500.0

    def test_unpaid_leave_deduction(self):
        agent = PayrollAgent(monthly_salary=15000.0, daily_rate=500.0)
        policy_data = {"approved_paid_days": 14, "unpaid_days": 1, "status": "PARTIALLY_APPROVED"}
        payroll = agent.execute(policy_data)

        assert payroll["salary_deduction"] == 500.0
        assert payroll["final_monthly_net"] == 14500.0
