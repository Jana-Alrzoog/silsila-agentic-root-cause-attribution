"""Payroll Agent: Computes downstream payroll financial impact."""

from typing import Any, Dict, Optional
from silsila.agents.base import BaseAgent
from silsila.models import PayrollImpact, get_employee_profile
from silsila.config import DEFAULT_MONTHLY_SALARY, DEFAULT_DAILY_RATE


class PayrollAgent(BaseAgent):
    """Computes payroll adjustments based on leave policy decisions.

    The authoritative monthly_salary and daily_rate are sourced from
    EmployeeProfile (via get_employee_profile) when not explicitly overridden
    at construction time.  Overriding at construction is supported to allow
    custom-profile scenarios in tests and evaluation.
    """

    def __init__(
        self,
        name: str = "PayrollAgent",
        monthly_salary: Optional[float] = None,  # None → read from EmployeeProfile
        daily_rate: Optional[float] = None,       # None → read from EmployeeProfile
    ):
        super().__init__(name)
        self._override_salary = monthly_salary
        self._override_rate = daily_rate

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input state expects output from PolicyAgent:
        {\"approved_paid_days\": int, \"unpaid_days\": int, \"status\": str, ...}

        Output state produces PayrollImpact.to_dict().
        """
        approved_paid = int(state.get("approved_paid_days", 0))
        unpaid = int(state.get("unpaid_days", 0))
        employee_id = state.get("employee_id", "EMP-101")

        # --- Authoritative salary/rate from EmployeeProfile ---
        if self._override_salary is not None and self._override_rate is not None:
            monthly_salary = self._override_salary
            daily_rate = self._override_rate
        else:
            profile = get_employee_profile(employee_id)
            monthly_salary = self._override_salary if self._override_salary is not None else profile.monthly_salary
            daily_rate = self._override_rate if self._override_rate is not None else profile.effective_daily_rate()

        paid_allowance = approved_paid * daily_rate
        salary_deduction = unpaid * daily_rate
        final_net = monthly_salary - salary_deduction
        net_adjustment = -salary_deduction

        notes = (
            f"Base: {monthly_salary:.2f} SAR. "
            f"Paid days: {approved_paid}. "
            f"Unpaid days: {unpaid} (-{salary_deduction:.2f} SAR). "
            f"Net salary: {final_net:.2f} SAR."
        )

        impact = PayrollImpact(
            base_monthly_salary=monthly_salary,
            daily_rate=daily_rate,
            approved_paid_days=approved_paid,
            paid_leave_allowance=paid_allowance,
            unpaid_days=unpaid,
            salary_deduction=salary_deduction,
            net_payout_adjustment=net_adjustment,
            final_monthly_net=final_net,
            payroll_notes=notes,
        )
        return impact.to_dict()
