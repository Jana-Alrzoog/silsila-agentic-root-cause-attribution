"""Policy Agent: Applies explicit simulated leave policy rules."""

from typing import Any, Dict
from silsila.agents.base import BaseAgent
from silsila.models import PolicyDecision, get_employee_profile
from silsila.config import (
    MAX_CONSECUTIVE_LEAVE_WITHOUT_VP_APPROVAL,
)


class PolicyAgent(BaseAgent):
    """Evaluates leave requests against simulated organizational policy.

    The authoritative accrued leave balance is sourced from EmployeeProfile
    (via get_employee_profile) when accrued_balance is not explicitly overridden
    at construction time.  Overriding at construction is supported to allow
    custom-profile scenarios in tests and evaluation.
    """

    def __init__(
        self,
        name: str = "PolicyAgent",
        accrued_balance: int = None,  # None → read from EmployeeProfile
        max_consecutive: int = MAX_CONSECUTIVE_LEAVE_WITHOUT_VP_APPROVAL,
    ):
        super().__init__(name)
        self._override_balance = accrued_balance  # None means "use profile"
        self.max_consecutive = max_consecutive

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input state expects output from IntakeAgent:
        {\"employee_id\": str, \"leave_type\": str, \"days\": int, ...}

        Output state produces PolicyDecision.to_dict().
        """
        requested_days = int(state.get("days", 0))
        leave_type = str(state.get("leave_type", "annual")).lower()
        employee_id = state.get("employee_id", "EMP-101")

        # --- Authoritative balance from EmployeeProfile ---
        if self._override_balance is not None:
            current_balance = self._override_balance
        else:
            profile = get_employee_profile(employee_id)
            current_balance = profile.accrued_annual_leave

        # Allow state to supply accrued_balance override (backward compat)
        current_balance = int(state.get("accrued_balance", current_balance))

        if requested_days <= 0:
            return PolicyDecision(
                status="REJECTED",
                requested_days=0,
                approved_paid_days=0,
                unpaid_days=0,
                previous_balance=current_balance,
                remaining_balance=current_balance,
                policy_notes="Invalid request: zero or negative days.",
            ).to_dict()

        if leave_type == "unpaid":
            # All days are unpaid, balance remains untouched
            decision = PolicyDecision(
                status="APPROVED",
                requested_days=requested_days,
                approved_paid_days=0,
                unpaid_days=requested_days,
                previous_balance=current_balance,
                remaining_balance=current_balance,
                policy_notes="Approved as unpaid leave per employee request.",
            )
            return decision.to_dict()

        # Standard Annual / Emergency / Sick Leave Policy
        if requested_days <= current_balance and requested_days <= self.max_consecutive:
            # Full standard approval
            approved_paid = requested_days
            unpaid = 0
            remaining = current_balance - approved_paid
            status = "APPROVED"
            notes = f"Approved {approved_paid} days paid annual leave."
        elif requested_days <= current_balance and requested_days > self.max_consecutive:
            # Exceeds single consecutive limit (14 days)
            approved_paid = self.max_consecutive
            unpaid = requested_days - self.max_consecutive
            remaining = current_balance - approved_paid
            status = "PARTIALLY_APPROVED"
            notes = (
                f"Consecutive limit ({self.max_consecutive} days) reached. "
                f"{unpaid} days designated unpaid pending VP review."
            )
        else:
            # Exceeds accrued balance
            approved_paid = min(requested_days, current_balance)
            unpaid = requested_days - approved_paid
            remaining = current_balance - approved_paid
            status = "PARTIALLY_APPROVED" if approved_paid > 0 else "REJECTED"
            notes = (
                f"Requested {requested_days} days exceeds accrued balance ({current_balance}). "
                f"{approved_paid} days approved paid; {unpaid} days unpaid."
            )

        decision = PolicyDecision(
            status=status,
            requested_days=requested_days,
            approved_paid_days=approved_paid,
            unpaid_days=unpaid,
            previous_balance=current_balance,
            remaining_balance=remaining,
            policy_notes=notes,
        )
        return decision.to_dict()
