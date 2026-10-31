"""Data models and schemas for SILSILA."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LeaveType(str, Enum):
    ANNUAL = "annual"
    SICK = "sick"
    EMERGENCY = "emergency"
    UNPAID = "unpaid"


# ---------------------------------------------------------------------------
# EmployeeProfile — single authoritative source for balances, salary, and rate
# ---------------------------------------------------------------------------

class EmployeeProfile(BaseModel):
    """Canonical employee record used by PolicyAgent, PayrollAgent, RecoveryEngine, and RecoveryVerifier."""
    employee_id: str
    accrued_annual_leave: int = 21       # days
    monthly_salary: float = 15000.0      # SAR
    days_in_month: int = 30
    daily_rate: float = 500.0            # SAR/day  (derived if not overridden)

    def effective_daily_rate(self) -> float:
        """Returns explicit daily_rate if set, otherwise salary / days_in_month."""
        return self.daily_rate


# Authoritative employee directory used throughout the system.
# Any agent, recovery engine, or verifier must call get_employee_profile() —
# NOT read salary/balance from upstream trace outputs — for ground-truth values.
EMPLOYEE_DIRECTORY: Dict[str, EmployeeProfile] = {
    "EMP-101": EmployeeProfile(employee_id="EMP-101", accrued_annual_leave=21, monthly_salary=15000.0, days_in_month=30, daily_rate=500.0),
    "EMP-102": EmployeeProfile(employee_id="EMP-102", accrued_annual_leave=21, monthly_salary=15000.0, days_in_month=30, daily_rate=500.0),
    "EMP-103": EmployeeProfile(employee_id="EMP-103", accrued_annual_leave=21, monthly_salary=15000.0, days_in_month=30, daily_rate=500.0),
    "EMP-104": EmployeeProfile(employee_id="EMP-104", accrued_annual_leave=21, monthly_salary=15000.0, days_in_month=30, daily_rate=500.0),
    "EMP-105": EmployeeProfile(employee_id="EMP-105", accrued_annual_leave=21, monthly_salary=15000.0, days_in_month=30, daily_rate=500.0),
    "EMP-106": EmployeeProfile(employee_id="EMP-106", accrued_annual_leave=21, monthly_salary=15000.0, days_in_month=30, daily_rate=500.0),
    "EMP-107": EmployeeProfile(employee_id="EMP-107", accrued_annual_leave=21, monthly_salary=15000.0, days_in_month=30, daily_rate=500.0),
    "EMP-999": EmployeeProfile(employee_id="EMP-999", accrued_annual_leave=10, monthly_salary=24000.0, days_in_month=30, daily_rate=800.0),
}

_DEFAULT_PROFILE = EmployeeProfile(
    employee_id="DEFAULT",
    accrued_annual_leave=21,
    monthly_salary=15000.0,
    days_in_month=30,
    daily_rate=500.0,
)


def get_employee_profile(employee_id: str) -> EmployeeProfile:
    """Returns the authoritative EmployeeProfile for the given employee ID.

    Falls back to the system default profile for unknown IDs, so evaluation
    scenarios with ad-hoc IDs do not cause KeyErrors.
    """
    return EMPLOYEE_DIRECTORY.get(employee_id, _DEFAULT_PROFILE)


def register_employee_profile(profile: EmployeeProfile) -> None:
    """Registers (or replaces) an EmployeeProfile in the directory.
    Useful in tests and evaluation scenarios that need custom salary/balance."""
    EMPLOYEE_DIRECTORY[profile.employee_id] = profile


# ---------------------------------------------------------------------------
# Core workflow models
# ---------------------------------------------------------------------------

class LeaveRequest(BaseModel):
    """Normalized leave request produced by IntakeAgent."""
    employee_id: str = "EMP-101"
    leave_type: LeaveType = LeaveType.ANNUAL
    days: int = Field(gt=0, description="Number of leave days requested")
    notes: str = ""
    original_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "employee_id": self.employee_id,
            "leave_type": self.leave_type.value if isinstance(self.leave_type, LeaveType) else str(self.leave_type),
            "days": int(self.days),
            "notes": self.notes,
            "original_query": self.original_query,
        }


class PolicyDecision(BaseModel):
    """Decision output produced by PolicyAgent."""
    status: str = "APPROVED"  # APPROVED, PARTIALLY_APPROVED, REJECTED
    requested_days: int
    approved_paid_days: int
    unpaid_days: int = 0
    previous_balance: int = 21
    remaining_balance: int = 16
    policy_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "requested_days": self.requested_days,
            "approved_paid_days": self.approved_paid_days,
            "unpaid_days": self.unpaid_days,
            "previous_balance": self.previous_balance,
            "remaining_balance": self.remaining_balance,
            "policy_notes": self.policy_notes,
        }


class PayrollImpact(BaseModel):
    """Financial impact computed by PayrollAgent."""
    base_monthly_salary: float = 15000.0
    daily_rate: float = 500.0
    approved_paid_days: int = 0
    paid_leave_allowance: float = 0.0
    unpaid_days: int = 0
    salary_deduction: float = 0.0
    net_payout_adjustment: float = 0.0
    final_monthly_net: float = 15000.0
    payroll_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_monthly_salary": self.base_monthly_salary,
            "daily_rate": self.daily_rate,
            "approved_paid_days": self.approved_paid_days,
            "paid_leave_allowance": self.paid_leave_allowance,
            "unpaid_days": self.unpaid_days,
            "salary_deduction": self.salary_deduction,
            "net_payout_adjustment": self.net_payout_adjustment,
            "final_monthly_net": self.final_monthly_net,
            "payroll_notes": self.payroll_notes,
        }


class TraceStep(BaseModel):
    """Represents a single agent execution step in the workflow trace."""
    step_index: int
    agent_name: str
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = "SUCCESS"  # Silent failures may still report SUCCESS!
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    """Full execution trace across the multi-agent workflow."""
    workflow_id: str
    input_query: str
    steps: List[TraceStep] = Field(default_factory=list)
    final_output: Dict[str, Any] = Field(default_factory=dict)
    injected_fault: Optional[Dict[str, Any]] = None  # Hidden from Investigator!

    def get_step(self, step_index: int) -> Optional[TraceStep]:
        for step in self.steps:
            if step.step_index == step_index:
                return step
        return None

    def get_agent_step(self, agent_name: str) -> Optional[TraceStep]:
        for step in self.steps:
            if step.agent_name == agent_name:
                return step
        return None


class SilentFailureAlert(BaseModel):
    """Alert raised when an invariant or outcome mismatch is detected."""
    detected: bool
    severity: str = "CRITICAL"
    invariant_violated: str
    description: str
    divergent_fields: Dict[str, Any] = Field(default_factory=dict)


class InvestigationStep(BaseModel):
    """A single step in the agentic tool-calling investigation timeline."""
    step_number: int
    thought: str
    tool_selected: str
    tool_arguments: Dict[str, Any] = Field(default_factory=dict)
    observation: Dict[str, Any] = Field(default_factory=dict)
    updated_hypothesis: str = ""


class AttributionReport(BaseModel):
    """Result of evidence-based root cause attribution."""
    faulty_agent: str
    faulty_step: int
    error_category: str
    evidence: str
    causal_reasoning: str
    confidence: float
    suggested_patch: Dict[str, Any]
    investigation_timeline: List[InvestigationStep] = Field(default_factory=list)
    investigation_mode: str = "agentic_llm"
    # "agentic_llm"               — LLM participated and produced parseable output
    # "rule_based"                — deterministic rule-based investigator
    # "escalated"                 — investigation could not attribute root cause
    # "llm_unavailable_fallback"  — LLM endpoint unreachable or disabled
    # "llm_invalid_output_fallback" — LLM reachable but returned unparseable JSON
    investigation_latency_ms: float = 0.0
    escalated: bool = False
    escalation_reason: str = ""

    # --- Patch provenance fields ---
    patch_source: str = "none"
    # "llm_proposed"  — patch came directly from a valid LLM response
    # "rule_derived"  — patch was derived from deterministic business rules
    # "none"          — no patch (clean run, escalation, or investigation failed)
    llm_candidate_patch: Optional[Dict[str, Any]] = None
    rule_candidate_patch: Optional[Dict[str, Any]] = None

    # --- Sanitizer audit fields ---
    sanitizer_modified_fields: List[str] = Field(default_factory=list)
    # List of field names that _validate_and_sanitize_patch added, filled in,
    # or changed.  Empty when no sanitizer modification occurred.


class RecoveryResult(BaseModel):
    """Result of safe state isolation, replay, and recovery verification."""
    success: bool
    original_failure: str
    applied_patch: Dict[str, Any]
    forked_step: int
    replayed_steps: List[TraceStep] = Field(default_factory=list)
    final_recovered_state: Dict[str, Any] = Field(default_factory=dict)
    verification_passed: bool
    verification_notes: str
