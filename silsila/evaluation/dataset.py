"""Phase 3 Benchmark Dataset — 60+ diverse evaluation cases.

Design principles:
- Fixed RNG seed (42) for reproducibility.
- Ground-truth labels stored here, separated from method inputs in runner.py.
- Dev/test split defined by fault family AND phrasing template BEFORE any tuning.
  Families used during regex / rule development are labeled 'dev'.
  Only families labeled 'test' should be treated as held-out.
- Varied: salary (12k, 15k, 20k, 24k), balance (10, 15, 21, 28),
  days (1–25), phrasings, fault agents, clean runs, ambiguous, wrong leave types,
  wrong-but-plausible patches.
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Benchmark scenario model
# ---------------------------------------------------------------------------

class BenchmarkCase(BaseModel):
    case_id: str
    name: str
    query: str
    employee_id: str

    # Employee context (registered in EMPLOYEE_DIRECTORY by runner)
    monthly_salary: float
    daily_rate: float
    accrued_balance: int

    # Fault injection
    fault_stage: str = "none"  # "none" | "intake" | "policy" | "payroll"
    fault_mutation: Dict[str, Any] = {}
    fault_description: str = ""

    # Ground truth — stored SEPARATELY from method inputs
    gt_expected_days: Optional[int] = None
    gt_faulty_agent: str = "None"
    gt_faulty_step: int = -1
    gt_expected_final: Dict[str, Any] = {}

    # Metadata
    split: str = "dev"    # "dev" | "test"
    fault_family: str = "clean"  # clean | intake_qty | intake_type | policy_balance | payroll_arith | ambiguous
    phrasing_template: str = "standard"


# ---------------------------------------------------------------------------
# Dev / test split declaration (by fault_family + phrasing_template)
# Families used during regex/rule DEVELOPMENT are labeled 'dev'.
# Families labeled 'test' are genuinely held-out.
# ---------------------------------------------------------------------------

# Dev families (used during Phase 1 & 2 development / regex design):
#   clean, intake_qty (basic Arabic-Indic), policy_balance (simple wipeout),
#   payroll_arith (triple-charge), intake_type (annual→unpaid)
# Test families (not used in rule/regex design):
#   intake_qty_word_number, intake_qty_week, payroll_arith_partial,
#   policy_balance_partial, ambiguous_request, wrong_leave_type

DEV_FAMILIES = {
    "clean",
    "intake_qty",          # basic Arabic-Indic numerals
    "intake_type",         # leave type corruption
    "policy_balance",      # balance wipeout
    "payroll_arith",       # triple-charge or full wipeout
}

TEST_FAMILIES = {
    "intake_qty_word",     # word-based numbers (ثلاثة, خمسة...)
    "intake_qty_week",     # week-based (أسبوع, أسبوعين)
    "payroll_arith_partial",  # partial over/undercharge
    "policy_balance_partial", # partial balance corruption
    "ambiguous_request",   # requests with no detectable quantity
    "wrong_leave_type",    # wrong leave type passed through
}


# ---------------------------------------------------------------------------
# Case construction helpers
# ---------------------------------------------------------------------------

def _clean(case_id, name, query, employee_id, salary, daily_rate, balance,
           expected_days, split="dev", family="clean", phrasing="standard"):
    """Construct a clean (no-fault) case."""
    approved = min(expected_days, 14, balance)
    unpaid = max(0, expected_days - approved)
    remaining = balance - approved
    deduction = unpaid * daily_rate
    net = salary - deduction

    return BenchmarkCase(
        case_id=case_id,
        name=name,
        query=query,
        employee_id=employee_id,
        monthly_salary=salary,
        daily_rate=daily_rate,
        accrued_balance=balance,
        fault_stage="none",
        fault_mutation={},
        fault_description="",
        gt_expected_days=expected_days,
        gt_faulty_agent="None",
        gt_faulty_step=-1,
        gt_expected_final={
            "approved_paid_days": approved,
            "unpaid_days": unpaid,
            "remaining_balance": remaining,
            "salary_deduction": deduction,
            "final_monthly_net": net,
        },
        split=split,
        fault_family=family,
        phrasing_template=phrasing,
    )


def _intake_qty(case_id, name, query, employee_id, salary, daily_rate, balance,
                true_days, injected_days, split="dev", family="intake_qty", phrasing="standard"):
    """Construct an intake quantity corruption case."""
    approved = min(true_days, 14, balance)
    unpaid = max(0, true_days - approved)
    remaining = balance - approved
    deduction = float(unpaid * daily_rate)
    net = float(salary - deduction)

    return BenchmarkCase(
        case_id=case_id,
        name=name,
        query=query,
        employee_id=employee_id,
        monthly_salary=salary,
        daily_rate=daily_rate,
        accrued_balance=balance,
        fault_stage="intake",
        fault_mutation={"days": injected_days},
        fault_description=f"Intake distorted {true_days} to {injected_days} days.",
        gt_expected_days=true_days,
        gt_faulty_agent="IntakeAgent",
        gt_faulty_step=0,
        gt_expected_final={
            "approved_paid_days": approved,
            "unpaid_days": unpaid,
            "remaining_balance": remaining,
            "salary_deduction": deduction,
            "final_monthly_net": net,
        },
        split=split,
        fault_family=family,
        phrasing_template=phrasing,
    )


def _policy_bal(case_id, name, query, employee_id, salary, daily_rate, balance,
                true_days, corrupted_remaining, split="dev", family="policy_balance"):
    """Construct a policy balance corruption case."""
    approved = min(true_days, 14, balance)
    unpaid = max(0, true_days - approved)
    correct_remaining = balance - approved

    return BenchmarkCase(
        case_id=case_id,
        name=name,
        query=query,
        employee_id=employee_id,
        monthly_salary=salary,
        daily_rate=daily_rate,
        accrued_balance=balance,
        fault_stage="policy",
        fault_mutation={"remaining_balance": corrupted_remaining},
        fault_description=f"Policy balance corrupted to {corrupted_remaining} instead of {correct_remaining}.",
        gt_expected_days=true_days,
        gt_faulty_agent="PolicyAgent",
        gt_faulty_step=1,
        gt_expected_final={
            "approved_paid_days": approved,
            "unpaid_days": unpaid,
            "remaining_balance": correct_remaining,
            "salary_deduction": 0.0,
            "final_monthly_net": salary,
        },
        split=split,
        fault_family=family,
    )


def _payroll_arith(case_id, name, query, employee_id, salary, daily_rate, balance,
                   true_days, injected_deduction, injected_net, unpaid_days=0,
                   split="dev", family="payroll_arith"):
    """Construct a payroll arithmetic corruption case."""
    approved = min(true_days, 14, balance) - unpaid_days
    correct_deduction = float(unpaid_days * daily_rate)
    correct_net = float(salary - correct_deduction)

    return BenchmarkCase(
        case_id=case_id,
        name=name,
        query=query,
        employee_id=employee_id,
        monthly_salary=salary,
        daily_rate=daily_rate,
        accrued_balance=balance,
        fault_stage="payroll",
        fault_mutation={"salary_deduction": injected_deduction, "final_monthly_net": injected_net},
        fault_description=f"Payroll deducted {injected_deduction} SAR instead of {correct_deduction} SAR.",
        gt_expected_days=true_days,
        gt_faulty_agent="PayrollAgent",
        gt_faulty_step=2,
        gt_expected_final={
            "approved_paid_days": min(true_days - unpaid_days, 14, balance),
            "unpaid_days": unpaid_days,
            "salary_deduction": correct_deduction,
            "final_monthly_net": correct_net,
        },
        split=split,
        fault_family=family,
    )


# ---------------------------------------------------------------------------
# Build the benchmark: 60+ cases
# ---------------------------------------------------------------------------

def build_benchmark() -> List[BenchmarkCase]:
    rng = random.Random(42)  # Fixed seed for reproducibility

    cases: List[BenchmarkCase] = []

    # -----------------------------------------------------------------------
    # DEV SPLIT — Clean runs (10 cases)
    # -----------------------------------------------------------------------
    clean_configs = [
        ("DEV-C-01", "أبي إجازة ٣ أيام", "EMP-D01", 15000, 500.0, 21, 3, "arabic_indic"),
        ("DEV-C-02", "طلب إجازة سنوية لمدة 3 أيام تبدأ الأحد", "EMP-D02", 15000, 500.0, 21, 3, "standard"),
        ("DEV-C-03", "أريد إجازة مرضية لمدة يومين", "EMP-D03", 15000, 500.0, 21, 2, "sick_dual"),
        ("DEV-C-04", "إجازة طارئة لمدة أسبوع", "EMP-D04", 15000, 500.0, 21, 7, "emergency_week"),
        ("DEV-C-05", "أحتاج إجازة اعتيادية لمدة 5 أيام", "EMP-D05", 20000, 666.67, 28, 5, "standard_high_sal"),
        ("DEV-C-06", "أريد إجازة بدون راتب لمدة يومين", "EMP-D06", 12000, 400.0, 10, 2, "unpaid_low_bal"),
        ("DEV-C-07", "طلب إجازة سنوية لمدة 10 أيام", "EMP-D07", 24000, 800.0, 21, 10, "ten_days"),
        ("DEV-C-08", "أريد إجازة لمدة 14 يوم", "EMP-D08", 15000, 500.0, 21, 14, "max_consecutive"),
        ("DEV-C-09", "أحتاج إجازة لمدة يوم واحد", "EMP-D09", 15000, 500.0, 15, 1, "one_day_low_bal"),
        ("DEV-C-10", "طلب إجازة لمدة أسبوعين", "EMP-D10", 15000, 500.0, 21, 14, "two_weeks"),
    ]
    for c in clean_configs:
        cid, query, emp, sal, rate, bal, exp_days, phrasing = c
        # Register employee profile
        cases.append(_clean(
            case_id=cid,
            name=f"Clean: {query[:40]}",
            query=query,
            employee_id=emp,
            salary=sal,
            daily_rate=rate,
            balance=bal,
            expected_days=exp_days,
            split="dev",
            family="clean",
            phrasing=phrasing,
        ))

    # -----------------------------------------------------------------------
    # DEV SPLIT — Intake quantity corruptions (10 cases, Arabic-Indic primary)
    # -----------------------------------------------------------------------
    intake_qty_dev = [
        ("DEV-IQ-01", "أبي إجازة ٥ أيام", "EMP-D11", 15000, 500.0, 21, 5, 15),
        ("DEV-IQ-02", "أحتاج إجازة ٤ أيام لظروف عائلية", "EMP-D12", 15000, 500.0, 21, 4, 10),
        ("DEV-IQ-03", "أريد إجازة لمدة 3 أيام", "EMP-D13", 12000, 400.0, 10, 3, 8),
        ("DEV-IQ-04", "إجازة لمدة 6 أيام", "EMP-D14", 20000, 666.67, 28, 6, 16),
        ("DEV-IQ-05", "أبغى إجازة ٣ أيام", "EMP-D15", 15000, 500.0, 15, 3, 9),
        ("DEV-IQ-06", "أريد إجازة لمدة 7 أيام", "EMP-D16", 24000, 800.0, 21, 7, 17),
        ("DEV-IQ-07", "إجازة لمدة 2 يوم", "EMP-D17", 15000, 500.0, 21, 2, 12),
        ("DEV-IQ-08", "أحتاج إجازة ٢ أيام", "EMP-D18", 12000, 400.0, 15, 2, 6),
        ("DEV-IQ-09", "أريد إجازة لمدة ١٠ أيام", "EMP-D19", 15000, 500.0, 21, 10, 20),
        ("DEV-IQ-10", "طلب إجازة لمدة 1 يوم", "EMP-D20", 20000, 666.67, 28, 1, 5),
    ]
    for c in intake_qty_dev:
        cid, query, emp, sal, rate, bal, true_d, inj_d = c
        cases.append(_intake_qty(
            case_id=cid,
            name=f"DEV Intake Qty: {query[:35]}",
            query=query,
            employee_id=emp,
            salary=sal,
            daily_rate=rate,
            balance=bal,
            true_days=true_d,
            injected_days=inj_d,
            split="dev",
            family="intake_qty",
        ))

    # -----------------------------------------------------------------------
    # DEV SPLIT — Intake type corruption (3 cases)
    # -----------------------------------------------------------------------
    dev_intake_type = [
        ("DEV-IT-01", "طلب إجازة اعتيادية لمدة 3 أيام", "EMP-D21", 15000, 500.0, 21, 3, "unpaid"),
        ("DEV-IT-02", "إجازة سنوية لمدة 5 أيام", "EMP-D22", 15000, 500.0, 21, 5, "unpaid"),
        ("DEV-IT-03", "أريد إجازة اعتيادية لمدة 2 يوم", "EMP-D23", 20000, 666.67, 28, 2, "unpaid"),
    ]
    for c in dev_intake_type:
        cid, query, emp, sal, rate, bal, true_d, bad_type = c
        approved = min(true_d, 14, bal)
        cases.append(BenchmarkCase(
            case_id=cid,
            name=f"DEV Intake Type: {query[:35]}",
            query=query,
            employee_id=emp,
            monthly_salary=sal,
            daily_rate=rate,
            accrued_balance=bal,
            fault_stage="intake",
            fault_mutation={"leave_type": bad_type},
            fault_description=f"Intake corrupted leave_type to '{bad_type}' instead of 'annual'.",
            gt_expected_days=true_d,
            gt_faulty_agent="IntakeAgent",
            gt_faulty_step=0,
            gt_expected_final={
                "approved_paid_days": approved,
                "unpaid_days": 0,
                "remaining_balance": bal - approved,
                "salary_deduction": 0.0,
                "final_monthly_net": float(sal),
            },
            split="dev",
            fault_family="intake_type",
        ))

    # -----------------------------------------------------------------------
    # DEV SPLIT — Policy balance corruptions (5 cases)
    # -----------------------------------------------------------------------
    dev_policy = [
        ("DEV-PO-01", "أرغب في إجازة اعتيادية لمدة 2 يوم", "EMP-D24", 15000, 500.0, 21, 2, 0),
        ("DEV-PO-02", "إجازة لمدة 5 أيام", "EMP-D25", 15000, 500.0, 21, 5, 100),
        ("DEV-PO-03", "أريد إجازة لمدة 3 أيام", "EMP-D26", 20000, 666.67, 28, 3, 0),
        ("DEV-PO-04", "طلب إجازة 4 أيام", "EMP-D27", 12000, 400.0, 10, 4, 0),
        ("DEV-PO-05", "أبغى إجازة 7 أيام", "EMP-D28", 15000, 500.0, 21, 7, 99),
    ]
    for c in dev_policy:
        cid, query, emp, sal, rate, bal, true_d, corrupt_rem = c
        cases.append(_policy_bal(cid, f"DEV Policy Bal: {query[:35]}", query, emp,
                                  sal, rate, bal, true_d, corrupt_rem, split="dev"))

    # -----------------------------------------------------------------------
    # DEV SPLIT — Payroll arithmetic corruptions (5 cases)
    # -----------------------------------------------------------------------
    dev_payroll = [
        # (cid, query, emp, sal, rate, bal, days, inj_ded, inj_net, unpaid_days)
        ("DEV-PA-01", "طلب إجازة بدون راتب لمدة 3 أيام", "EMP-D29", 15000, 500.0, 21, 3, 4500.0, 10500.0, 3),
        ("DEV-PA-02", "إجازة بدون راتب ليومين", "EMP-D30", 20000, 666.67, 21, 2, 5000.0, 15000.0, 2),
        ("DEV-PA-03", "أريد إجازة غير مدفوعة لمدة 5 أيام", "EMP-D31", 15000, 500.0, 21, 5, 9000.0, 6000.0, 5),
        ("DEV-PA-04", "طلب إجازة بدون راتب لمدة يومين", "EMP-D32", 24000, 800.0, 21, 2, 8000.0, 16000.0, 2),
        ("DEV-PA-05", "إجازة بدون راتب لمدة 4 أيام", "EMP-D33", 12000, 400.0, 10, 4, 6400.0, 5600.0, 4),
    ]
    for c in dev_payroll:
        cid, query, emp, sal, rate, bal, days, inj_d, inj_n, unpaid = c
        cases.append(_payroll_arith(cid, f"DEV Payroll Arith: {query[:35]}", query, emp,
                                     sal, rate, bal, days, inj_d, inj_n, unpaid_days=unpaid, split="dev"))

    # -----------------------------------------------------------------------
    # TEST SPLIT — Word-number intake corruptions (5 cases, family=intake_qty_word)
    # -----------------------------------------------------------------------
    test_word_intake = [
        ("TEST-IW-01", "إجازة لمدة ثلاثة أيام", "EMP-T01", 15000, 500.0, 21, 3, 13),
        ("TEST-IW-02", "أريد إجازة لمدة خمسة أيام", "EMP-T02", 15000, 500.0, 21, 5, 15),
        ("TEST-IW-03", "طلب إجازة سبعة أيام", "EMP-T03", 20000, 666.67, 28, 7, 17),
        ("TEST-IW-04", "أحتاج إجازة لمدة عشرة أيام", "EMP-T04", 15000, 500.0, 21, 10, 20),
        ("TEST-IW-05", "إجازة أربعة أيام لظروف عائلية", "EMP-T05", 12000, 400.0, 15, 4, 14),
    ]
    for c in test_word_intake:
        cid, query, emp, sal, rate, bal, true_d, inj_d = c
        cases.append(_intake_qty(
            case_id=cid,
            name=f"TEST Intake Word-Num: {query[:35]}",
            query=query,
            employee_id=emp,
            salary=sal,
            daily_rate=rate,
            balance=bal,
            true_days=true_d,
            injected_days=inj_d,
            split="test",
            family="intake_qty_word",
            phrasing="word_number",
        ))

    # -----------------------------------------------------------------------
    # TEST SPLIT — Week-based intake corruptions (3 cases, family=intake_qty_week)
    # -----------------------------------------------------------------------
    test_week_intake = [
        ("TEST-IK-01", "إجازة طارئة لمدة أسبوع", "EMP-T06", 15000, 500.0, 21, 7, 17),
        ("TEST-IK-02", "طلب إجازة لمدة أسبوعين", "EMP-T07", 15000, 500.0, 21, 14, 24),
        ("TEST-IK-03", "أريد إجازة لمدة أسبوع ونصف", "EMP-T08", 20000, 666.67, 28, 7, 12),  # ambiguous "week and half" abstains -> extractor gets 7
    ]
    for c in test_week_intake:
        cid, query, emp, sal, rate, bal, true_d, inj_d = c
        cases.append(_intake_qty(
            case_id=cid,
            name=f"TEST Intake Week: {query[:35]}",
            query=query,
            employee_id=emp,
            salary=sal,
            daily_rate=rate,
            balance=bal,
            true_days=true_d,
            injected_days=inj_d,
            split="test",
            family="intake_qty_week",
            phrasing="week_expr",
        ))

    # -----------------------------------------------------------------------
    # TEST SPLIT — Partial payroll corruption (5 cases, family=payroll_arith_partial)
    # -----------------------------------------------------------------------
    test_pay_partial = [
        # Under-deduction (charged less than owed)
        ("TEST-PP-01", "إجازة غير مدفوعة لمدة 5 أيام", "EMP-T09", 15000, 500.0, 21, 5, 500.0, 14500.0, 5),
        # Over-deduction by 2x
        ("TEST-PP-02", "طلب إجازة بدون راتب لمدة 2 يوم", "EMP-T10", 15000, 500.0, 21, 2, 2000.0, 13000.0, 2),
        # Off by one day's pay
        ("TEST-PP-03", "إجازة بدون راتب لمدة 3 أيام", "EMP-T11", 20000, 666.67, 21, 3, 2666.68, 17333.32, 3),
        # Wrong sign (negative deduction = effectively a bonus)
        ("TEST-PP-04", "إجازة بدون راتب ليوم واحد", "EMP-T12", 12000, 400.0, 10, 1, -400.0, 12400.0, 1),
        # Double-charge on 4-day unpaid
        ("TEST-PP-05", "إجازة بدون راتب لمدة 4 أيام", "EMP-T13", 24000, 800.0, 21, 4, 6400.0, 17600.0, 4),
    ]
    for c in test_pay_partial:
        cid, query, emp, sal, rate, bal, days, inj_d, inj_n, unpaid = c
        cases.append(_payroll_arith(cid, f"TEST Payroll Partial: {query[:35]}", query, emp,
                                     sal, rate, bal, days, inj_d, inj_n, unpaid_days=unpaid,
                                     split="test", family="payroll_arith_partial"))

    # -----------------------------------------------------------------------
    # TEST SPLIT — Partial policy balance corruptions (5 cases)
    # -----------------------------------------------------------------------
    test_pol_partial = [
        ("TEST-PB-01", "إجازة لمدة 3 أيام", "EMP-T14", 15000, 500.0, 21, 3, 21),    # no deduction (rem stays at max)
        ("TEST-PB-02", "أريد إجازة 5 أيام", "EMP-T15", 20000, 666.67, 28, 5, 10),   # partial off
        ("TEST-PB-03", "طلب إجازة 4 أيام", "EMP-T16", 15000, 500.0, 15, 4, 15),     # balance unchanged (should be 11)
        ("TEST-PB-04", "إجازة لمدة 7 أيام", "EMP-T17", 12000, 400.0, 10, 7, 0),     # balance wiped
        ("TEST-PB-05", "أريد إجازة 2 يوم", "EMP-T18", 15000, 500.0, 21, 2, 30),     # balance inflated
    ]
    for c in test_pol_partial:
        cid, query, emp, sal, rate, bal, true_d, corrupt_rem = c
        cases.append(_policy_bal(cid, f"TEST Policy Partial: {query[:35]}", query, emp,
                                  sal, rate, bal, true_d, corrupt_rem,
                                  split="test", family="policy_balance_partial"))

    # -----------------------------------------------------------------------
    # TEST SPLIT — Ambiguous requests (4 cases, family=ambiguous_request)
    # Extractor returns None -> no fault can be detected by semantic invariant
    # -----------------------------------------------------------------------
    ambiguous_cases = [
        ("TEST-AM-01", "إجازة لفترة وجيزة", "EMP-T19", 15000, 500.0, 21, None),
        ("TEST-AM-02", "أريد إجازة قصيرة", "EMP-T20", 15000, 500.0, 21, None),
        ("TEST-AM-03", "طلب إجازة متقطعة", "EMP-T21", 20000, 666.67, 28, None),
        ("TEST-AM-04", "إجازة بشكل عاجل", "EMP-T22", 12000, 400.0, 10, None),
    ]
    for cid, query, emp, sal, rate, bal, _ in ambiguous_cases:
        cases.append(BenchmarkCase(
            case_id=cid,
            name=f"TEST Ambiguous: {query[:35]}",
            query=query,
            employee_id=emp,
            monthly_salary=sal,
            daily_rate=rate,
            accrued_balance=bal,
            fault_stage="none",
            fault_mutation={},
            fault_description="",
            gt_expected_days=None,
            gt_faulty_agent="None",
            gt_faulty_step=-1,
            gt_expected_final={},
            split="test",
            fault_family="ambiguous_request",
            phrasing_template="ambiguous",
        ))

    # -----------------------------------------------------------------------
    # TEST SPLIT — Wrong leave type (wrong-but-plausible patch) (3 cases)
    # -----------------------------------------------------------------------
    wrong_type_cases = [
        ("TEST-WT-01", "طلب إجازة مرضية لمدة 3 أيام", "EMP-T23", 15000, 500.0, 21, 3, "annual"),
        ("TEST-WT-02", "أريد إجازة طارئة لمدة 2 يوم", "EMP-T24", 15000, 500.0, 21, 2, "unpaid"),
        ("TEST-WT-03", "إجازة اعتيادية لمدة 5 أيام", "EMP-T25", 20000, 666.67, 28, 5, "sick"),
    ]
    for c in wrong_type_cases:
        cid, query, emp, sal, rate, bal, true_d, bad_type = c
        approved = min(true_d, 14, bal)
        cases.append(BenchmarkCase(
            case_id=cid,
            name=f"TEST Wrong Type: {query[:35]}",
            query=query,
            employee_id=emp,
            monthly_salary=sal,
            daily_rate=rate,
            accrued_balance=bal,
            fault_stage="intake",
            fault_mutation={"leave_type": bad_type},
            fault_description=f"Intake corrupted leave_type from correct to '{bad_type}'.",
            gt_expected_days=true_d,
            gt_faulty_agent="IntakeAgent",
            gt_faulty_step=0,
            gt_expected_final={
                "approved_paid_days": approved,
                "unpaid_days": 0,
                "remaining_balance": bal - approved,
                "salary_deduction": 0.0,
                "final_monthly_net": float(sal),
            },
            split="test",
            fault_family="wrong_leave_type",
            phrasing_template="standard",
        ))

    # -----------------------------------------------------------------------
    # Additional clean runs for TEST split (5 more)
    # -----------------------------------------------------------------------
    extra_clean_test = [
        ("TEST-C-01", "أريد إجازة لمدة 6 أيام", "EMP-T26", 20000, 666.67, 28, 6, "test"),
        ("TEST-C-02", "طلب إجازة لمدة 12 يوم", "EMP-T27", 24000, 800.0, 21, 12, "test"),
        ("TEST-C-03", "إجازة مرضية لمدة يومين", "EMP-T28", 12000, 400.0, 15, 2, "test"),
        ("TEST-C-04", "أبغى إجازة 9 أيام", "EMP-T29", 15000, 500.0, 21, 9, "test"),
        ("TEST-C-05", "إجازة سنوية ليوم واحد", "EMP-T30", 15000, 500.0, 10, 1, "test"),
    ]
    for c in extra_clean_test:
        cid, query, emp, sal, rate, bal, exp_days, split = c
        cases.append(_clean(cid, f"TEST Clean: {query[:35]}", query, emp,
                            sal, rate, bal, exp_days, split=split, family="clean"))

    return cases


PHASE3_BENCHMARK: List[BenchmarkCase] = build_benchmark()

# Backward compatibility aliases
BENCHMARK_SCENARIOS = PHASE3_BENCHMARK
EvalScenario = BenchmarkCase


def get_dev_cases() -> List[BenchmarkCase]:
    return [c for c in PHASE3_BENCHMARK if c.split == "dev"]


def get_test_cases() -> List[BenchmarkCase]:
    return [c for c in PHASE3_BENCHMARK if c.split == "test"]

