"""Benchmark Evaluation Runner: Multi-method comparative evaluation for SILSILA.

Supports evaluation methods:
1. blame_last_agent: Naive heuristic baseline attributing fault to PayrollAgent (Step 2) with no patch.
2. rule_based: Deterministic invariant auditor using procedural rules.
3. llm_disabled_fallback: AgenticInvestigator ReAct harness with LLM disabled, exercising the deterministic fallback loop.
4. llm_live: Real AgenticInvestigator invoking local Ollama (qwen2.5:3b). Marked NOT RUN if unreachable.

Integrates with Phase 3 benchmark dataset, maintaining strict separation between
ground-truth labels and investigator inputs, and recording comprehensive metrics
with exact numerators and denominators.
"""

import os
import time
from typing import Any, Dict, List, Optional

from silsila.evaluation.dataset import (
    PHASE3_BENCHMARK,
    BENCHMARK_SCENARIOS,
    BenchmarkCase,
    EvalScenario,
    get_dev_cases,
    get_test_cases,
)
from silsila.models import (
    EmployeeProfile,
    register_employee_profile,
    AttributionReport,
)
from silsila.pipeline import WorkflowPipeline, FaultInjector
from silsila.investigation.detector import SilentFailureDetector
from silsila.investigation.investigator import (
    RuleBasedInvestigator,
    AgenticInvestigator,
)
from silsila.investigation.recovery import RecoveryEngine
from silsila.investigation.verifier import RecoveryVerifier
from silsila.llm_client import OllamaClient


class BenchmarkRunner:
    """Executes benchmark scenarios across configured evaluation methods."""

    VALID_METHODS = [
        "blame_last_agent",
        "rule_based",
        "llm_disabled_fallback",
        "llm_live",
    ]

    def __init__(self, method: str = "rule_based", max_tool_calls: int = 5):
        if method not in self.VALID_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {self.VALID_METHODS}")
        self.method = method
        self.max_tool_calls = max_tool_calls

        self.pipeline = WorkflowPipeline()
        self.detector = SilentFailureDetector()
        self.recovery_engine = RecoveryEngine()
        self.verifier = RecoveryVerifier()

        # Initialize investigator based on method
        if self.method == "rule_based":
            self.investigator = RuleBasedInvestigator()
        elif self.method == "llm_disabled_fallback":
            self.investigator = AgenticInvestigator(
                client=OllamaClient(),
                max_tool_calls=self.max_tool_calls,
            )
        elif self.method == "llm_live":
            self.investigator = AgenticInvestigator(
                client=OllamaClient(),
                max_tool_calls=self.max_tool_calls,
            )
        else:
            self.investigator = None  # blame_last_agent handled directly

    def _blame_last_agent(self, trace, alert) -> AttributionReport:
        """Naive baseline: always blames the last agent (PayrollAgent, Step 2) with no patch."""
        return AttributionReport(
            faulty_agent="PayrollAgent",
            faulty_step=2,
            error_category="Heuristic: Blame Last Agent",
            evidence="Heuristic baseline attributed failure to the terminal agent in the workflow DAG.",
            causal_reasoning="Naive baseline: downstream agent blamed regardless of upstream root cause.",
            confidence=0.33,
            suggested_patch={},
            investigation_timeline=[],
            investigation_mode="heuristic_baseline",
            patch_source="none",
        )

    def run_case(self, case: BenchmarkCase) -> Dict[str, Any]:
        """Runs an individual evaluation scenario with ground truth strictly isolated."""
        # 1. Authoritative employee profile registration (authoritative source)
        profile = EmployeeProfile(
            employee_id=case.employee_id,
            accrued_annual_leave=case.accrued_balance,
            monthly_salary=case.monthly_salary,
            days_in_month=30,
            daily_rate=case.daily_rate,
        )
        register_employee_profile(profile)

        # 2. Setup fault injector if configured
        fault_injector = None
        if case.fault_stage != "none":
            fault_injector = FaultInjector(
                target_stage=case.fault_stage,
                mutation=case.fault_mutation,
                fault_description=case.fault_description,
            )

        # 3. Execute multi-agent pipeline
        trace = self.pipeline.run(
            query=case.query,
            employee_id=case.employee_id,
            fault_injector=fault_injector,
        )

        # 4. Silent failure detection (independent invariants, NO ground-truth leakage)
        alert = self.detector.detect(trace)

        is_actual_fault = (case.fault_stage != "none")
        detection_correct = (alert.detected == is_actual_fault)

        attribution_correct = False
        recovery_verified = False
        false_recovery_claim = False
        report: Optional[AttributionReport] = None
        recovery = None

        if alert.detected:
            # 5. Investigate root cause
            if self.method == "blame_last_agent":
                report = self._blame_last_agent(trace, alert)
            elif self.method == "llm_disabled_fallback":
                orig_env = os.environ.get("SILSILA_DISABLE_LLM")
                try:
                    os.environ["SILSILA_DISABLE_LLM"] = "1"
                    report = self.investigator.investigate(trace, alert)
                finally:
                    if orig_env is None:
                        os.environ.pop("SILSILA_DISABLE_LLM", None)
                    else:
                        os.environ["SILSILA_DISABLE_LLM"] = orig_env
            else:
                report = self.investigator.investigate(trace, alert)

            # Check attribution against ground truth
            gt_agent = case.gt_faulty_agent.lower()
            gt_step = case.gt_faulty_step
            if report.faulty_agent.lower() == gt_agent and report.faulty_step == gt_step:
                attribution_correct = True

            # 6. Safe sandbox recovery & replay
            recovery = self.recovery_engine.recover(trace, report)

            # 7. Independent verification without ground-truth leakage
            verified_recovery = self.verifier.verify(
                original_trace=trace,
                recovery_result=recovery,
            )
            recovery_verified = verified_recovery.verification_passed

            # 8. Check for false recovery claims:
            # If verifier passed recovery, verify against case.gt_expected_final
            if recovery_verified and case.gt_expected_final:
                mismatches = []
                final_st = recovery.final_recovered_state
                for k, expected_val in case.gt_expected_final.items():
                    actual_val = final_st.get(k)
                    if actual_val is None:
                        mismatches.append(f"{k} missing in recovered state")
                    elif isinstance(expected_val, float):
                        if abs(float(actual_val) - expected_val) > 0.05:
                            mismatches.append(f"{k}: got {actual_val}, expected {expected_val}")
                    elif actual_val != expected_val:
                        mismatches.append(f"{k}: got {actual_val}, expected {expected_val}")

                if mismatches:
                    false_recovery_claim = True
                    recovery_verified = False  # Revoke invalid verification
        else:
            # No failure detected
            if not is_actual_fault:
                attribution_correct = True
                recovery_verified = True
            else:
                # Silent failure missed (False Negative)
                attribution_correct = False
                recovery_verified = False

        return {
            "case_id": case.case_id,
            "name": case.name,
            "split": case.split,
            "fault_family": case.fault_family,
            "phrasing_template": case.phrasing_template,
            "is_actual_fault": is_actual_fault,
            "fault_stage": case.fault_stage,
            "detected": alert.detected,
            "detection_correct": detection_correct,
            "invariant_violated": alert.invariant_violated if alert.detected else "None",
            "attribution_correct": attribution_correct,
            "attributed_agent": report.faulty_agent if report else "None",
            "attributed_step": report.faulty_step if report else -1,
            "expected_agent": case.gt_faulty_agent,
            "expected_step": case.gt_faulty_step,
            "patch_source": report.patch_source if report else "none",
            "sanitizer_modified_fields": report.sanitizer_modified_fields if report else [],
            "investigation_mode": report.investigation_mode if report else "none",
            "recovery_verified": recovery_verified,
            "false_recovery_claim": false_recovery_claim,
            "verification_notes": (
                recovery.verification_notes if recovery else "No failure occurred."
            ),
        }

    def run_all(self, cases: List[BenchmarkCase] = PHASE3_BENCHMARK) -> Dict[str, Any]:
        """Runs all benchmark cases and aggregates detailed metrics with denominators."""
        # Check live LLM reachability if method is llm_live
        if self.method == "llm_live":
            client = OllamaClient()
            if not client.is_available():
                return {
                    "method": self.method,
                    "status": "NOT RUN",
                    "reason": "Ollama server unreachable in this execution environment (WinError 10013 / offline sandbox).",
                    "instructions": "Execute locally in an unsandboxed terminal with Ollama running: python run_eval.py --methods llm_live --out results.json",
                    "total_cases": len(cases),
                }

        results = []
        total = len(cases)
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        correct_attributions = 0
        successful_recoveries = 0
        faulty_count = 0
        clean_count = 0

        # Split tracking
        dev_total = 0
        dev_correct_attrib = 0
        test_total = 0
        test_correct_attrib = 0

        # Provenance tracking
        patch_sources = {"llm_proposed": 0, "rule_derived": 0, "none": 0}
        sanitizer_modifications_count = 0
        fallback_count = 0
        false_recovery_claims_count = 0

        for case in cases:
            res = self.run_case(case)
            results.append(res)

            is_fault = res["is_actual_fault"]
            detected = res["detected"]

            if case.split == "dev":
                dev_total += 1
                if res["attribution_correct"]:
                    dev_correct_attrib += 1
            else:
                test_total += 1
                if res["attribution_correct"]:
                    test_correct_attrib += 1

            if is_fault:
                faulty_count += 1
                if detected:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                clean_count += 1
                if detected:
                    false_positives += 1
                else:
                    true_negatives += 1

            if res["attribution_correct"]:
                correct_attributions += 1
            if res["recovery_verified"]:
                successful_recoveries += 1
            if res["false_recovery_claim"]:
                false_recovery_claims_count += 1

            ps = res.get("patch_source", "none")
            patch_sources[ps] = patch_sources.get(ps, 0) + 1

            if res.get("sanitizer_modified_fields"):
                sanitizer_modifications_count += 1

            mode = res.get("investigation_mode", "")
            if mode in ("llm_unavailable_fallback", "llm_invalid_output_fallback"):
                fallback_count += 1

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 1.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 1.0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "method": self.method,
            "status": "COMPLETED",
            "total_cases": total,
            "faulty_cases": faulty_count,
            "clean_cases": clean_count,
            "detection": {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1_score, 4),
                "precision_fraction": f"{true_positives}/{true_positives + false_positives}",
                "recall_fraction": f"{true_positives}/{true_positives + false_negatives}",
            },
            "attribution": {
                "overall_accuracy": round(correct_attributions / total, 4) if total else 0.0,
                "overall_fraction": f"{correct_attributions}/{total}",
                "dev_accuracy": round(dev_correct_attrib / dev_total, 4) if dev_total else 0.0,
                "dev_fraction": f"{dev_correct_attrib}/{dev_total}",
                "test_accuracy": round(test_correct_attrib / test_total, 4) if test_total else 0.0,
                "test_fraction": f"{test_correct_attrib}/{test_total}",
            },
            "recovery": {
                "recovery_verification_rate": round(successful_recoveries / total, 4) if total else 0.0,
                "recovery_verification_fraction": f"{successful_recoveries}/{total}",
                "false_recovery_claims": false_recovery_claims_count,
                "false_recovery_claims_fraction": f"{false_recovery_claims_count}/{total}",
            },
            "provenance_and_fallbacks": {
                "patch_sources": patch_sources,
                "sanitizer_modifications_count": sanitizer_modifications_count,
                "fallback_count": fallback_count,
                "fallback_fraction": f"{fallback_count}/{faulty_count}" if faulty_count else "0/0",
            },
            "case_results": results,
        }
