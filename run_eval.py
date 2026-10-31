"""CLI runner for SILSILA Phase 3 held-out evaluation benchmark.

Usage:
    python run_eval.py --methods blame_last_agent,rule_based,llm_disabled_fallback,llm_live --out results.json
    python run_eval.py --methods rule_based
    python run_eval.py --methods llm_live --out live_results.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List

from silsila.evaluation.dataset import PHASE3_BENCHMARK
from silsila.evaluation.runner import BenchmarkRunner


def parse_args():
    parser = argparse.ArgumentParser(description="SILSILA Multi-Method Evaluation Benchmark")
    parser.add_argument(
        "--methods",
        type=str,
        default="blame_last_agent,rule_based,llm_disabled_fallback,llm_live",
        help="Comma-separated list of methods to evaluate. Options: blame_last_agent, rule_based, llm_disabled_fallback, llm_live (or 'all')",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results.json",
        help="Output JSON file path for evaluation metrics and detailed results",
    )
    return parser.parse_args()


def print_method_summary(res: Dict):
    method = res["method"]
    status = res["status"]
    print(f"\nMethod: {method.upper()} [{status}]")
    print("-" * 75)

    if status == "NOT RUN":
        print(f"  Reason: {res.get('reason')}")
        print(f"  Instructions: {res.get('instructions')}")
        return

    det = res["detection"]
    att = res["attribution"]
    rec = res["recovery"]
    prov = res["provenance_and_fallbacks"]

    print(f"  Total Cases:               {res['total_cases']} (Faulty: {res['faulty_cases']}, Clean: {res['clean_cases']})")
    print(f"  Detection Precision:       {det['precision'] * 100:.1f}% ({det['precision_fraction']})")
    print(f"  Detection Recall:          {det['recall'] * 100:.1f}% ({det['recall_fraction']})")
    print(f"  Detection F1:              {det['f1']:.4f}")
    print(f"  Attribution Accuracy:      {att['overall_accuracy'] * 100:.1f}% ({att['overall_fraction']})")
    print(f"    - Dev Split:             {att['dev_accuracy'] * 100:.1f}% ({att['dev_fraction']})")
    print(f"    - Test Split (Held-out): {att['test_accuracy'] * 100:.1f}% ({att['test_fraction']})")
    print(f"  Recovery Verification:     {rec['recovery_verification_rate'] * 100:.1f}% ({rec['recovery_verification_fraction']})")
    print(f"  False Recovery Claims:     {rec['false_recovery_claims_fraction']}")
    print(f"  Patch Sources:             llm_proposed={prov['patch_sources'].get('llm_proposed', 0)}, "
          f"rule_derived={prov['patch_sources'].get('rule_derived', 0)}, "
          f"none={prov['patch_sources'].get('none', 0)}")
    print(f"  Fallback Rate:             {prov['fallback_fraction']}")


def print_comparison_table(results_by_method: Dict[str, Dict]):
    print("\n" + "=" * 90)
    print("SILSILA PHASE 3 BENCHMARK: COMPARATIVE METHOD SUMMARY")
    print("=" * 90)
    print(f"{'Method':<25} | {'Status':<10} | {'Attrib (All)':<15} | {'Attrib (Dev)':<15} | {'Attrib (Test)':<15} | {'Recovery':<12}")
    print("-" * 90)

    for method, res in results_by_method.items():
        status = res.get("status", "UNKNOWN")
        if status == "NOT RUN":
            print(f"{method:<25} | {'NOT RUN':<10} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15} | {'N/A':<12}")
        else:
            att = res["attribution"]
            rec = res["recovery"]
            all_str = f"{att['overall_accuracy']*100:.1f}% ({att['overall_fraction']})"
            dev_str = f"{att['dev_accuracy']*100:.1f}% ({att['dev_fraction']})"
            test_str = f"{att['test_accuracy']*100:.1f}% ({att['test_fraction']})"
            rec_str = f"{rec['recovery_verification_rate']*100:.1f}%"
            print(f"{method:<25} | {status:<10} | {all_str:<15} | {dev_str:<15} | {test_str:<15} | {rec_str:<12}")

    print("=" * 90)


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()

    if args.methods.strip().lower() == "all":
        methods = BenchmarkRunner.VALID_METHODS
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print("=" * 75)
    print("SILSILA: Autonomous Agentic Failure Investigation & Recovery Benchmark")
    print(f"Evaluating {len(methods)} methods across {len(PHASE3_BENCHMARK)} cases (Fixed RNG seed: 42)")
    print("=" * 75)

    all_results = {}

    for method in methods:
        if method not in BenchmarkRunner.VALID_METHODS:
            print(f"Warning: Skipping unknown method '{method}'")
            continue

        print(f"\n[+] Running evaluation for: {method} ...")
        runner = BenchmarkRunner(method=method)
        summary = runner.run_all(PHASE3_BENCHMARK)
        all_results[method] = summary
        print_method_summary(summary)

    print_comparison_table(all_results)

    # Save to out file
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results successfully saved to: {args.out}")


if __name__ == "__main__":
    main()
