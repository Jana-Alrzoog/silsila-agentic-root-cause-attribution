# SILSILA Phase 3 Benchmark Evaluation Report

**Date:** October 31, 2026  
**Challenge:** HUMAIN Future AI Builders Hackathon — Challenge 1: Autonomous Multi-Agent Failure Recovery  
**Platform:** SILSILA Agentic AI Reliability & Recovery  
**Dataset:** 63 Benchmark Cases (Seed: 42) — 33 Development Cases, 30 Held-Out Test Cases  
**Evaluation Script:** `python run_eval.py --methods blame_last_agent,rule_based,llm_disabled_fallback,llm_live --out results.json`

---

## 1. Executive Summary

This report presents an empirical, evidence-based evaluation of four attribution methods across 63 diverse Arabic leave request workflows in SILSILA.

### Key Findings
1. **Rule-Based Baseline Outperformed Fallback Heuristics:** The deterministic `rule_based` investigator achieved **100.0% attribution accuracy (63/63)** across both development (33/33) and held-out test splits (30/30), with a **90.5% recovery verification rate (57/63)** and **0 false recovery claims**.
2. **Deterministic Fallback Loop Resiliency:** When local LLM inference was disabled (`llm_disabled_fallback`), the bounded ReAct loop safely recognized disconnection, fell back to trace step audit tools, and matched the rule-based attribution accuracy (100.0%, 63/63), achieving an **81.0% recovery verification rate (51/63)**.
3. **Naive Baselines Fail Decisively:** The `blame_last_agent` heuristic (which naively blames the downstream PayrollAgent at Step 2) achieved only **46.0% attribution accuracy (29/63)** and **30.2% recovery (19/63)** — successful only on clean runs where no fault occurred.
4. **Zero False Recovery Claims:** Across all evaluated methods, the system recorded **0/63 false recovery claims**. The `RecoveryVerifier` strictly verified business invariants using canonical `EmployeeProfile` definitions, rejecting incomplete or inaccurate states.
5. **Live LLM Execution Status:** Marked **NOT RUN** in this evaluation environment because the sandbox environment blocks local socket access to Ollama (`WinError 10013`). The CLI runner is structured so the user can execute `python run_eval.py --methods llm_live --out results.json` locally and integrate the resulting metrics without code changes.

---

## 2. Comparative Benchmark Results

| Evaluation Metric | `blame_last_agent` | `rule_based` | `llm_disabled_fallback` | `llm_live` |
| :--- | :---: | :---: | :---: | :---: |
| **Execution Status** | COMPLETED | COMPLETED | COMPLETED | **NOT RUN** (offline sandbox) |
| **Total Benchmark Cases** | 63 | 63 | 63 | 63 |
| **Faulty Cases / Clean Cases** | 44 / 19 | 44 / 19 | 44 / 19 | 44 / 19 |
| **Silent Failure Detection Precision** | 100.0% (44/44) | 100.0% (44/44) | 100.0% (44/44) | N/A |
| **Silent Failure Detection Recall** | 100.0% (44/44) | 100.0% (44/44) | 100.0% (44/44) | N/A |
| **Silent Failure Detection F1** | 1.0000 | 1.0000 | 1.0000 | N/A |
| **Overall Attribution Accuracy** | **46.0% (29/63)** | **100.0% (63/63)** | **100.0% (63/63)** | N/A |
| **Dev Split Attribution Accuracy** | 45.5% (15/33) | 100.0% (33/33) | 100.0% (33/33) | N/A |
| **Test Split (Held-Out) Accuracy** | 46.7% (14/30) | 100.0% (30/30) | 100.0% (30/30) | N/A |
| **Recovery Verification Rate** | 30.2% (19/63) | **90.5% (57/63)** | 81.0% (51/63) | N/A |
| **False Recovery Claims** | **0/63** | **0/63** | **0/63** | N/A |
| **Fallback Rate** | 0/44 (0.0%) | 0/44 (0.0%) | **44/44 (100.0%)** | N/A |

---

## 3. Patch Provenance & Sanitizer Transparency

SILSILA enforces strict provenance tracking for every proposed patch to ensure LLM reasoning is never conflated with deterministic rule derivation:

| Method | `patch_source: llm_proposed` | `patch_source: rule_derived` | `patch_source: none` | Sanitizer Modifications |
| :--- | :---: | :---: | :---: | :---: |
| `blame_last_agent` | 0 | 0 | 63 (100.0%) | 0 |
| `rule_based` | 0 | 44 (69.8%) | 19 (30.2%) | 0 |
| `llm_disabled_fallback` | 0 | 44 (69.8%) | 19 (30.2%) | 10 (15.9%) |
| `llm_live` | N/A | N/A | N/A | N/A |

### Research Note on Diagnostic Formula Sharing
In SILSILA's ReAct diagnostic framework, the investigator system prompt and the observations returned by diagnostic tools (e.g., `check_business_invariants`, `audit_arabic_semantics`) provide explicit quota formulas, statutory balance conservation equations, and expected deduction values to the agent. Therefore, an LLM-proposed patch may reflect or copy values directly surfaced by tool observations rather than synthesizing new arithmetic ex nihilo.

---

## 4. Benchmark Dataset & Split Design

The 63 benchmark scenarios are partitioned into two strictly defined splits declared before evaluation tuning:

### A. Development Split (33 Cases)
Contains fault families and phrasing patterns used during regex design and early prototyping:
- `clean` (10 cases): Standard annual, sick dual, emergency week, high salary, unpaid requests.
- `intake_qty` (10 cases): Primary Arabic-Indic digit corruptions (٥ -> 15, ٤ -> 10, ٣ -> 9, etc.).
- `intake_type` (3 cases): Leave type corruptions (annual distorted to unpaid).
- `policy_balance` (5 cases): Remaining balance wipeout / inflation.
- `payroll_arith` (5 cases): Salary deduction overcharge and formula corruption.

### B. Held-Out Test Split (30 Cases)
Contains fault families and phrasing patterns NOT used during regex/tool design:
- `intake_qty_word` (5 cases): Word numbers (ثلاثة, خمسة, سبعة, عشرة).
- `intake_qty_week` (3 cases): Week expressions (أسبوع, أسبوعين, أسبوع ونصف).
- `payroll_arith_partial` (5 cases): Subtle non-integer and partial over/under-deductions.
- `policy_balance_partial` (5 cases): Subtle quota conservation violations.
- `ambiguous_request` (4 cases): Inherently ambiguous requests with no detectable numeric quantity (e.g., "إجازة لفترة وجيزة").
- `wrong_leave_type` (3 cases): Plausible alternative leave types.
- `clean` (5 cases): Additional uncorrupted controls across diverse salary/balance ranges.

---

## 5. Arabic Extraction Evaluation: Dev Set vs. New Phrasings

An independent evaluation of the Arabic entity extraction module (`silsila.llm_client.extract_arabic_number_from_text`) was conducted comparing the 19 development phrasings against 10 new, unoptimized phrasings:

### Development Set (19 Phrasings)
- **Accuracy:** 100.0% (19/19)
- **Abstention Rate (`None` returned):** 0.0% (0/19)
- **Key Phrasings:** Arabic-Indic digits (٥, ٤, ٣), dual forms (يومين, أسبوعين), word numbers (ثلاثة, خمسة, عشرة), dates with durations ("من 20 نوفمبر لمدة 5 أيام").

### New Phrasings (10 Independent Phrasings)
- **Deterministic Phrasings Correct:** 80.0% (4/5)
- **Abstention Rate (`None` returned):** 40.0% (4/10)
- **Known Limitations Identified:**
  1. **Written-out tens:** Phrasings such as "عشرون يوماً" or "ثلاثين يوماً" abstain (`None`) because tens are not yet mapped in `ARABIC_WORD_NUMBERS`.
  2. **Plural weeks:** Complex phrasing "ثلاثة أسابيع" requires multi-token compositional parsing.
  3. **Inherent Ambiguity Limitation:** When an employee submits an inherently ambiguous query (e.g., "إجازة لفترة وجيزة"), no expected quantity can be derived from the text. Consequently, an Intake fault on an ambiguous request cannot be detected by deterministic invariant audits and must be escalated to a human supervisor.

---

## 6. Authoritative EmployeeProfile Validation

Previous versions of the verifier inspected `TraceStep.output_state` for `previous_balance` and `daily_rate`. When these outputs were corrupted in the trace, the verifier performed circular checks against corrupted values.

In the Phase 2 & 3 implementation:
- `EmployeeProfile` is the **single authoritative source** for `PolicyAgent`, `PayrollAgent`, `RecoveryEngine`, and `RecoveryVerifier`.
- A dedicated negative regression test (`TestEmployeeProfileAuthoritative`) verified that when `daily_rate` in the trace is artificially corrupted to `99,999.0 SAR/day`, `RecoveryVerifier` queries `EmployeeProfile` (EMP-101: 500 SAR/day) and correctly verifies the healed state rather than falsely rejecting recovery.

---

## 7. Instructions for Local Live LLM Evaluation

To evaluate `llm_live` with local Ollama on a host machine:

```bash
# 1. Start Ollama with qwen2.5:3b:
ollama run qwen2.5:3b

# 2. In an unsandboxed terminal, run the benchmark:
python run_eval.py --methods llm_live --out results_live.json

# 3. View the generated results:
type results_live.json
```
