# SILSILA Phase 2 — Implementation Changelog & Technical Rationale

This document details all modifications made to `silsila_v2` during Phase 2. The original project at `C:\Users\janaa\.gemini\antigravity\scratch\silsila` remains **100% untouched and byte-identical** (verified via SHA-256 manifest).

---

## 1. Summary of Changes by File

### A. `silsila/config.py`
- **Change:** Added `SILSILA_DISABLE_LLM` environment switch:
  ```python
  SILSILA_DISABLE_LLM = os.getenv("SILSILA_DISABLE_LLM", "0") in ("1", "true", "True")
  ```
- **Rationale:** Enables deterministic testing of non-LLM fallback pathways and isolates test suites from environment network / sandbox restrictions.

### B. `silsila/llm_client.py`
- **Change 1:** Updated `OllamaClient.is_available()` and `OllamaClient.generate()` to respect `SILSILA_DISABLE_LLM`.
- **Change 2:** Overhauled `extract_arabic_number_from_text`:
  - Resolved **Finding A.5**: Eliminated the optional day-suffix regex greedy match that falsely captured calendar day numbers like `"من 20 نوفمبر"` as requested leave duration.
  - Prioritizes duration indicators (`لمدة 5 أيام`, `فترة 3 أيام`) and day units (`5 أيام`) over isolated date digits.
- **Change 3:** Added `extract_leave_type_from_text(text: str)`:
  - Detects explicit Arabic textual indicators for `"unpaid"` (بدون راتب, غير مدفوع), `"sick"` (مرضية, مريض), `"emergency"` (طارئة, اضطرارية), and `"annual"` (سنوية, اعتيادية).

### C. `silsila/investigation/detector.py`
- **Change 1:** Added **Semantic Leave Type Invariant** (Invariant 1b):
  - Flags a silent failure alert if the raw query requested annual leave (`سنوية`, `اعتيادية`) but Intake emitted `unpaid`, or vice-versa.
  - Resolves **Finding A.6** (annual-to-unpaid leave-type corruption blind spot).
- **Change 2:** Ground-truth leakage prevention:
  - Ensured `expected_outcome` oracle is strictly optional and not invoked during standard benchmark evaluation.

### D. `silsila/investigation/tools.py`
- **Change 1:** In `audit_arabic_semantics`:
  - Added leave type congruency checks (`has_leave_type_discrepancy`), enriching diagnostic evidence gathered by the Investigator.
- **Change 2:** In `test_counterfactual`:
  - Resolved **Finding A.3** (Weak counterfactual validity): For Step 0 patches, the proposed days and leave type are validated against the original request text evidence. Wrong-but-plausible patches (e.g., proposing 7 or 12 days for a 5-day request) are strictly rejected.
  - Context awareness: Replaced fixed default balances and salaries with values extracted from trace execution context (`previous_balance`, `base_monthly_salary`, `daily_rate`).

### E. `silsila/investigation/investigator.py`
- **Change 1:** Accurate LLM vs Fallback Mode Tracking (Resolves **Finding A.1**):
  - Tracked `llm_invocations_succeeded`.
  - When LLM inference is unreachable or disabled, the resulting report accurately sets:
    ```python
    investigation_mode = "llm_unavailable_fallback"
    ```
    Fallback execution is **never** mislabeled as `agentic_llm`.
- **Change 2:** Enforced Strict Attribution Schema Consistency (Resolves **Finding A.7**):
  - In `_resolve_finish_attribution`, candidate pairs like `("PayrollAgent", 0)` or `("IntakeAgent", 2)` are strictly reconciled against diagnostic evidence. An agent is only attributed to its valid workflow step.
- **Change 3:** Eliminated Deterministic Patch Override of Bad Proposals (Resolves **Finding A.8**):
  - In `_validate_and_sanitize_patch`, removed logic that silently overwrote an LLM's proposed deduction with a mathematically calculated one. If a candidate patch is wrong, it remains as proposed so the verifier can reject or escalate.
- **Change 4:** In `RuleBasedInvestigator`:
  - Added semantic leave-type audit and context-aware balance/salary extraction.

### F. `silsila/investigation/recovery.py`
- **Change:** Context-Aware Sandbox Replay (Resolves **Finding A.4**):
  - Downstream replay agents (`PolicyAgent` and `PayrollAgent`) are initialized using the employee profile context from the trace rather than fixed defaults (21 days / 15,000 SAR).
  - Preserved proposed deduction values during isolated replay so verification can test their actual validity.

### G. `silsila/investigation/verifier.py`
- **Change:** Context-Aware Independent Verification:
  - In `verify()`, recomputed expected outcomes using context extracted from trace steps (`previous_balance`, `base_monthly_salary`, `daily_rate`) rather than fixed constants.

### H. `silsila/evaluation/runner.py`
- **Change:** Decoupled Benchmark Evaluation from Ground-Truth Leakage (Resolves **Finding A.2**):
  - Removed passing `expected_outcome=scenario.expected_final_outcome` into `self.detector.detect()` and `self.verifier.verify()`.
  - Detection and verification operate strictly on business invariants and request evidence. Ground truth is used solely for evaluation metric scoring.

### I. `pyproject.toml` & `requirements.txt`
- **Change 1:** Added `fastapi>=0.100.0` and `uvicorn>=0.22.0` to project dependencies.
- **Change 2:** Registered the `llm` marker in `[tool.pytest.ini_options]` for tests requiring a live local Ollama instance.

### J. `tests/test_first_scenario.py`
- **Change:** Updated assertion to accommodate both live LLM (`confidence=0.90`) and offline fallback (`confidence=0.80`) while validating attribution correctness and verified recovery.

### K. `tests/test_agentic_investigator.py`
- **Change:** Added `@pytest.mark.llm` and `if not OllamaClient().is_available(): pytest.skip(...)` so live LLM tests skip gracefully when Ollama is offline.

### L. `tests/test_research_integrity.py` (New File)
- **Content:** Added 8 targeted regression tests:
  - `test_date_with_duration_does_not_confuse_detector`: Verifies `"من 20 نوفمبر لمدة 5 أيام"` extracts 5 days without triggering false alarms.
  - `test_arabic_indic_numeral_primary_scenario`: Verifies `٥` extracts 5.
  - `test_word_numbers_and_multi_day_terms`: Verifies multi-day terms (أسبوع, أسبوعين, يومين).
  - `test_annual_to_unpaid_leave_type_corruption_detected`: Verifies leave-type errors are detected and healed.
  - `test_wrong_but_plausible_days_patch_rejected`: Verifies counterfactual rejects 7, 12, 1, 20 days for a 5-day request.
  - `test_inconsistent_pairs_reconciled_or_escalated`: Verifies `("PayrollAgent", 0)` is strictly prevented.
  - `test_non_default_salary_and_balance_recovery`: Verifies recovery with custom salary (24,000 SAR) and balance (10 days).
  - `test_disable_llm_switch_uses_fallback_mode`: Verifies `SILSILA_DISABLE_LLM=1` produces `llm_unavailable_fallback`.

---

## 2. Backward Compatibility Verification
- **FastAPI Endpoints:** `POST /api/run` and `GET /api/health` signatures and response field structures (`success`, `original_request`, `final_status`, `trace`, `alert`, `investigation`, `recovery`, `verification`, `comparison`, `events`) are completely preserved.
- **HTML Frontend:** `silsila.html` element IDs and event structure are 100% untouched.
- **FaultInjector Signature:** `__init__(target_stage, mutation, fault_description)` remains identical.
