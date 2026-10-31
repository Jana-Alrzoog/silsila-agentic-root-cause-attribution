<div align="center">

# SILSILA

### Evidence-Grounded Root-Cause Attribution & Verified Recovery for Multi-Agent Systems

**Know Who Broke the Chain. Prove Why. Recover Safely.**

*An agentic reliability prototype for tracing silent failures to their originating agent and execution step.*

</div>

## Overview

In multi-agent workflows, an upstream error can propagate through several agents and surface as a plausible but incorrect final output. Detecting the anomaly is only the beginning: **which agent introduced it, what evidence supports that attribution, and can a correction be verified?**

**SILSILA** combines **structured execution traces, semantic and business invariants, tool-assisted investigation, counterfactual testing, sandboxed replay, and independent post-repair verification**. Its current proof of concept uses an Arabic HR leave workflow with three sequential agents; the architecture is intended to explore wider reliability patterns, not claim production-wide generalization.

## Architecture

<img width="1672" height="941" alt="Multi-Agent HR Leave Workflow" src="https://github.com/user-attachments/assets/f2361908-40ee-4cb6-8e89-3441b1ca5609" />

*Conceptual illustration of the investigation workflow. Continuous learning, automatic approval, and some depicted tool names are illustrative rather than validated implementation features.*

```text
Arabic Request → IntakeAgent → PolicyAgent → PayrollAgent
                         ↓ Structured execution trace
                  SilentFailureDetector
                         ↓ Violation + evidence
                AgenticInvestigator (Qwen2.5 / Ollama)
                         ↓ Diagnostic tools + attribution
                 Counterfactual Patch Test
                         ↓
               RecoveryEngine (sandbox replay)
                         ↓
                  RecoveryVerifier
                    ↙          ↘
            Verified Recovery   Human Escalation
```

The **EmployeeProfile** supplies authoritative leave-balance and salary information to the workflow and verifier, avoiding reliance on potentially corrupted intermediate outputs.

## Core Capabilities

| Capability | Engineering mechanism |
|---|---|
| **Silent-failure detection** | Arabic semantic consistency checks and domain-specific business invariants |
| **Agent/step attribution** | Trace-level fault localization supported by diagnostic observations |
| **Agentic investigation** | Bounded, ReAct-style Qwen2.5:3B loop with `audit_arabic_semantics`, `check_business_invariants`, and `test_counterfactual` |
| **Safe recovery** | Minimal candidate patches, counterfactual checks, and downstream sandbox replay |
| **Verification & escalation** | Authoritative-state invariant checks; unsafe or unverified repairs are not claimed as successful |
| **Auditability** | Explicit `investigation_mode`, `patch_source`, and `sanitizer_modified_fields` |

**Example — originating fault vs. visible symptom:** For `أبي إجازة ٥ أيام`, an injected `IntakeAgent` error transforms **5 → 15 days**. The corrupted value propagates into downstream policy/payroll calculations. SILSILA attributes the defect to **IntakeAgent (Step 0)**, tests `{"days": 5}`, replays the affected workflow, and verifies a restored **16-day balance** and **0 SAR deduction**. The agent producing the final financial output was *not* necessarily the root cause.

## Evaluation

A **controlled, fixed-seed benchmark** contains **63 scenarios** (**44 faulty, 19 clean**), split into **33 development** and **30 held-out** cases.

| Method | Attribution accuracy | Verified recovery |
|---|---:|---:|
| Blame last agent | 46.0% (29/63) | 30.2% (19/63) |
| Rule-based auditor | **100.0% (63/63)** | **90.5% (57/63)** |
| LLM-disabled fallback | **100.0% (63/63)** | 81.0% (51/63) |
| Live Qwen investigator | **Not benchmarked** | **Not benchmarked** |

**All three completed offline methods recorded 0/63 false recovery claims** under the benchmark's verification criteria. Four interactive workflow scenarios also completed with expected outcomes; some live investigations encountered Ollama timeouts and used explicit fallback logic. **These figures do not establish fallback-free LLM accuracy or generalization to arbitrary multi-agent systems.** Full methodology and split-level results: [`EVAL_REPORT.md`](EVAL_REPORT.md) · [`results.json`](results.json).

## Quick Start

**Requirements:** Python 3.12, dependencies in `requirements.txt`; optional [Ollama](https://ollama.com/) for live Qwen investigation.

```bash
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b  # optional: for live LLM investigation
python -m uvicorn web.server:app --host 127.0.0.1 --port 8001
```

Open **http://127.0.0.1:8001/** and select **Recorded Replay** or **Live AI Run**. The backend exposes `GET /api/health` and `POST /api/run`.

```bash
# Offline regression tests
python -m pytest tests/ -q -m "not llm"

# Reproduce the controlled benchmark without overwriting published results
python run_eval.py --methods blame_last_agent,rule_based,llm_disabled_fallback --out results_reproduced.json
```

**Reported offline validation:** 45 tests passed; 2 LLM-dependent tests excluded. Local CPU inference may be slow and may trigger clearly labeled fallback modes.

## Research Scope

SILSILA is a **research-inspired prototype**, not an unrestricted causal-inference engine. Counterfactual replay evaluates candidate interventions within a **controlled three-agent HR simulator**. Performance depends on trace quality and the coverage of domain invariants; ambiguous inputs and new fault families remain open challenges. Diagnostic tools may expose formulas or expected values to the LLM, so LLM patch proposals should not be treated as independent discovery. Broader workloads, independent datasets, and completed live-LLM benchmarking are future evaluation needs.

---

<div align="center"><strong>Detect the symptom. Attribute the origin. Verify the recovery.</strong></div>
