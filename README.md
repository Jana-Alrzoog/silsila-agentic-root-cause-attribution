# SILSILA

### Evidence-Grounded Root-Cause Attribution & Verified Recovery for Multi-Agent Systems

**Know Who Broke the Chain. Prove Why. Recover Safely.**

*An agentic reliability framework for tracing silent failures to their originating agent and execution step.*

---

## Overview

**SILSILA** is a research-inspired agentic reliability framework designed to address a critical challenge in multi-agent systems: **identifying which agent introduced a failure, at which execution step, and whether a corrective intervention can safely restore system integrity.**

Unlike monitoring approaches that focus primarily on detecting faulty outcomes, SILSILA investigates the **origin of failure** through structured execution traces, evidence-grounded root-cause attribution, LLM-driven diagnostic tool use, and counterfactual intervention testing.

By integrating **tool-augmented investigation, sandboxed workflow replay, and invariant-based recovery verification**, SILSILA establishes an auditable process for **fault localization, corrective intervention, and validated recovery**.

The current prototype demonstrates this approach through an Arabic HR leave-processing workflow involving three sequential agents: `IntakeAgent`, `PolicyAgent`, and `PayrollAgent`.

## System Architecture

<img width="1672" height="941" alt="Multi-Agent HR Leave Workflow" src="https://github.com/user-attachments/assets/04a9022f-5df5-44ff-acca-dc4d54001ec7" />

*Conceptual system architecture. Some illustrated elements represent potential extensions rather than validated implementation features.*

SILSILA follows five stages:

1. **Detect** — Identify silent inconsistencies using Arabic semantic validation and business invariants.
2. **Investigate** — Examine execution traces with a tool-augmented, ReAct-style LLM investigator.
3. **Attribute** — Localize the originating agent and execution step using diagnostic evidence.
4. **Recover** — Evaluate minimal patches through counterfactual testing and isolated downstream replay.
5. **Verify** — Validate recovered outcomes against authoritative state and domain invariants, escalating cases that cannot be safely resolved.

## Technical Capabilities

| Capability | Engineering Mechanism |
|---|---|
| **Multi-Agent Execution** | Python-based Intake, Policy, and Payroll agents |
| **Silent Failure Detection** | Arabic semantic consistency and business invariant checks |
| **Agentic Investigation** | Qwen2.5:3B via Ollama with bounded ReAct-style tool use |
| **Root-Cause Attribution** | Trace-level agent and step fault localization |
| **Counterfactual Recovery** | Candidate patch validation and sandboxed downstream replay |
| **Independent Verification** | Authoritative `EmployeeProfile` and post-recovery invariant checks |
| **Auditability** | `investigation_mode`, `patch_source`, and `sanitizer_modified_fields` |
| **Application Layer** | FastAPI backend and interactive HTML dashboard |

### Root-Cause Attribution in Action

**Input:** `أبي إجازة ٥ أيام`

An injected `IntakeAgent` fault transforms **5 days into 15 days**, propagating the corrupted value through downstream policy and payroll calculations.

SILSILA attributes the failure to **IntakeAgent — Step 0**, evaluates the correction `{"days": 5}`, replays the affected workflow, and verifies:

- **Remaining leave balance:** 16 days
- **Salary deduction:** 0 SAR
- **Recovery verification:** Passed

**Key insight:** The agent producing the incorrect final output is not necessarily the agent responsible for introducing the failure.

## Live Agentic Investigation

A real **Qwen2.5:3B** investigation successfully completed end-to-end using Ollama on CPU, without deterministic fallback.

| Metric | Validated Result |
|---|---|
| **Live Smoke Test** | **Passed (1/1)** |
| Investigation Mode | `agentic_llm` |
| Patch Provenance | `llm_proposed` |
| Root Cause | IntakeAgent — Step 0 |
| Diagnostic Tools | Semantic Audit · Agent Handoff Comparison · Counterfactual Testing |
| Proposed Correction | `{"days": 5}` |
| Execution Time | 43.47 seconds |
| Recovery Verification | Passed |

This validates operational **LLM-driven, tool-assisted investigation and verified recovery** in the demonstrated scenario, rather than a fallback-only implementation.

## Evaluation

SILSILA was evaluated on a **63-scenario controlled benchmark** comprising **44 faulty and 19 clean executions**, partitioned into **33 development** and **30 held-out test** cases.

| Method | Attribution Accuracy | Verified Recovery |
|---|---:|---:|
| Blame Last Agent | 46.0% | 30.2% |
| Rule-Based Auditor | **100.0%** | **90.5%** |
| LLM-Disabled Fallback | **100.0%** | 81.0% |

All three completed offline methods recorded **zero false recovery claims** under the benchmark's verification criteria.

The deterministic benchmark and the successful live Qwen smoke test evaluate different aspects of SILSILA. The live result demonstrates functional LLM tool use; full-scale LLM benchmarking remains future work.

**Reproducibility:** [Evaluation Report](EVAL_REPORT.md) · [Benchmark Results](results.json)

## Quick Start

**Requirements:** Python 3.12 and project dependencies. Ollama is required for live LLM investigation.

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Download Qwen for live investigation
ollama pull qwen2.5:3b

# Start SILSILA
python -m uvicorn web.server:app --host 127.0.0.1 --port 8001
```

Open **http://127.0.0.1:8001/** and select **Recorded Replay** or **Live AI Run**.

To reproduce the offline evaluations:

```bash
# Run offline regression tests
python -m pytest tests/ -q -m "not llm"

# Reproduce benchmark results
python run_eval.py --methods blame_last_agent,rule_based,llm_disabled_fallback --out results_reproduced.json
```

**Validation:** 45 offline regression tests passed, with three LLM-dependent tests deselected. A separate live Qwen investigation smoke test passed.

## Research Scope & Limitations

SILSILA is a **research-inspired prototype**, currently evaluated within a controlled three-agent HR workflow. Its counterfactual interventions demonstrate recovery consistency within the simulator, rather than unrestricted causal identification.

Performance depends on trace quality, semantic interpretation, and domain-invariant coverage. Diagnostic tools may expose expected values to the LLM; therefore, proposed patches do not necessarily represent independent numerical discovery. Generalization to unseen workflows and broader fault families requires further evaluation.

---

**Detect the symptom. Attribute the origin. Verify the recovery.**
