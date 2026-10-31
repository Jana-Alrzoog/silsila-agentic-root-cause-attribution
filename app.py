"""SILSILA (AgentRescue) — Enterprise Agent Reliability Platform.

HUMAIN Hackathon — Challenge 1: Autonomous Multi-Agent Failure Recovery
Detect. Investigate. Recover. Verify.
"""

import base64
import json
import os
import time
import streamlit as st

from silsila.pipeline import WorkflowPipeline, FaultInjector
from silsila.investigation.detector import SilentFailureDetector
from silsila.investigation.investigator import (
    RootCauseInvestigator,
    AgenticInvestigator,
    RuleBasedInvestigator,
)
from silsila.investigation.recovery import RecoveryEngine
from silsila.investigation.verifier import RecoveryVerifier
from silsila.evaluation.dataset import BENCHMARK_SCENARIOS
from silsila.evaluation.runner import BenchmarkRunner

# ----------------------------------------------------
# Base Directory & Asset Handling
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "silsila_logo.png")

logo_b64 = ""
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode("utf-8")

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="SILSILA | Enterprise Agent Reliability",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------
# Enterprise Design System (HUMAIN Inspired)
# ----------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #F5F8F9;
        color: #101B30;
    }}

    /* Main Container Padding */
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }}

    /* Top Navigation Header */
    .silsila-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E4EBEF;
        border-radius: 12px;
        padding: 14px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(16, 27, 48, 0.03);
    }}
    .silsila-brand-group {{
        display: flex;
        align-items: center;
        gap: 16px;
    }}
    .silsila-logo-img {{
        height: 38px;
        width: auto;
        object-fit: contain;
    }}
    .silsila-hackathon-pill {{
        background: #F0F4F7;
        color: #5A6B82;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #E1E8EE;
    }}
    .silsila-status-group {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    /* Badges & Pills */
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 9px;
        border-radius: 5px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}
    .badge-teal {{
        background-color: #E6F7F9;
        color: #008C9E;
        border: 1px solid #B8E8ED;
    }}
    .badge-mint {{
        background-color: #E8FAF6;
        color: #0CA58D;
        border: 1px solid #B4F0E4;
    }}
    .badge-danger {{
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
    }}
    .badge-amber {{
        background-color: #FFFBEB;
        color: #D97706;
        border: 1px solid #FDE68A;
    }}
    .badge-slate {{
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
    }}
    .badge-outline {{
        background-color: #FFFFFF;
        color: #101B30;
        border: 1px solid #D2DDE6;
    }}

    /* Card Panels */
    .h-card {{
        background: #FFFFFF;
        border: 1px solid #E4EBEF;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(16, 27, 48, 0.04);
    }}
    .h-card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #F0F4F7;
    }}
    .h-card-title {{
        font-size: 0.92rem;
        font-weight: 700;
        color: #101B30;
        letter-spacing: -0.01em;
        text-transform: uppercase;
    }}
    .h-card-subtitle {{
        font-size: 0.78rem;
        color: #6B7C93;
        font-weight: 400;
        margin-top: 2px;
    }}

    /* Metric Cards (HUMAIN Style) */
    .kpi-container {{
        background: #FFFFFF;
        border: 1px solid #E4EBEF;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(16, 27, 48, 0.03);
    }}
    .kpi-label {{
        font-size: 0.70rem;
        font-weight: 700;
        color: #6B7C93;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 1.55rem;
        font-weight: 800;
        color: #101B30;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    .kpi-subtext {{
        font-size: 0.76rem;
        color: #8C9BAE;
        margin-top: 4px;
        font-weight: 500;
    }}

    /* Arabic Input Box Display */
    .arabic-display {{
        direction: rtl;
        text-align: right;
        font-family: 'IBM Plex Sans Arabic', -apple-system, sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #101B30;
        background: #FFFFFF;
        border: 1px solid #E4EBEF;
        border-right: 4px solid #008C9E;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 16px;
    }}

    /* Workflow Pipeline Step Card */
    .pipeline-step-card {{
        background: #FFFFFF;
        border: 1px solid #E4EBEF;
        border-radius: 8px;
        padding: 14px;
        transition: all 0.15s ease-in-out;
    }}
    .pipeline-step-card.faulty {{
        border-color: #DC2626;
        background-color: #FFFDFD;
        border-left: 4px solid #DC2626;
    }}
    .pipeline-step-card.cascade {{
        border-color: #D97706;
        background-color: #FFFEFA;
        border-left: 4px solid #D97706;
    }}
    .pipeline-step-card.healthy {{
        border-color: #0CA58D;
        background-color: #FDFFFE;
        border-left: 4px solid #0CA58D;
    }}
    .step-title {{
        font-size: 0.88rem;
        font-weight: 700;
        color: #101B30;
    }}
    .step-role {{
        font-size: 0.75rem;
        color: #6B7C93;
        margin-bottom: 8px;
    }}

    /* Investigation ReAct Timeline */
    .react-item {{
        position: relative;
        padding-left: 18px;
        margin-left: 10px;
        border-left: 2px solid #008C9E;
        padding-bottom: 14px;
    }}
    .react-dot {{
        position: absolute;
        left: -6px;
        top: 2px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #008C9E;
    }}
    .react-header {{
        font-size: 0.82rem;
        font-weight: 700;
        color: #101B30;
        margin-bottom: 4px;
    }}
    .react-thought {{
        font-size: 0.80rem;
        color: #334155;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 8px 12px;
        line-height: 1.45;
    }}

    /* State Comparison Grid */
    .diff-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
    }}
    .diff-table th {{
        text-align: left;
        padding: 8px 10px;
        background: #F5F8F9;
        color: #5A6B82;
        font-weight: 600;
        font-size: 0.72rem;
        text-transform: uppercase;
        border-bottom: 1px solid #E4EBEF;
    }}
    .diff-table td {{
        padding: 8px 10px;
        border-bottom: 1px solid #F0F4F7;
        color: #101B30;
    }}
    .val-corrupt {{
        color: #DC2626;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }}
    .val-recovered {{
        color: #0CA58D;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }}

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #E4EBEF;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.5rem;
    }}
    
    /* Code blocks */
    code {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.80rem !important;
        background-color: #F1F5F9;
        color: #0F172A;
        padding: 2px 5px;
        border-radius: 4px;
    }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Top Navigation Header
# ----------------------------------------------------
header_logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" class="silsila-logo-img" alt="SILSILA Logo"/>'
    if logo_b64
    else '<span style="font-weight: 800; font-size: 1.3rem; color: #101B30; letter-spacing: -0.02em;">SILSILA</span>'
)

st.markdown(f"""
<div class="silsila-header">
    <div class="silsila-brand-group">
        {header_logo_html}
        <span class="silsila-hackathon-pill">Built for HUMAIN Hackathon — Challenge 1</span>
    </div>
    <div class="silsila-status-group">
        <span class="badge badge-teal">LOCAL OLLAMA: QWEN2.5:3B</span>
        <span class="badge badge-mint">SANDBOX ISOLATION: ACTIVE</span>
        <span class="badge badge-slate">DECISIVE ATTRIBUTION</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Top Navigation Bar (Enterprise View Selector)
# ----------------------------------------------------
selected_view = st.radio(
    label="Workspace View",
    options=[
        "Live Workflow Studio",
        "Investigation Inspector",
        "Evaluation Lab",
        "Architecture & Research",
    ],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
)

# ----------------------------------------------------
# Sidebar: Presets, Fault Sandbox & Controls
# ----------------------------------------------------
with st.sidebar:
    if logo_b64:
        st.markdown(
            f'<div style="text-align:center; padding: 4px 0 14px 0;">'
            f'<img src="data:image/png;base64,{logo_b64}" style="max-height: 34px; width: auto;" alt="Logo"/>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div style="font-size: 0.72rem; font-weight: 700; color: #6B7C93; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;">Scenario Presets</div>', unsafe_allow_html=True)
    
    preset_options = [
        "Primary Demo: أبي إجازة ٥ أيام (Intake ٥ -> 15)",
        "Clean Control: طلب إجازة سنوية 3 أيام (Clean Run)",
        "Policy Bug: أرغب في إجازة اعتيادية 2 يوم (Quota Wiped)",
        "Payroll Bug: إجازة بدون راتب 3 أيام (Triple Deduction)",
        "Custom Arabic Request",
    ]
    chosen_preset = st.selectbox("Select Benchmark Preset", preset_options, index=0)

    if chosen_preset.startswith("Primary Demo"):
        default_query = "أبي إجازة ٥ أيام"
        default_stage = "intake"
        default_mut = {"days": 15}
    elif chosen_preset.startswith("Clean Control"):
        default_query = "طلب إجازة سنوية لمدة 3 أيام تبدأ الأحد"
        default_stage = "none"
        default_mut = {}
    elif chosen_preset.startswith("Policy Bug"):
        default_query = "أرغب في إجازة اعتيادية لمدة 2 يوم"
        default_stage = "policy"
        default_mut = {"remaining_balance": 0}
    elif chosen_preset.startswith("Payroll Bug"):
        default_query = "طلب إجازة بدون راتب لمدة 3 أيام"
        default_stage = "payroll"
        default_mut = {"salary_deduction": 4500.0, "final_monthly_net": 10500.0}
    else:
        default_query = "أبي إجازة ٥ أيام"
        default_stage = "intake"
        default_mut = {"days": 15}

    st.markdown("---")
    st.markdown('<div style="font-size: 0.72rem; font-weight: 700; color: #6B7C93; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Request Parameters</div>', unsafe_allow_html=True)
    user_query = st.text_input("Arabic Leave Request", value=default_query)
    emp_id = st.text_input("Employee ID", value="EMP-101")

    st.markdown("---")
    st.markdown('<div style="font-size: 0.72rem; font-weight: 700; color: #6B7C93; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Fault Injection Sandbox</div>', unsafe_allow_html=True)
    
    stage_choices = ["none", "intake", "policy", "payroll"]
    target_stage = st.selectbox(
        "Injected Fault Stage",
        stage_choices,
        index=stage_choices.index(default_stage),
    )

    mutation_payload = default_mut
    if target_stage == "intake":
        injected_days = st.number_input("Injected Days in Intake Output", value=15, min_value=1, max_value=30)
        mutation_payload = {"days": int(injected_days)}
    elif target_stage == "policy":
        mutation_payload = {"remaining_balance": 0}
        st.caption("Fault: Forces remaining leave quota to 0 days.")
    elif target_stage == "payroll":
        mutation_payload = {"salary_deduction": 4500.0, "final_monthly_net": 10500.0}
        st.caption("Fault: Triples standard deduction to 4,500 SAR.")

    st.markdown('<div style="font-size: 0.70rem; color: #8C9BAE; line-height: 1.35; margin-top: 4px;">Investigator operates with strict zero-knowledge: ground truth fault metadata is never exposed.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size: 0.72rem; font-weight: 700; color: #6B7C93; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Attribution Engine</div>', unsafe_allow_html=True)
    engine_choice = st.radio(
        "Investigator Mode",
        ["Agentic AI (Ollama qwen2.5:3b + ReAct Tools)", "Rule-Based Baseline (Procedural DAG)"],
        index=0,
    )

    run_clicked = st.button("Execute Workflow & Investigate", type="primary", use_container_width=True)


# ----------------------------------------------------
# Pipeline Execution & Caching
# ----------------------------------------------------
if "current_trace" not in st.session_state or run_clicked:
    with st.spinner("Executing multi-agent workflow (Intake -> Policy -> Payroll)..."):
        fault_injector = None
        if target_stage != "none":
            fault_injector = FaultInjector(
                target_stage=target_stage,
                mutation=mutation_payload,
                fault_description=f"Injected silent fault at {target_stage}",
            )

        pipeline = WorkflowPipeline()
        trace = pipeline.run(query=user_query, employee_id=emp_id, fault_injector=fault_injector)

        # 1. Detection
        detector = SilentFailureDetector()
        alert = detector.detect(trace)

        # 2. Investigation
        is_agentic = "Agentic AI" in engine_choice
        start_time = time.perf_counter()
        if is_agentic:
            investigator = AgenticInvestigator(max_tool_calls=5)
            report = investigator.investigate(trace, alert)
        else:
            investigator = RuleBasedInvestigator()
            report = investigator.investigate(trace, alert)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # 3. Recovery & Independent Verification
        recovery = None
        verified = None
        if alert.detected and not report.escalated:
            rec_engine = RecoveryEngine()
            recovery = rec_engine.recover(trace, report)
            verifier = RecoveryVerifier()
            verified = verifier.verify(trace, recovery)

        st.session_state.current_trace = trace
        st.session_state.current_alert = alert
        st.session_state.current_report = report
        st.session_state.current_recovery = recovery
        st.session_state.current_verified = verified
        st.session_state.elapsed_ms = elapsed_ms

trace = st.session_state.current_trace
alert = st.session_state.current_alert
report = st.session_state.current_report
recovery = st.session_state.current_recovery
verified = st.session_state.current_verified
elapsed_ms = st.session_state.elapsed_ms

step_0 = trace.get_step(0)
step_1 = trace.get_step(1)
step_2 = trace.get_step(2)


# ====================================================
# VIEW 1: LIVE WORKFLOW STUDIO (Split-View Workspace)
# ====================================================
if selected_view == "Live Workflow Studio":

    # KPI Summary Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown('<div class="kpi-container"><div class="kpi-label">Detection Status</div>', unsafe_allow_html=True)
        if alert.detected:
            st.markdown('<span class="badge badge-danger">VIOLATION DETECTED</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-subtext">{alert.invariant_violated}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-mint">INVARIANTS PASS</span>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-subtext">No silent failure</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with k2:
        st.markdown('<div class="kpi-container"><div class="kpi-label">Root Cause Origin</div>', unsafe_allow_html=True)
        if alert.detected:
            if report.escalated:
                st.markdown('<span class="badge badge-amber">ESCALATED</span>', unsafe_allow_html=True)
                st.markdown('<div class="kpi-subtext">Human inspection needed</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge badge-danger">{report.faulty_agent}</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="kpi-subtext">Decisive Step: {report.faulty_step}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-slate">NONE</span>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-subtext">Execution nominal</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with k3:
        st.markdown('<div class="kpi-container"><div class="kpi-label">Investigator Loop</div>', unsafe_allow_html=True)
        if report.investigation_mode == "agentic_llm":
            st.markdown('<span class="badge badge-teal">AUTONOMOUS AGENT</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-subtext">{len(report.investigation_timeline)} diagnostic tool calls</div>', unsafe_allow_html=True)
        elif report.investigation_mode == "escalated":
            st.markdown('<span class="badge badge-amber">ESCALATED</span>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-subtext">Bounded fallback</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-slate">RULE-BASED</span>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-subtext">Procedural traversal</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with k4:
        st.markdown('<div class="kpi-container"><div class="kpi-label">Recovery Verification</div>', unsafe_allow_html=True)
        if verified and verified.verification_passed:
            st.markdown('<span class="badge badge-mint">VERIFIED & HEALED</span>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-subtext">Sandbox replay passed</div>', unsafe_allow_html=True)
        elif alert.detected:
            st.markdown('<span class="badge badge-danger">UNRECOVERED</span>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-subtext">Verification pending</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-mint">CLEAN STATE</span>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-subtext">No recovery required</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with k5:
        st.markdown('<div class="kpi-container"><div class="kpi-label">Diagnosis Latency</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value">{elapsed_ms:,.0f} ms</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-subtext">Confidence: {report.confidence * 100:.0f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # Split-View: Left Column (Workflow Execution) vs Right Column (Reliability Inspector)
    left_col, right_col = st.columns([1.05, 1.15])

    # ------------------ LEFT COLUMN: WORKFLOW STREAM ------------------
    with left_col:
        st.markdown("""
        <div class="h-card">
            <div class="h-card-header">
                <div>
                    <div class="h-card-title">Production Workflow Stream</div>
                    <div class="h-card-subtitle">User Arabic Request & Multi-Agent Execution Chain</div>
                </div>
                <span class="badge badge-outline">DAG: Sequential 3-Step</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="arabic-display">{user_query}</div>', unsafe_allow_html=True)

        # Agent Step 0: IntakeAgent
        is_root_0 = alert.detected and report.faulty_step == 0
        card_class_0 = "faulty" if is_root_0 else ("healthy" if verified and verified.verification_passed else "healthy")
        badge_0 = '<span class="badge badge-danger">ROOT CAUSE</span>' if is_root_0 else '<span class="badge badge-mint">HEALTHY</span>'
        
        st.markdown(f"""
        <div class="pipeline-step-card {card_class_0}" style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="step-title">Step 0: IntakeAgent</div>
                {badge_0}
            </div>
            <div class="step-role">LLM Natural Language Understanding (qwen2.5:3b)</div>
            <div style="font-size: 0.80rem; line-height: 1.5; color: #334155;">
                Parsed Days: <b>{step_0.output_state.get('days')}</b> | Leave Type: <b>{step_0.output_state.get('leave_type')}</b><br>
                Self-Reported Status: <span class="badge badge-mint">SUCCESS</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Agent Step 1: PolicyAgent
        is_root_1 = alert.detected and report.faulty_step == 1
        is_cascade_1 = alert.detected and report.faulty_step == 0
        card_class_1 = "faulty" if is_root_1 else ("cascade" if is_cascade_1 else "healthy")
        badge_1 = (
            '<span class="badge badge-danger">ROOT CAUSE</span>' if is_root_1 
            else ('<span class="badge badge-amber">CASCADE CORRUPTED</span>' if is_cascade_1 else '<span class="badge badge-mint">HEALTHY</span>')
        )
        st.markdown(f"""
        <div class="pipeline-step-card {card_class_1}" style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="step-title">Step 1: PolicyAgent</div>
                {badge_1}
            </div>
            <div class="step-role">Saudi Labor Law Quota & Entitlement Policy</div>
            <div style="font-size: 0.80rem; line-height: 1.5; color: #334155;">
                Approved Paid: <b>{step_1.output_state.get('approved_paid_days')} days</b> | Unpaid: <b>{step_1.output_state.get('unpaid_days')} days</b><br>
                Remaining Balance: <b>{step_1.output_state.get('remaining_balance')} days</b> (from initial 21)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Agent Step 2: PayrollAgent
        is_root_2 = alert.detected and report.faulty_step == 2
        is_cascade_2 = alert.detected and report.faulty_step in [0, 1]
        card_class_2 = "faulty" if is_root_2 else ("cascade" if is_cascade_2 else "healthy")
        badge_2 = (
            '<span class="badge badge-danger">ROOT CAUSE</span>' if is_root_2 
            else ('<span class="badge badge-amber">CASCADE CORRUPTED</span>' if is_cascade_2 else '<span class="badge badge-mint">HEALTHY</span>')
        )
        st.markdown(f"""
        <div class="pipeline-step-card {card_class_2}" style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="step-title">Step 2: PayrollAgent</div>
                {badge_2}
            </div>
            <div class="step-role">Payroll Deduction & Compensation Accounting</div>
            <div style="font-size: 0.80rem; line-height: 1.5; color: #334155;">
                Salary Deduction: <b>{step_2.output_state.get('salary_deduction'):,.2f} SAR</b><br>
                Final Monthly Net: <b>{step_2.output_state.get('final_monthly_net'):,.2f} SAR</b> (Base: 12,000 SAR)
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; margin-top: 14px; font-size: 0.78rem; color: #475569;">
                <b>The Silent Failure Dilemma:</b> Every individual agent reported execution <code>SUCCESS</code>. 
                Downstream agents accurately processed flawed upstream inputs. Standard status checks pass, while business data is severely corrupted.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------ RIGHT COLUMN: RELIABILITY INSPECTOR ------------------
    with right_col:
        st.markdown("""
        <div class="h-card">
            <div class="h-card-header">
                <div>
                    <div class="h-card-title">Agent Reliability Inspector</div>
                    <div class="h-card-subtitle">Autonomous Diagnostic Loop & Counterfactual Verification</div>
                </div>
                <span class="badge badge-teal">SILSILA Engine</span>
            </div>
        """, unsafe_allow_html=True)

        if alert.detected:
            st.markdown(f"""
            <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 12px 14px; margin-bottom: 16px;">
                <div style="font-weight: 700; font-size: 0.85rem; color: #DC2626; margin-bottom: 2px;">
                    Silent Invariant Violation Detected
                </div>
                <div style="font-size: 0.80rem; color: #7F1D1D;">
                    <b>{alert.invariant_violated}:</b> {alert.description}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Autonomous Tool-Calling Timeline
            st.markdown('<div style="font-size: 0.78rem; font-weight: 700; color: #101B30; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 12px;">Diagnostic Investigation Timeline</div>', unsafe_allow_html=True)
            
            if report.investigation_timeline:
                for idx, t_step in enumerate(report.investigation_timeline):
                    st.markdown(f"""
                    <div class="react-item">
                        <div class="react-dot"></div>
                        <div class="react-header">Iteration {t_step.step_number}: Invoked Tool <code>{t_step.tool_selected}</code></div>
                        <div class="react-thought">
                            <b>Hypothesis & Analysis:</b><br>{t_step.thought}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander(f"Inspect Iteration {t_step.step_number} Arguments & Observation", expanded=False):
                        st.json({
                            "Tool": t_step.tool_selected,
                            "Arguments": t_step.tool_arguments,
                            "Observation": t_step.observation,
                        })

                st.markdown(f"""
                <div style="background: #FFFFFF; border: 1px solid #E4EBEF; border-left: 4px solid #DC2626; border-radius: 6px; padding: 12px 16px; margin: 12px 0;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #101B30;">
                        Attribution Verdict: <span class="badge badge-danger">{report.faulty_agent}</span> at Step {report.faulty_step}
                    </div>
                    <div style="font-size: 0.78rem; color: #475569; margin-top: 4px;">
                        <b>Causal Evidence:</b> {report.evidence}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption("Rule-based baseline mode active. Direct procedural causal DAG traversal executed.")
                st.markdown(f"**Diagnosed Agent:** `{report.faulty_agent}` (Step {report.faulty_step})")
                st.markdown(f"**Evidence:** {report.evidence}")

            # Safe Isolated State Recovery
            if recovery and recovery.success:
                st.markdown('<div style="font-size: 0.78rem; font-weight: 700; color: #101B30; text-transform: uppercase; letter-spacing: 0.04em; margin: 16px 0 10px 0;">Isolated State Recovery & Verification</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="font-size: 0.78rem; color: #475569; margin-bottom: 8px;">
                    Minimal Causal State Patch at <b>Step {recovery.forked_step}</b>: <code>{json.dumps(recovery.applied_patch)}</code>
                </div>
                """, unsafe_allow_html=True)

                # State Diff Table
                rec_state = recovery.final_recovered_state
                orig_out = trace.final_output
                
                st.markdown(f"""
                <table class="diff-table">
                    <thead>
                        <tr>
                            <th>State Attribute</th>
                            <th>Corrupted Workflow Output</th>
                            <th>Healed Sandbox Replay</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Requested Days</td>
                            <td class="val-corrupt">{orig_out.get('requested_days')} days</td>
                            <td class="val-recovered">{rec_state.get('requested_days')} days</td>
                        </tr>
                        <tr>
                            <td>Remaining Balance</td>
                            <td class="val-corrupt">{orig_out.get('remaining_balance')} days</td>
                            <td class="val-recovered">{rec_state.get('remaining_balance')} days</td>
                        </tr>
                        <tr>
                            <td>Salary Deduction</td>
                            <td class="val-corrupt">{orig_out.get('salary_deduction', 0):,.2f} SAR</td>
                            <td class="val-recovered">{rec_state.get('salary_deduction', 0):,.2f} SAR</td>
                        </tr>
                        <tr>
                            <td>Final Monthly Net</td>
                            <td class="val-corrupt">{orig_out.get('final_monthly_net', 0):,.2f} SAR</td>
                            <td class="val-recovered">{rec_state.get('final_monthly_net', 0):,.2f} SAR</td>
                        </tr>
                    </tbody>
                </table>
                """, unsafe_allow_html=True)

                st.write("")
                if verified and verified.verification_passed:
                    st.markdown(f"""
                    <div style="background: #E8FAF6; border: 1px solid #B4F0E4; border-radius: 6px; padding: 10px 14px;">
                        <div style="font-size: 0.80rem; font-weight: 700; color: #0CA58D;">
                            Independent Recovery Verification Passed
                        </div>
                        <div style="font-size: 0.74rem; color: #065F46; margin-top: 2px;">
                            {verified.verification_notes}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 6px; padding: 10px 14px;">
                        <div style="font-size: 0.80rem; font-weight: 700; color: #DC2626;">
                            Independent Recovery Verification Rejected
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #E8FAF6; border: 1px solid #B4F0E4; border-radius: 8px; padding: 16px; text-align: center;">
                <div style="font-weight: 700; font-size: 0.95rem; color: #0CA58D; margin-bottom: 4px;">
                    Clean Multi-Agent Execution Verified
                </div>
                <div style="font-size: 0.80rem; color: #065F46;">
                    All semantic invariants, quota policies, and payroll balances passed without violation.
                    No root cause investigation or state recovery required.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ====================================================
# VIEW 2: INVESTIGATION INSPECTOR (Deep-Dive Analysis)
# ====================================================
elif selected_view == "Investigation Inspector":
    st.markdown("""
    <div class="h-card">
        <div class="h-card-header">
            <div>
                <div class="h-card-title">Full Multi-Agent Execution Trace & Contracts</div>
                <div class="h-card-subtitle">Comprehensive JSON inspection of inputs, outputs, timestamps, and contract enforcement</div>
            </div>
            <span class="badge badge-teal">Workflow ID: {trace.workflow_id}</span>
        </div>
    """.format(trace=trace), unsafe_allow_html=True)

    trace_summary_rows = []
    for s in trace.steps:
        is_fault = alert.detected and s.step_index == report.faulty_step
        status_tag = "ROOT CAUSE" if is_fault else ("CLEAN" if not alert.detected else "CASCADE CORRUPTED")
        trace_summary_rows.append({
            "Step Index": s.step_index,
            "Agent Name": s.agent_name,
            "Reported Status": s.status,
            "Diagnostic Tag": status_tag,
            "Timestamp": s.timestamp.split("T")[-1][:8] if "T" in s.timestamp else s.timestamp,
            "Output Payload": json.dumps(s.output_state),
        })
    st.dataframe(trace_summary_rows, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Step-by-Step Contract & State Inspection")

    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.markdown("**Step 0: IntakeAgent**")
        st.json({"input": step_0.input_state, "output": step_0.output_state})
    with t_col2:
        st.markdown("**Step 1: PolicyAgent**")
        st.json({"input": step_1.input_state, "output": step_1.output_state})
    with t_col3:
        st.markdown("**Step 2: PayrollAgent**")
        st.json({"input": step_2.input_state, "output": step_2.output_state})

    st.markdown('</div>', unsafe_allow_html=True)


# ====================================================
# VIEW 3: EVALUATION LAB (HUMAIN Style Benchmark)
# ====================================================
elif selected_view == "Evaluation Lab":
    st.markdown("""
    <div class="h-card">
        <div class="h-card-header">
            <div>
                <div class="h-card-title">Quantitative Evaluation Lab</div>
                <div class="h-card-subtitle">Empirical benchmark across 7 held-out Arabic enterprise workflow scenarios</div>
            </div>
            <span class="badge badge-teal">Benchmark Suite v1.0</span>
        </div>
    """, unsafe_allow_html=True)

    run_eval_btn = st.button("Execute Full Benchmark Suite", type="primary")

    if run_eval_btn or "cached_bench" not in st.session_state:
        with st.spinner("Running quantitative benchmark suite across all scenarios..."):
            runner = BenchmarkRunner()
            bench_data = runner.run_all()
            st.session_state.cached_bench = bench_data

    bench = st.session_state.cached_bench

    # HUMAIN Style Metric Cards
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Detection Recall</div>
            <div class="kpi-value">{bench['detection_recall']*100:.1f}%</div>
            <div class="kpi-subtext">True positive rate</div>
        </div>
        """, unsafe_allow_html=True)

    with b2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Detection Precision</div>
            <div class="kpi-value">{bench['detection_precision']*100:.1f}%</div>
            <div class="kpi-subtext">Positive predictive value</div>
        </div>
        """, unsafe_allow_html=True)

    with b3:
        clean_denom = max(1, bench['clean_scenarios'])
        fpr = (bench['false_positives'] / clean_denom) * 100
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">False Positive Rate</div>
            <div class="kpi-value">{fpr:.1f}%</div>
            <div class="kpi-subtext">Clean scenario alerts</div>
        </div>
        """, unsafe_allow_html=True)

    with b4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Attribution Accuracy</div>
            <div class="kpi-value">{bench['attribution_accuracy']*100:.1f}%</div>
            <div class="kpi-subtext">Decisive agent pinpointed</div>
        </div>
        """, unsafe_allow_html=True)

    with b5:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">Recovery Verification</div>
            <div class="kpi-value">{bench['recovery_verification_rate']*100:.1f}%</div>
            <div class="kpi-subtext">Verified safe state</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Comparative Architecture Breakdown (HUMAIN Styled)
    st.markdown("#### Comparative Architecture Breakdown")
    cmp_left, cmp_right = st.columns(2)

    with cmp_left:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid #E4EBEF; border-radius:8px; padding:18px;">
            <div style="font-weight:700; font-size:0.90rem; color:#101B30; margin-bottom:8px;">
                Rule-Based Baseline (Procedural DAG)
            </div>
            <div style="font-size:0.80rem; color:#475569; line-height:1.6;">
                • <b>Mechanism:</b> Hardcoded sequential if/else checks across predefined trace schema.<br>
                • <b>Latency:</b> Fast (&lt; 5 ms), near zero compute overhead.<br>
                • <b>Tool Calling:</b> None. Traverses static branches without diagnostic evidence collection.<br>
                • <b>Adaptability:</b> Zero. Rigidly fails on novel Arabic dialectal phrasing or unexpected fault types.<br>
                • <b>Enterprise Feasibility:</b> Brittle in open-world multi-agent deployments.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with cmp_right:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid #E4EBEF; border-radius:8px; padding:18px;">
            <div style="font-weight:700; font-size:0.90rem; color:#101B30; margin-bottom:8px;">
                SILSILA Autonomous Investigator (Ollama qwen2.5:3b)
            </div>
            <div style="font-size:0.80rem; color:#475569; line-height:1.6;">
                • <b>Mechanism:</b> Dynamic ReAct tool-calling loop reasoning over trace and business invariants.<br>
                • <b>Latency:</b> 800 - 3,200 ms per autonomous iteration.<br>
                • <b>Tool Calling:</b> Selects from 5 validated diagnostic tools (semantic audit, contract check, counterfactual).<br>
                • <b>Adaptability:</b> High. Robust to dialectal variations, multi-hop errors, and cascading failures.<br>
                • <b>Safety:</b> Bounded loop (max 5 calls), timeout protection, and sandbox-isolated recovery.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Benchmark Scenario Details")
    bench_table_rows = []
    for sc in bench["scenario_results"]:
        bench_table_rows.append({
            "Scenario Name": sc["scenario_name"],
            "Target Stage": sc["fault_stage"].upper(),
            "Detected": "YES" if sc["detected"] else "NO",
            "Attributed Agent": sc["attributed_agent"],
            "Attributed Step": f"Step {sc['attributed_step']}" if sc["attributed_step"] >= 0 else "N/A",
            "Recovery Verified": "PASSED" if sc["recovery_verified"] else "REJECTED",
        })
    st.dataframe(bench_table_rows, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ====================================================
# VIEW 4: ARCHITECTURE & RESEARCH
# ====================================================
else:
    st.markdown("""
    <div class="h-card">
        <div class="h-card-header">
            <div>
                <div class="h-card-title">Closed-Loop Autonomous Recovery Architecture</div>
                <div class="h-card-subtitle">Theoretical design, causal tracing, and open-source foundations</div>
            </div>
            <span class="badge badge-mint">Challenge 1 Architecture</span>
        </div>
    """, unsafe_allow_html=True)

    # 4-Phase System Diagram
    st.markdown("""
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 24px;">
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-top: 3px solid #008C9E; border-radius:8px; padding:14px;">
            <div style="font-size:0.75rem; font-weight:700; color:#008C9E; text-transform:uppercase;">Phase 1</div>
            <div style="font-size:0.88rem; font-weight:700; color:#101B30; margin:4px 0;">Causal Tracing</div>
            <div style="font-size:0.76rem; color:#64748B; line-height:1.4;">
                Immutable recording of agent inputs, outputs, timestamps, and stage handoff contracts across the execution DAG.
            </div>
        </div>
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-top: 3px solid #008C9E; border-radius:8px; padding:14px;">
            <div style="font-size:0.75rem; font-weight:700; color:#008C9E; text-transform:uppercase;">Phase 2</div>
            <div style="font-size:0.88rem; font-weight:700; color:#101B30; margin:4px 0;">Invariant Detection</div>
            <div style="font-size:0.76rem; color:#64748B; line-height:1.4;">
                Zero-shot audit against business invariants: Arabic quantity consistency, quota conservation, and salary balance.
            </div>
        </div>
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-top: 3px solid #008C9E; border-radius:8px; padding:14px;">
            <div style="font-size:0.75rem; font-weight:700; color:#008C9E; text-transform:uppercase;">Phase 3</div>
            <div style="font-size:0.88rem; font-weight:700; color:#101B30; margin:4px 0;">Agentic Investigation</div>
            <div style="font-size:0.76rem; color:#64748B; line-height:1.4;">
                Local Ollama LLM autonomously selects diagnostic tools to isolate the earliest decisive causal failure.
            </div>
        </div>
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-top: 3px solid #0CA58D; border-radius:8px; padding:14px;">
            <div style="font-size:0.75rem; font-weight:700; color:#0CA58D; text-transform:uppercase;">Phase 4</div>
            <div style="font-size:0.88rem; font-weight:700; color:#101B30; margin:4px 0;">Verified Recovery</div>
            <div style="font-size:0.76rem; color:#64748B; line-height:1.4;">
                Isolated sandbox forks execution at faulty step, applies minimal state patch, replays downstream stages, and verifies invariants.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Open-Source Foundations & Academic Citations")

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid #E4EBEF; border-radius:8px; padding:16px;">
            <div style="font-weight:700; font-size:0.90rem; color:#101B30;">1. Blamr</div>
            <div style="font-size:0.74rem; color:#008C9E; margin-bottom:8px;">prithvirajualluri/blamr (MIT License)</div>
            <div style="font-size:0.78rem; color:#475569; line-height:1.5;">
                Inspiration for causal DAG execution tracing, confidence gating, and local LLM-powered failure explanation. SILSILA extends this into multi-tool autonomous ReAct diagnostics and multilingual Arabic support.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid #E4EBEF; border-radius:8px; padding:16px;">
            <div style="font-weight:700; font-size:0.90rem; color:#101B30;">2. Who&When</div>
            <div style="font-size:0.74rem; color:#008C9E; margin-bottom:8px;">ICML 2025 Spotlight (Zhang et al.)</div>
            <div style="font-size:0.78rem; color:#475569; line-height:1.5;">
                Theoretical foundation for decisive step-level error localization in collaborative agent graphs. SILSILA incorporates the principle of isolating the earliest causal distortion rather than symptom-bearing downstream steps.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid #E4EBEF; border-radius:8px; padding:16px;">
            <div style="font-weight:700; font-size:0.90rem; color:#101B30;">3. Rewind / TraceElephant</div>
            <div style="font-size:0.74rem; color:#008C9E; margin-bottom:8px;">agentoptics/rewind</div>
            <div style="font-size:0.78rem; color:#475569; line-height:1.5;">
                Architectural principles of counterfactual execution branching, state isolation, and deterministic replay recovery without mutating active production environments.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
