"""Zero Hunger Smart Canteen — Food Waste Optimization & Sequential LLM Prompt Pipeline.

A multi-page Streamlit application powered by Qwen2.5 7B (via local Ollama) and Python Analytics.
Includes both the Analytics Dashboard and the 5-Stage Sequential Prompt Engineering Pipeline.
"""

import io
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import importlib
import src.analytics
import src.data_loader
import src.ollama_client
import src.pipeline_engine
import src.prompt_manager

# Ensure all submodules are freshly reloaded in Streamlit's long-lived Python daemon
importlib.reload(src.analytics)
importlib.reload(src.data_loader)
importlib.reload(src.ollama_client)
importlib.reload(src.pipeline_engine)
importlib.reload(src.prompt_manager)

from src.analytics import (
    calculate_demand_baseline,
    calculate_menu_performance,
    calculate_summary_metrics,
    calculate_waste_by_attribute,
    get_daily_waste_trend,
)
from src.data_loader import (
    REQUIRED_COLUMNS,
    filter_dataset,
    load_and_validate_data,
    load_uploaded_csv,
)
from src.ollama_client import OllamaClient
from src.pipeline_engine import (
    STAGE_METADATA,
    assemble_stage_user_prompt,
    can_run_stage,
    execute_pipeline_stage,
    generate_consolidated_report,
    initialize_pipeline_state,
    invalidate_downstream_stages,
    reset_pipeline_state,
)

# Load environment configuration
load_dotenv()

st.set_page_config(
    page_title="Zero Hunger Canteen | Prompt Engineering Pipeline",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Theme & Light-Mode Layout Overrides
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Critical light-mode overrides to prevent OS dark-mode clashing */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    [data-testid="stSidebar"], [data-testid="stSidebarContent"], section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        color: #0f172a !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    p, span, label, div {
        color: #334155;
    }
    
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1350px;
    }
    
    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 65%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a !important;
        margin: 0 0 4px 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #475569 !important;
        margin: 0;
        line-height: 1.5;
    }
    .sdg-badge {
        display: inline-flex;
        align-items: center;
        background-color: #dcfce7;
        color: #15803d !important;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 16px;
        margin-bottom: 6px;
        letter-spacing: 0.03em;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        height: 100%;
    }
    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b !important;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .kpi-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a !important;
        margin: 0 0 4px 0;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #059669 !important;
        font-weight: 500;
    }
    .kpi-subtext-warn {
        color: #e11d48 !important;
    }

    /* Pipeline Stage Cards */
    .stage-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px;
        padding: 14px 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        transition: all 0.15s ease;
        height: 100%;
    }
    .stage-card-active {
        border: 2px solid #10b981 !important;
        background-color: #f0fdf4 !important;
    }
    .stage-card-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0f172a !important;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .stage-card-desc {
        font-size: 0.76rem;
        color: #64748b !important;
        line-height: 1.35;
        margin-bottom: 8px;
        min-height: 38px;
    }
    
    /* Status chips */
    .chip-pending {
        background-color: #f1f5f9;
        color: #64748b;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 10px;
        display: inline-block;
    }
    .chip-running {
        background-color: #fef3c7;
        color: #b45309;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 10px;
        display: inline-block;
    }
    .chip-completed {
        background-color: #dcfce7;
        color: #15803d;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 10px;
        display: inline-block;
    }
    .chip-failed {
        background-color: #fee2e2;
        color: #b91c1c;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 10px;
        display: inline-block;
    }
    .chip-stale {
        background-color: #ffedd5;
        color: #c2410c;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 10px;
        display: inline-block;
    }

    /* Container cards */
    .light-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        margin-bottom: 16px;
    }

    /* Response container */
    .ai-response-box {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 4px solid #10b981 !important;
        border-radius: 10px;
        padding: 20px 22px;
        line-height: 1.65;
        font-size: 0.92rem;
        color: #1e293b !important;
    }
    
    /* Code and prompt preview panels */
    .prompt-box {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px;
        padding: 12px 14px;
        font-family: monospace;
        font-size: 0.82rem;
        white-space: pre-wrap;
        color: #0f172a !important;
        max-height: 350px;
        overflow-y: auto;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        background-color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = initialize_pipeline_state()

if "selected_stage_id" not in st.session_state:
    st.session_state.selected_stage_id = 1

if "dataset_signature" not in st.session_state:
    st.session_state.dataset_signature = None

# Initialize Ollama client
ollama_client = OllamaClient()


# ---------------------------------------------------------
# Sidebar Navigation & Dataset Configuration
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 4px 0 12px 0;">
        <span style="font-size: 1.3rem;">🌱</span>
        <span style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-left: 5px;">Zero Hunger</span>
        <div style="font-size: 0.78rem; color: #64748b;">Smart Canteen Optimization</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Navigation
app_mode = st.sidebar.radio(
    "Navigation View:",
    options=[
        "🔄 LLM Prompt Engineering Pipeline",
        "📊 Analytics Dashboard",
        "📖 Methodology & SDG 2",
    ],
    index=0,
)

st.sidebar.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# Dataset Loading / Upload Section
with st.sidebar.expander("📁 Dataset Source & Upload", expanded=True):
    data_source_mode = st.radio(
        "Choose Data Source:",
        options=["Sample Canteen Dataset (500 records)", "Upload Custom CSV File"],
        index=0,
    )

    df_active: Optional[pd.DataFrame] = None
    dataset_name = "synthetic_canteen_data_500.csv"
    data_load_error = None

    if data_source_mode == "Upload Custom CSV File":
        uploaded_file = st.file_uploader(
            "Upload Canteen CSV",
            type=["csv"],
            help="File must contain required columns (date, menu_item, meals_prepared, meals_sold, leftover_portions, etc.)",
        )
        if uploaded_file is not None:
            dataset_name = uploaded_file.name
            try:
                df_active = load_uploaded_csv(uploaded_file)
            except Exception as e:
                data_load_error = str(e)
        else:
            st.info("Please select and upload a CSV file.")
    else:
        try:
            df_active = load_and_validate_data()
        except Exception as e:
            data_load_error = str(e)

    # Invalidate pipeline if dataset changed
    if df_active is not None and not df_active.empty:
        current_sig = f"{dataset_name}_{len(df_active)}_{df_active['date'].min()}_{df_active['date'].max()}"
        if st.session_state.dataset_signature != current_sig:
            st.session_state.dataset_signature = current_sig
            reset_pipeline_state(st.session_state.pipeline_state)
            st.session_state.selected_stage_id = 1

    if data_load_error:
        st.error(f"⚠️ Dataset Error:\n{data_load_error}")
        st.stop()

    if df_active is not None:
        st.caption(f"Loaded **{len(df_active)}** records (`{dataset_name}`)")
        st.caption(f"Date: `{df_active['date'].min()}` to `{df_active['date'].max()}`")

# Local Ollama Health Monitor
with st.sidebar.expander("🤖 Local Model Status", expanded=False):
    is_connected, status_msg, available_models = ollama_client.check_health()
    if is_connected:
        st.markdown(
            f'<div style="background:#ecfdf5; color:#047857; padding:4px 10px; border-radius:12px; font-weight:600; font-size:0.8rem;">● Online ({ollama_client.model})</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Endpoint: `{ollama_client.base_url}`")
    else:
        st.markdown(
            '<div style="background:#fff1f2; color:#be123c; padding:4px 10px; border-radius:12px; font-weight:600; font-size:0.8rem;">● Offline / Missing</div>',
            unsafe_allow_html=True,
        )
        st.caption(status_msg)
        st.code(f"ollama run {ollama_client.model}", language="bash")

    if st.button("🔄 Check Model Connection", key="check_ollama_btn", use_container_width=True):
        st.rerun()

# Model Execution & Performance Tuning (Crucial for CPU runtimes)
with st.sidebar.expander("⚡ Model Performance & Speed", expanded=False):
    ui_num_predict = st.select_slider(
        "Max Response Tokens",
        options=[150, 200, 250, 350],
        value=200,
        help="200 tokens generates in ~90-120s on CPU without hitting timeouts, giving concise structured findings.",
    )
    ui_timeout = st.slider(
        "Timeout Limit (seconds)",
        min_value=120,
        max_value=600,
        value=360,
        step=30,
        help="Maximum time to wait before timing out (default 360s / 6 min).",
    )
    st.caption("ℹ️ *Running locally on CPU. Lower token limits keep inference responsive and prevent timeouts.*")

st.sidebar.markdown("---")
st.sidebar.caption("Pure Python arithmetic + local Qwen2.5 7B reasoning.")


# ===========================================================================
# VIEW 1: LLM PROMPT ENGINEERING PIPELINE (Main Objective)
# ===========================================================================
if app_mode == "🔄 LLM Prompt Engineering Pipeline":

    # Header Banner
    st.markdown(
        """
        <div class="hero-banner">
            <div class="sdg-badge">🔄 5-STAGE SEQUENTIAL PIPELINE • UN SDG 2 & 12.3</div>
            <h1 class="hero-title">LLM Prompt Engineering Pipeline</h1>
            <p class="hero-subtitle">
                A true sequential LLM chain. <strong>Python computes all numerical metrics</strong>, and 
                <strong>each stage feeds its verified reasoning output directly into the next stage's prompt</strong>.
                Inspect inputs, review exact assembled prompts, edit instructions, and observe deterministic data flow.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_active is None or df_active.empty:
        st.warning("⚠️ Please provide a valid dataset to execute the pipeline.")
        st.stop()

    # -------------------------------------------------------------
    # 1. Pipeline Overview: 5 Connected Stage Cards
    # -------------------------------------------------------------
    st.markdown("<h4 style='margin:0 0 10px 0; font-size:1.05rem;'>Pipeline Progress Sequence</h4>", unsafe_allow_html=True)
    
    cols = st.columns(5)
    for idx, s_id in enumerate(range(1, 6)):
        stage_info = st.session_state.pipeline_state[s_id]
        status = stage_info["status"]
        duration = stage_info["duration_seconds"]
        is_selected = (st.session_state.selected_stage_id == s_id)

        # Chip class
        chip_cls = f"chip-{status}"
        status_text = status.upper()
        if duration > 0 and status == "completed":
            time_badge = f"⏱️ {duration}s"
        else:
            time_badge = ""

        with cols[idx]:
            card_border_cls = "stage-card-active" if is_selected else "stage-card"
            st.markdown(
                f"""
                <div class="{card_border_cls}">
                    <div class="stage-card-title">
                        <span>{stage_info['icon']}</span>
                        <span>Stage {s_id}</span>
                    </div>
                    <div style="font-size: 0.8rem; font-weight: 600; color: #1e293b; margin-bottom: 3px;">
                        {stage_info['short_name']}
                    </div>
                    <div class="stage-card-desc">
                        {stage_info['objective'][:75]}...
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="{chip_cls}">{status_text}</span>
                        <span style="font-size: 0.72rem; color: #64748b;">{time_badge}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Inspect Stage {s_id}", key=f"select_stage_{s_id}", use_container_width=True):
                st.session_state.selected_stage_id = s_id
                st.rerun()

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. Pipeline Execution Action Bar (Step-by-Step & Full Run)
    # -------------------------------------------------------------
    st.markdown("<div class='light-card'>", unsafe_allow_html=True)
    c_exec1, c_exec2, c_exec3, c_exec4 = st.columns([1.3, 1.3, 1.4, 1.2])

    current_id = st.session_state.selected_stage_id
    current_stage = st.session_state.pipeline_state[current_id]
    can_run_curr, run_reason = can_run_stage(current_id, st.session_state.pipeline_state)

    with c_exec1:
        btn_label = f"▶️ Run Stage {current_id}" if current_stage["status"] in ["pending", "failed"] else f"🔄 Rerun Stage {current_id}"
        if st.button(btn_label, type="primary", use_container_width=True, disabled=not can_run_curr):
            if not is_connected:
                st.error("❌ Ollama is offline. Please start Ollama before executing.")
            else:
                with st.spinner(f"Executing Stage {current_id} ({current_stage['short_name']}) via Qwen2.5 7B (~{ui_num_predict} tokens on CPU)..."):
                    execute_pipeline_stage(
                        stage_id=current_id,
                        df=df_active,
                        pipeline_state=st.session_state.pipeline_state,
                        ollama_client=ollama_client,
                        num_predict=ui_num_predict,
                        timeout=ui_timeout,
                    )
                st.rerun()

    with c_exec2:
        # Next stage shortcut
        next_id = current_id + 1 if current_id < 5 else None
        if next_id:
            can_run_next, _ = can_run_stage(next_id, st.session_state.pipeline_state)
            if st.button(f"⏭️ Advance to Stage {next_id}", use_container_width=True, disabled=not can_run_next):
                st.session_state.selected_stage_id = next_id
                st.rerun()
        else:
            st.button("🏁 End of Pipeline", disabled=True, use_container_width=True)

    with c_exec3:
        # Mode B: Run Full Pipeline
        if st.button("⚡ Run Full Pipeline (Stages 1-5)", use_container_width=True):
            if not is_connected:
                st.error("❌ Ollama is offline. Please start Ollama.")
            else:
                progress_bar = st.progress(0, text="Starting 5-stage sequential pipeline...")
                halted = False
                for step_idx, s in enumerate(range(1, 6)):
                    progress_bar.progress(
                        (step_idx) / 5.0,
                        text=f"Executing Stage {s} of 5: {STAGE_METADATA[s]['short_name']} (~{ui_num_predict} tokens)...",
                    )
                    st.session_state.selected_stage_id = s
                    result = execute_pipeline_stage(
                        stage_id=s,
                        df=df_active,
                        pipeline_state=st.session_state.pipeline_state,
                        ollama_client=ollama_client,
                        num_predict=ui_num_predict,
                        timeout=ui_timeout,
                    )
                    if result["status"] == "failed":
                        progress_bar.empty()
                        st.error(f"❌ Pipeline halted at Stage {s}: {result['error']}")
                        halted = True
                        break

                if not halted:
                    progress_bar.progress(1.0, text="✅ All 5 stages completed successfully!")
                st.rerun()

    with c_exec4:
        if st.button("🗑️ Reset Pipeline", use_container_width=True, help="Clear all outputs and reset stages to Pending"):
            reset_pipeline_state(st.session_state.pipeline_state)
            st.session_state.selected_stage_id = 1
            st.rerun()

    # Informative dependency notes
    if not can_run_curr and current_stage["status"] == "pending":
        st.info(f"🔒 **Dependency Notice**: {run_reason}")
    elif current_stage["status"] == "stale":
        st.warning(f"⚠️ **Stale Output Notice**: {current_stage.get('stale_reason', 'Upstream dependency was rerun.')}")
    elif current_stage["status"] == "failed":
        st.error(f"❌ **Stage Error**: {current_stage.get('error')}")

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3. Detailed Stage Inspector (Tabs: Prompt, Inputs, Output, Meta)
    # -------------------------------------------------------------
    st.markdown(
        f"<h3 style='font-size:1.25rem; color:#0f172a; margin:14px 0 10px 0;'>"
        f"{current_stage['icon']} Stage {current_id} Inspector: {current_stage['name']}"
        f"</h3>",
        unsafe_allow_html=True,
    )

    t_prompt, t_inputs, t_output, t_meta = st.tabs(
        ["✏️ Assembled Prompt & Editor", "📥 Input Data & Preceding Outputs", "📤 LLM Response (Qwen2.5 7B)", "⚙️ Stage Trace & Metadata"]
    )

    # ---------------- TAB 1: PROMPT & EDITOR ----------------
    with t_prompt:
        st.markdown(
            """
            <div style="font-size: 0.86rem; color: #475569; margin-bottom: 12px;">
                Inspect the exact user prompt assembled for this stage. You can edit the prompt below to test how 
                prompt adjustments change model behavior. Changes are preserved in your session.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Assemble prompt preview
        default_prompt, input_dict, prev_used = assemble_stage_user_prompt(
            stage_id=current_id,
            df=df_active,
            pipeline_state=st.session_state.pipeline_state,
        )

        active_prompt_text = current_stage["custom_user_prompt"] or default_prompt

        c_badge, c_reset = st.columns([3, 1])
        with c_badge:
            if current_stage["is_custom"]:
                st.markdown('<span style="background:#fef3c7; color:#b45309; padding:3px 9px; border-radius:12px; font-size:0.78rem; font-weight:600;">⚠️ Custom Prompt in Use</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="background:#f1f5f9; color:#475569; padding:3px 9px; border-radius:12px; font-size:0.78rem; font-weight:600;">✓ Default Assembled Prompt</span>', unsafe_allow_html=True)
        with c_reset:
            if current_stage["is_custom"]:
                if st.button("Reset to Default Prompt", key=f"reset_prompt_{current_id}", use_container_width=True):
                    current_stage["custom_user_prompt"] = None
                    current_stage["is_custom"] = False
                    st.rerun()

        edited_prompt = st.text_area(
            "User Prompt to be sent to Ollama:",
            value=active_prompt_text,
            height=320,
            key=f"prompt_editor_{current_id}",
            help="You can edit this prompt and run this stage with your modified prompt text.",
        )

        if edited_prompt != active_prompt_text:
            current_stage["custom_user_prompt"] = edited_prompt
            current_stage["is_custom"] = True
            st.caption("✏️ Custom changes saved to current session.")

        with st.expander("System Instruction Provided to Ollama"):
            st.code(current_stage["system_prompt"], language="text")

    # ---------------- TAB 2: INPUT DATA & PRECEDING OUTPUTS ----------------
    with t_inputs:
        st.markdown("<h4 style='font-size:0.95rem; margin-bottom:6px;'>1. Deterministic Python Metrics Supplied:</h4>", unsafe_allow_html=True)
        
        # Display summary input data table
        if current_stage.get("inputs_summary"):
            input_items = current_stage["inputs_summary"]
            df_inputs = pd.DataFrame(
                [{"Metric / Feature": k, "Value Provided": str(v)} for k, v in input_items.items()]
            )
            st.dataframe(df_inputs, use_container_width=True, hide_index=True)
        else:
            st.caption("No custom metric table available.")

        st.markdown("<h4 style='font-size:0.95rem; margin-top:14px; margin-bottom:6px;'>2. Outputs from Preceding Stages Fed Into This Stage:</h4>", unsafe_allow_html=True)

        req_upstream = current_stage["previous_stages_required"]
        if not req_upstream:
            st.info("ℹ️ Stage 1 is the initial stage and receives only verified Python dataset metrics.")
        else:
            for up_id in req_upstream:
                up_stage = st.session_state.pipeline_state[up_id]
                up_status = up_stage["status"]
                with st.expander(f"📦 Preceding Output: Stage {up_id} ({up_stage['short_name']}) — Status: {up_status.upper()}", expanded=True):
                    if up_stage["raw_output"]:
                        st.markdown(
                            f"<div class='prompt-box'>{up_stage['raw_output']}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption(f"No output yet from Stage {up_id} (Status: {up_status}).")

    # ---------------- TAB 3: LLM RESPONSE ----------------
    with t_output:
        if current_stage["status"] == "completed":
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <span style="font-weight:700; font-size:1.05rem; color:#0f172a;">Generated Response from Qwen2.5 7B</span>
                    <span style="font-size:0.8rem; background:#f1f5f9; padding:4px 10px; border-radius:14px; color:#475569;">
                        ⏱️ Duration: {current_stage['duration_seconds']}s • {current_stage['eval_count']} tokens • {current_stage.get('timestamp', '')}
                    </span>
                </div>
                <div class="ai-response-box">
                {current_stage['raw_output']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif current_stage["status"] == "stale":
            st.warning("⚠️ **Stale Output**: An upstream stage was rerun after this output was generated. Please rerun this stage to refresh.")
            st.markdown(f"<div class='ai-response-box'>{current_stage['raw_output']}</div>", unsafe_allow_html=True)
        elif current_stage["status"] == "running":
            st.info("⏳ Stage is currently executing locally with Qwen2.5 7B...")
        elif current_stage["status"] == "failed":
            st.error(f"❌ Execution failed: {current_stage.get('error')}")
            if current_stage.get("raw_output"):
                st.markdown(f"<div class='ai-response-box'>{current_stage['raw_output']}</div>", unsafe_allow_html=True)
        else:
            st.info(f"⏳ Stage {current_id} has not been executed yet. Click **▶️ Run Stage {current_id}** above.")

    # ---------------- TAB 4: STAGE METADATA ----------------
    with t_meta:
        meta_data = {
            "Stage ID": current_id,
            "Stage Name": current_stage["name"],
            "Execution Status": current_stage["status"].upper(),
            "Duration (seconds)": f"{current_stage['duration_seconds']}s",
            "Tokens Evaluated": current_stage["eval_count"],
            "Timestamp": current_stage.get("timestamp") or "N/A",
            "Model Used": ollama_client.model,
            "Endpoint": ollama_client.base_url,
            "Custom Prompt Active": "Yes" if current_stage["is_custom"] else "No",
            "Upstream Dependencies Required": str(current_stage["previous_stages_required"]),
            "Error State": str(current_stage.get("error") or "None"),
        }
        df_meta = pd.DataFrame([{"Property": k, "Value": str(v)} for k, v in meta_data.items()])
        st.dataframe(df_meta, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # 4. Final Consolidated Report (Shown when Stage 5 finishes)
    # -------------------------------------------------------------
    stage5 = st.session_state.pipeline_state[5]
    if stage5["status"] in ["completed", "stale"]:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#0f172a;'>📋 Final Consolidated Pipeline Report</h3>", unsafe_allow_html=True)
        st.caption("Consolidated synthesis incorporating Python verified metrics and outputs from all 5 completed stages.")

        final_report_md = generate_consolidated_report(df_active, st.session_state.pipeline_state)

        with st.expander("📄 View Full Consolidated Report", expanded=True):
            st.markdown(final_report_md)

        c_dl1, c_dl2 = st.columns([1, 1])
        with c_dl1:
            st.download_button(
                label="📥 Download Consolidated Report (.md)",
                data=final_report_md,
                file_name=f"canteen_pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with c_dl2:
            st.download_button(
                label="📥 Download Consolidated Report (.txt)",
                data=final_report_md,
                file_name=f"canteen_pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )


# ===========================================================================
# VIEW 2: ANALYTICS DASHBOARD (Preserving Existing Functionality)
# ===========================================================================
elif app_mode == "📊 Analytics Dashboard":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="sdg-badge">📊 CAMPUS DINING INTELLIGENCE</div>
            <h1 class="hero-title">Canteen Food Waste Analytics Dashboard</h1>
            <p class="hero-subtitle">
                Explore empirical canteen operational metrics, sales volume, leftovers, and waste drivers. 
                All arithmetic calculations are strictly computed in Python.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_active is None or df_active.empty:
        st.warning("⚠️ No records loaded. Please select or upload a dataset.")
        st.stop()

    # Sidebar Filters (Only relevant on Analytics Dashboard view)
    st.sidebar.markdown("**Dashboard Filters**")

    min_date = df_active["date"].min()
    max_date = df_active["date"].max()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_date, max_date

    all_items = sorted(df_active["menu_item"].unique())
    selected_items = st.sidebar.multiselect("Menu Items", options=all_items, default=all_items)

    all_weather = sorted(df_active["weather"].unique())
    selected_weather = st.sidebar.multiselect("Weather", options=all_weather, default=all_weather)

    all_exams = sorted(df_active["exam_period"].unique())
    selected_exams = st.sidebar.multiselect("Exam Period", options=all_exams, default=all_exams)

    all_events = sorted(df_active["college_event"].unique())
    selected_events = st.sidebar.multiselect("College Event", options=all_events, default=all_events)

    df_filtered = filter_dataset(
        df_active,
        start_date=start_d,
        end_date=end_d,
        menu_items=selected_items if selected_items else None,
        weather=selected_weather if selected_weather else None,
        exam_period=selected_exams if selected_exams else None,
        college_event=selected_events if selected_events else None,
    )

    st.sidebar.caption(f"Showing **{len(df_filtered)}** of **{len(df_active)}** records")

    if df_filtered.empty:
        st.warning("⚠️ No records match the selected filters. Please adjust the filters in the sidebar.")
    else:
        # Calculate summary metrics
        metrics = calculate_summary_metrics(df_filtered)

        # 4 KPI cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Meals Prepared</div>
                    <div class="kpi-val">{metrics['total_prepared']:,}</div>
                    <div class="kpi-subtext" style="color:#64748b;">Avg {metrics['avg_prepared_per_record']:.0f} per service</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Meals Sold</div>
                    <div class="kpi-val">{metrics['total_sold']:,}</div>
                    <div class="kpi-subtext">{(metrics['total_sold']/metrics['total_prepared']*100 if metrics['total_prepared']>0 else 0):.1f}% consumption rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Surplus Waste</div>
                    <div class="kpi-val">{metrics['total_waste_kg']:.1f} <span style="font-size:1rem; color:#64748b;">kg</span></div>
                    <div class="kpi-subtext kpi-subtext-warn">{metrics['total_leftovers']:,} portions ({metrics['waste_percentage']:.1f}% waste)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Realized Revenue</div>
                    <div class="kpi-val">₹{int(metrics['total_revenue_inr']):,}</div>
                    <div class="kpi-subtext" style="color:#64748b;">₹{int(metrics['estimated_waste_cost_inr']):,} unrecovered surplus</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Charts
        df_menu_perf = calculate_menu_performance(df_filtered)
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("<div class='light-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='margin:0 0 10px 0; font-size:1rem;'>🍱 Prepared vs. Sold by Menu Item</h4>", unsafe_allow_html=True)
            fig_prep_sold = go.Figure()
            fig_prep_sold.add_trace(go.Bar(name="Prepared", x=df_menu_perf["menu_item"], y=df_menu_perf["total_prepared"], marker=dict(color="#cbd5e1")))
            fig_prep_sold.add_trace(go.Bar(name="Sold", x=df_menu_perf["menu_item"], y=df_menu_perf["total_sold"], marker=dict(color="#10b981")))
            fig_prep_sold.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=40),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=False, tickangle=-25),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig_prep_sold, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_right:
            st.markdown("<div class='light-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='margin:0 0 10px 0; font-size:1rem;'>📉 Waste Percentage by Item</h4>", unsafe_allow_html=True)
            sorted_menu = df_menu_perf.sort_values(by="waste_percentage", ascending=True)
            fig_waste_pct = px.bar(
                sorted_menu,
                x="waste_percentage",
                y="menu_item",
                orientation="h",
                color="waste_percentage",
                color_continuous_scale=[[0.0, "#d1fae5"], [0.5, "#fed7aa"], [1.0, "#f87171"]],
                labels={"waste_percentage": "Waste %", "menu_item": ""},
            )
            fig_waste_pct.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=30),
                height=320,
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Waste %"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_waste_pct, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        col_trend, col_ops = st.columns(2)

        with col_trend:
            st.markdown("<div class='light-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='margin:0 0 10px 0; font-size:1rem;'>📅 Daily Waste Trend Over Time</h4>", unsafe_allow_html=True)
            daily_df = get_daily_waste_trend(df_filtered)
            fig_trend = px.area(daily_df, x="date", y="total_waste_kg", labels={"date": "Date", "total_waste_kg": "Waste (kg)"})
            fig_trend.update_traces(line_color="#10b981", fillcolor="rgba(16, 185, 129, 0.12)")
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=30),
                height=280,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_ops:
            st.markdown("<div class='light-card'>", unsafe_allow_html=True)
            c_h, c_s = st.columns([1, 1])
            with c_h:
                st.markdown("<h4 style='margin:0; font-size:1rem;'>🌦️ Waste by Operational Context</h4>", unsafe_allow_html=True)
            with c_s:
                ops_dim = st.selectbox("Dimension", options=["weather", "exam_period", "college_event"], label_visibility="collapsed")
            df_attr = calculate_waste_by_attribute(df_filtered, ops_dim)
            fig_ops = px.bar(df_attr, x=ops_dim, y="total_waste_kg", color_discrete_sequence=["#6366f1"], labels={"total_waste_kg": "Waste (kg)", ops_dim: ""})
            fig_ops.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=30),
                height=280,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig_ops, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Table & CSV export
        with st.expander("📋 View Filtered Records & Export CSV", expanded=False):
            st.dataframe(df_filtered, use_container_width=True, height=260)
            csv_data = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered Records as CSV",
                data=csv_data,
                file_name=f"canteen_filtered_{start_d}_{end_d}.csv",
                mime="text/csv",
            )


# ===========================================================================
# VIEW 3: METHODOLOGY & SDG 2 ZERO HUNGER
# ===========================================================================
elif app_mode == "📖 Methodology & SDG 2":

    st.markdown(
        """
        <div class="hero-banner">
            <div class="sdg-badge">🌍 UN SDG 2 & 12.3 METHODOLOGY</div>
            <h1 class="hero-title">Academic Methodology & SDG Framework</h1>
            <p class="hero-subtitle">
                Understanding the architectural principles, prompt engineering constraints, and sustainability goals 
                governing this intelligent canteen food waste optimization prototype.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        st.markdown(
            """
            <div class="kpi-card">
                <div style="font-size:1.4rem; margin-bottom:8px;">🔢</div>
                <div style="font-weight:700; color:#0f172a; margin-bottom:6px;">1. Strict Python Arithmetic</div>
                <div style="font-size:0.85rem; color:#64748b; line-height:1.5;">
                    Counts, sums, leftovers, waste percentages, and financial metrics are computed strictly in Python. 
                    The LLM never computes arithmetic, completely preventing mathematical hallucinations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_p2:
        st.markdown(
            """
            <div class="kpi-card">
                <div style="font-size:1.4rem; margin-bottom:8px;">🤖</div>
                <div style="font-weight:700; color:#0f172a; margin-bottom:6px;">2. Sequential Prompt Chain</div>
                <div style="font-size:0.85rem; color:#64748b; line-height:1.5;">
                    The 5-stage pipeline enforces a true dependency chain: Stage 1 diagnoses &rarr; Stage 2 plans prep &rarr; 
                    Stage 3 balances the menu &rarr; Stage 4 handles surplus &rarr; Stage 5 consolidates institutional impact.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_p3:
        st.markdown(
            """
            <div class="kpi-card">
                <div style="font-size:1.4rem; margin-bottom:8px;">🛡️</div>
                <div style="font-weight:700; color:#0f172a; margin-bottom:6px;">3. Food Safety Mandate</div>
                <div style="font-size:0.85rem; color:#64748b; line-height:1.5;">
                    Surplus handling follows the Food Recovery Hierarchy under strict hygiene controls. The LLM is 
                    forbidden from suggesting donation for food held in the danger zone (>2h).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    with st.expander("📊 Dataset Disclosure & Demonstration Boundaries", expanded=True):
        st.markdown(
            """
            The included dataset (`data/synthetic_canteen_data_500.csv`) contains 500 synthetic records simulating daily college 
            canteen operations (meals prepared, sold, leftover portions, estimated waste kg, revenue, weather, exams, events). 
            It provides an empirical testbed for evaluating kitchen demand fluctuations and prompt engineering stability.
            """
        )