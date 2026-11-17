# 🍲 Zero Hunger Smart Canteen: Sequential LLM Prompt Engineering Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/LLM-Qwen2.5%207B%20(Local)-orange.svg)](https://ollama.ai)
[![UN SDG](https://img.shields.io/badge/UN%20SDG-2%20(Zero%20Hunger)%20%26%2012.3-green.svg)](https://sdgs.un.org/goals)
[![Tests](https://img.shields.io/badge/pytest-39%20passed-brightgreen.svg)](tests/)

A locally running college-canteen food waste analytics and recommendation system. Powered by a **deterministic Python analytical core** paired with a **true 5-stage sequential LLM prompt-engineering chain using Qwen2.5 7B via Ollama**.

---

## 🎯 Project Overview & UN SDG Alignment

Institutional dining halls prepare large food batches where small estimation mismatches accumulate into significant preventable waste. This project targets two United Nations Sustainable Development Goals:
- **UN SDG 2 (Zero Hunger)**: Advancing food security by rescuing wholesome surplus portions and preventing edible food loss.
- **UN SDG 12.3 (Responsible Consumption and Production)**: Halving per-capita food waste through data-backed batch planning and source reduction.

### Key Architectural Principles
1. **Separation of Arithmetic and Reasoning**:
   - **Python / Pandas** computes all counts, sums, leftovers, waste percentages, and financial metrics with 100% mathematical precision.
   - The **Local LLM (Qwen2.5 7B)** is strictly provided with pre-calculated numbers and is instructed **never** to compute arithmetic or hallucinate numbers. It acts as an operational domain advisor diagnosing root causes, planning prep quantities, and recommending safe food handling.
2. **True Sequential Prompt Chain**:
   - The application is **not** just a static dashboard. It demonstrates **sequential prompt engineering** where each downstream stage strictly receives verified outputs from preceding stages.
3. **100% Local & Private**:
   - Runs locally via Ollama (`http://localhost:11434`). No cloud APIs, no subscription fees, and no data leaves your machine.

---

## 🔄 Sequential Pipeline Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │          CSV Upload / Sample Dataset         │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │  Python Data Validation & Metrics Analytics  │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  STAGE 1: Food Waste Analyst                                           │
    │  Input: Dataset summary, prepared vs sold, leftovers, waste %, weather │
    │  Output: Diagnoses waste patterns, high-waste dishes, hypotheses       │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ (Passes Stage 1 waste diagnoses)
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  STAGE 2: Demand Estimator                                             │
    │  Input: Python sales baselines (median, min-max) + Stage 1 Output      │
    │  Output: Recommends prep targets and staged cooking batches            │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ (Passes Stage 1 + Stage 2 outputs)
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  STAGE 3: Menu Optimizer                                               │
    │  Input: Menu portfolio table + Stage 1 Analysis + Stage 2 Demand Targets│
    │  Output: Recipe rotation, portion adjustments, ingredient synergies    │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ (Passes Stages 1, 2, 3 outputs)
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  STAGE 4: Surplus Food Manager                                         │
    │  Input: Leftover kg + Stages 1, 2, 3 Outputs + Food Safety Rules      │
    │  Output: Edibility check, safe holding protocols, donation windows     │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │ (Passes Stages 1, 2, 3, 4 outputs)
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  STAGE 5: Impact Analyst                                               │
    │  Input: Python verified totals + All 4 Preceding Stage Outputs         │
    │  Output: Consolidated synthesis, SDG 2 & 12.3 alignment, action plan  │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │  Final Consolidated Report (.md and .txt)    │
                 └──────────────────────────────────────────────┘
```

---

## 🧠 The 5 Sequential Stages in Detail

| Stage | Role & Objective | Inputs Consumed | Key Constraints & Outputs |
| :--- | :--- | :--- | :--- |
| **Stage 1: Food Waste Analyst** | Diagnose food waste patterns and overproduction hotspots. | Python dataset metrics, menu summary table, weather/exam context. | Separates empirical facts from hypotheses; identifies high-waste items; outputs briefing for Stage 2. |
| **Stage 2: Demand Estimator** | Recommend realistic prep quantities and batching strategies. | Python historical sales (median, min-max) + **Stage 1 output**. | Transparent baseline recommendation; no false ML claims; emphasizes staged cooking; feeds Stage 3. |
| **Stage 3: Menu Optimizer** | Practical menu changes balancing waste, demand, and variety. | Menu portfolio aggregates + **Stage 1 output** + **Stage 2 output**. | Avoids removing dishes on single observation; proposes ingredient synergies; feeds Stage 4. |
| **Stage 4: Surplus Food Manager** | Safe, timely handling of edible surplus. | Leftovers volume + **Stage 1, 2, 3 outputs** + ambient conditions. | **Strict safety mandate**: Never suggest donating danger-zone (>2h) food; Food Recovery Hierarchy; feeds Stage 5. |
| **Stage 5: Impact Analyst** | Executive consolidated synthesis. | Verified Python totals + **All 4 preceding stage outputs**. | No invented carbon/dollar multipliers; qualitative benefits clearly labelled; SDG 2 & 12.3 roadmap. |

---

## 🎮 Interactive Execution Modes

### Mode A: Step-by-Step Execution (Academic Demonstration)
1. **Dependency Unlocking**: Initially only Stage 1 is runnable. When Stage 1 completes, Stage 2 unlocks, and so forth through Stage 5.
2. **Contextual Action Buttons**:
   - `▶️ Run Stage X`: Executes the currently selected stage.
   - `🔄 Rerun Stage X`: Re-executes the stage with updated prompts or data.
   - `⏭️ Advance to Stage X+1`: Focuses on the next unlocked stage.
   - `🗑️ Reset Pipeline`: Clears all stage outputs without deleting data.
3. **Downstream Invalidation**: If you modify or rerun an earlier stage (e.g. Stage 2), all downstream stages (Stages 3, 4, 5) are automatically marked **STALE** with clear warnings explaining that upstream findings have changed.

### Mode B: Run Full Pipeline
- Click **⚡ Run Full Pipeline (Stages 1-5)**.
- Executes all 5 stages in order with a real-time progress bar.
- Halts cleanly if any stage fails, preserving all earlier successful stage outputs.

---

## 🔍 Prompt Engineering Transparency & Editing

Every stage provides a 4-tab inspection panel:
1. **✏️ Assembled Prompt & Editor**: View the exact system instructions and assembled user prompt. You can **edit the prompt directly in the UI** to test prompt variants! The UI tags custom prompts with a badge and provides a *Reset to Default Prompt* button.
2. **📥 Input Data & Preceding Outputs**: View the exact Python calculations and preceding stage outputs passed into the stage.
3. **📤 LLM Response**: View the real, un-simulated output from Qwen2.5 7B with execution duration (`⏱️ 14.2s`), token metrics, and timestamps.
4. **⚙️ Stage Trace & Metadata**: View execution status, model parameters, and dependency graphs.

---

## 📁 Dataset Upload & Schema Support

You can run the pipeline with the provided synthetic canteen dataset or upload your own CSV file via the sidebar.

### Required CSV Schema
The CSV should contain the following columns:
```csv
date,day_of_week,menu_item,price_inr,meals_prepared,meals_sold,leftover_portions,estimated_waste_kg,waste_percentage,revenue_inr,weather,exam_period,college_event
```
*(Common aliases such as `price` for `price_inr`, `sold` for `meals_sold`, and `leftover` for `leftover_portions` are automatically recognized).*

Uploading a new dataset automatically resets the pipeline state to ensure recommendations reflect the new data.

---

## 🚀 Setup & Execution Guide

### Step 1: Open PowerShell / Terminal
```powershell
cd "c:\sem 5\PE lab\zero-hunger-canteen"
```

### Step 2: Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Ensure Ollama is Running
```powershell
ollama run qwen2.5:7b
```
*(Verify endpoint at `http://localhost:11434`).*

### Step 4: Run the Streamlit Application
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧪 Automated Unit & Integration Tests

The project includes 39 automated unit and integration tests covering data loading, calculations, sequential prompt assembly, dependency unlocking, stale invalidation, and Ollama error handling:

```powershell
pytest -v
```

Output:
```text
tests/test_analytics.py .......                                          [ 17%]
tests/test_data_loader.py ........                                       [ 38%]
tests/test_ollama_client.py ........                                     [ 58%]
tests/test_pipeline_engine.py ........                                   [ 79%]
tests/test_prompt_manager.py ........                                    [100%]
============================== 39 passed in 0.55s ==============================
```

---

## 🛠️ Troubleshooting

1. **"Ollama Offline / Missing"**:
   - Ensure the Ollama service is active. Run `ollama serve` or open the Ollama desktop app.
   - Verify the model is downloaded: `ollama run qwen2.5:7b`.
2. **"Stage X is locked"**:
   - Upstream stages must be completed first. In Step-by-Step mode, execute stages in order (Stage 1 &rarr; Stage 2 &rarr; ...).
3. **"Stage X is stale"**:
   - An earlier stage was rerun. Simply click `🔄 Rerun Stage X` to refresh the output with the new upstream findings.
4. **"Timeout during generation"**:
   - The default timeout is set to 240 seconds (`OLLAMA_TIMEOUT_SECONDS=240` in `.env`). On CPU-only hardware, allow the local model sufficient time to complete inference.
