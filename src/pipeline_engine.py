"""Sequential LLM Prompt Engineering Pipeline Engine for Zero Hunger Smart Canteen.

Implements a true 5-stage sequential LLM execution chain where each downstream stage
strictly receives the verified outputs of preceding stages and deterministic Python metrics.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd

from src.analytics import (
    calculate_demand_baseline,
    calculate_menu_performance,
    calculate_summary_metrics,
    calculate_waste_by_attribute,
    format_menu_summary_table,
)
from src.ollama_client import OllamaClient

STAGE_METADATA: Dict[int, Dict[str, Any]] = {
    1: {
        "id": 1,
        "name": "Stage 1: Food Waste Analyst",
        "short_name": "Food Waste Analyst",
        "icon": "🔍",
        "objective": "Analyze the uploaded canteen dataset and diagnose primary food waste patterns, distinguishing empirical facts from hypotheses.",
        "previous_stages_required": [],
        "role": "Expert College Canteen Food Waste Analyst (UN SDG 2 & SDG 12.3)",
    },
    2: {
        "id": 2,
        "name": "Stage 2: Demand Estimator",
        "short_name": "Demand Estimator",
        "icon": "📈",
        "objective": "Synthesize Python historical sales baselines and Stage 1 waste diagnoses to recommend realistic preparation quantities and staged cooking.",
        "previous_stages_required": [1],
        "role": "Canteen Demand & Production Planning Specialist",
    },
    3: {
        "id": 3,
        "name": "Stage 3: Menu Optimizer",
        "short_name": "Menu Optimizer",
        "icon": "📋",
        "objective": "Generate practical menu rotation, portion sizing, and batch balancing adjustments using Stage 1 waste analysis and Stage 2 prep targets.",
        "previous_stages_required": [1, 2],
        "role": "Canteen Menu Engineer & Food Waste Reduction Strategist",
    },
    4: {
        "id": 4,
        "name": "Stage 4: Surplus Food Manager",
        "short_name": "Surplus Food Manager",
        "icon": "🍱",
        "objective": "Provide safe, timely handling and redistribution protocols for surplus food adhering strictly to food safety and the Food Recovery Hierarchy.",
        "previous_stages_required": [1, 2, 3],
        "role": "Campus Food Recovery & Safe Surplus Redistribution Coordinator",
    },
    5: {
        "id": 5,
        "name": "Stage 5: Impact Analyst",
        "short_name": "Impact Analyst",
        "icon": "🌱",
        "objective": "Consolidate Python-calculated metrics and findings across all 4 preceding stages into a unified Zero Hunger (SDG 2 & 12.3) implementation report.",
        "previous_stages_required": [1, 2, 3, 4],
        "role": "Lead Sustainability & Impact Assessment Analyst (UN SDG 2 & SDG 12.3)",
    },
}


def initialize_pipeline_state() -> Dict[int, Dict[str, Any]]:
    """Initialize a fresh session pipeline state for all 5 stages."""
    state: Dict[int, Dict[str, Any]] = {}
    for stage_id, meta in STAGE_METADATA.items():
        state[stage_id] = {
            "id": stage_id,
            "name": meta["name"],
            "short_name": meta["short_name"],
            "icon": meta["icon"],
            "objective": meta["objective"],
            "previous_stages_required": meta["previous_stages_required"],
            "status": "pending",  # "pending", "running", "completed", "failed", "stale"
            "status_message": "Ready to execute" if stage_id == 1 else f"Waiting for Stage {stage_id - 1}",
            "system_prompt": f"You are an {meta['role']}. Ground all reasoning strictly in the provided data. Do not fabricate numbers or claim unverified causation.",
            "default_user_prompt": "",
            "custom_user_prompt": None,
            "assembled_prompt": "",
            "is_custom": False,
            "raw_output": "",
            "duration_seconds": 0.0,
            "eval_count": 0,
            "error": None,
            "timestamp": None,
            "inputs_summary": {},
            "previous_stages_used": [],
            "stale_reason": None,
        }
    return state


def can_run_stage(stage_id: int, pipeline_state: Dict[int, Dict[str, Any]]) -> Tuple[bool, str]:
    """Check if a stage satisfies upstream dependencies and can be executed."""
    if stage_id == 1:
        return True, "Stage 1 has no upstream dependencies."

    required = STAGE_METADATA[stage_id]["previous_stages_required"]
    for req in required:
        req_status = pipeline_state[req]["status"]
        if req_status != "completed":
            return (
                False,
                f"Stage {stage_id} requires Stage {req} ({STAGE_METADATA[req]['short_name']}) to be completed. Current status: '{req_status}'.",
            )
    return True, "All required upstream stages are completed."


def invalidate_downstream_stages(stage_id: int, pipeline_state: Dict[int, Dict[str, Any]]) -> None:
    """Mark all stages downstream of stage_id as stale if they were previously completed or stale."""
    for s_id in range(stage_id + 1, 6):
        if pipeline_state[s_id]["status"] in ["completed", "stale"]:
            pipeline_state[s_id]["status"] = "stale"
            pipeline_state[s_id]["stale_reason"] = (
                f"Upstream Stage {stage_id} ({STAGE_METADATA[stage_id]['short_name']}) was modified or rerun. "
                f"Rerun Stage {s_id} to incorporate the updated upstream findings."
            )
            pipeline_state[s_id]["status_message"] = "Outdated by upstream rerun"


def reset_pipeline_state(pipeline_state: Dict[int, Dict[str, Any]], clear_custom_prompts: bool = False) -> None:
    """Reset all stages to pending and clear outputs."""
    for s_id in range(1, 6):
        pipeline_state[s_id]["status"] = "pending"
        pipeline_state[s_id]["status_message"] = "Ready to execute" if s_id == 1 else f"Waiting for Stage {s_id - 1}"
        pipeline_state[s_id]["raw_output"] = ""
        pipeline_state[s_id]["duration_seconds"] = 0.0
        pipeline_state[s_id]["eval_count"] = 0
        pipeline_state[s_id]["error"] = None
        pipeline_state[s_id]["timestamp"] = None
        pipeline_state[s_id]["stale_reason"] = None
        pipeline_state[s_id]["previous_stages_used"] = []
        pipeline_state[s_id]["assembled_prompt"] = ""
        if clear_custom_prompts:
            pipeline_state[s_id]["custom_user_prompt"] = None
            pipeline_state[s_id]["is_custom"] = False


# ---------------------------------------------------------------------------
# Sequential Prompt Assembly
# ---------------------------------------------------------------------------

def assemble_stage_user_prompt(
    stage_id: int,
    df: pd.DataFrame,
    pipeline_state: Dict[int, Dict[str, Any]],
    user_override_prompt: Optional[str] = None,
) -> Tuple[str, Dict[str, Any], List[int]]:
    """Assemble the exact user prompt for a given stage, injecting Python analytics and preceding outputs.
    
    Returns:
        (assembled_prompt: str, inputs_summary: dict, previous_stages_used: list[int])
    """
    if user_override_prompt is not None and user_override_prompt.strip():
        # User explicitly supplied custom prompt text
        return user_override_prompt.strip(), {"mode": "custom_override"}, []

    metrics = calculate_summary_metrics(df)
    menu_perf = calculate_menu_performance(df)
    menu_table = format_menu_summary_table(menu_perf, max_rows=10)

    # Summarize conditions
    weather_summary = ", ".join(f"{k}: {v}" for k, v in df["weather"].value_counts().items()) if not df.empty else "N/A"
    exam_summary = ", ".join(f"{k}: {v}" for k, v in df["exam_period"].value_counts().items()) if not df.empty else "N/A"
    event_summary = ", ".join(f"{k}: {v}" for k, v in df["college_event"].value_counts().items()) if not df.empty else "N/A"

    date_range_str = f"{metrics['date_min']} to {metrics['date_max']}" if metrics["date_min"] != "N/A" else "N/A"

    # ---------------- STAGE 1 ----------------
    if stage_id == 1:
        inputs_summary = {
            "Total Records": metrics["total_records"],
            "Meals Prepared": metrics["total_prepared"],
            "Meals Sold": metrics["total_sold"],
            "Leftover Portions": metrics["total_leftovers"],
            "Estimated Waste": f"{metrics['total_waste_kg']} kg",
            "Waste Percentage": f"{metrics['waste_percentage']}%",
            "Realized Revenue": f"₹{int(metrics['total_revenue_inr'])}",
            "Date Range": date_range_str,
        }

        prompt = f"""# ROLE
You are an expert Food Waste Analyst for a college canteen working toward UN SDG 2 (Zero Hunger) and SDG 12.3 (Halving Food Waste).

# OBJECTIVE
Analyze the uploaded canteen dataset and identify important food waste patterns. Diagnose operational waste drivers and distinguish observed facts from hypotheses.

# SUPPLIED DATA (Calculated by Python)
- Date Range: {date_range_str}
- Total Records Analyzed: {metrics['total_records']}
- Total Meals Prepared: {metrics['total_prepared']:,}
- Total Meals Sold: {metrics['total_sold']:,}
- Total Leftover Portions (Unsold): {metrics['total_leftovers']:,}
- Estimated Total Waste Weight: {metrics['total_waste_kg']:.2f} kg
- Overall Calculated Waste Percentage: {metrics['waste_percentage']:.2f}%
- Total Realized Revenue: ₹{int(metrics['total_revenue_inr']):,}
- Operating Conditions Distribution:
  * Weather: {weather_summary}
  * Exam Periods: {exam_summary}
  * College Events: {event_summary}

- Menu-Wise Performance Summary:
{menu_table}

# CONSTRAINTS & RULES
1. Ground all quantitative conclusions purely on the supplied metrics. Do NOT invent new numbers, percentages, or savings.
2. Identify the menu items with the greatest observed waste percentage and volume.
3. Compare prepared meals with sold meals across the menu portfolio.
4. Clearly separate observed empirical facts (e.g. recorded leftovers) from hypotheses (e.g. weather or exams). Never assert causation without direct evidence.
5. Provide a clear structured summary of findings that can be passed directly to Stage 2 (Demand Estimator).
6. CONCISENESS MANDATE: Keep the entire output strictly under 250 words total. Provide direct, concise bullet points for each heading. Avoid rambling filler.

# REQUIRED OUTPUT FORMAT

## 1. Executive Summary & Waste Severity
Summarize the supplied figures and classify the overall waste level (Low: <10%, Moderate: 10-20%, High: >20%).

## 2. Key Food Waste Patterns
Detail the major discrepancies between prepared meals and sold meals across dates and categories.

## 3. High-Waste Menu Items
Identify the top problematic items, their waste percentages, and kilograms lost.

## 4. Possible Explanations (Labelled as Hypotheses)
Provide up to 3 potential drivers (e.g., foot traffic variation, weather, exam attendance), explicitly labelling each as a hypothesis.

## 5. Data Limitations & Monitoring Needs
State the limitations of this dataset and 2 specific data points needed to confirm hypotheses.

## 6. Structured Summary for Stage 2
Provide a concise bulleted briefing of high-waste items and overproduction hotspots to inform Stage 2 Demand Estimation.
"""
        return prompt.strip(), inputs_summary, []

    # ---------------- STAGE 2 ----------------
    elif stage_id == 2:
        stage1_output = pipeline_state[1]["raw_output"]
        inputs_summary = {
            "Stage 1 Output Length": f"{len(stage1_output)} chars",
            "Unique Menu Items": metrics["unique_menu_items_count"],
            "Operating Conditions": f"Weather: {weather_summary} | Exams: {exam_summary} | Events: {event_summary}",
        }

        # Calculate baseline sample for top 3 items
        baseline_lines = []
        for item in df["menu_item"].unique()[:5]:
            base = calculate_demand_baseline(df, item)
            baseline_lines.append(
                f"- {item}: Median Sales = {base['median_meals_sold']} portions | "
                f"Range = [{base['min_meals_sold']} - {base['max_meals_sold']}] | "
                f"Python Target (Median + 5% buffer) = {base['calculated_baseline_target']} portions"
            )
        baseline_text = "\n".join(baseline_lines)

        prompt = f"""# ROLE
You are a Canteen Demand & Production Planning Specialist focused on eliminating kitchen overproduction and food waste.

# OBJECTIVE
Use the dataset statistics and the Stage 1 Food Waste Analysis findings to recommend appropriate meal preparation quantities and staged cooking strategies.

# INPUT DATA 1: HISTORICAL SALES & DEMAND STATISTICS (Calculated by Python)
{baseline_text}

Operating Context Summary:
- Weather Distribution: {weather_summary}
- Exam Period Distribution: {exam_summary}
- College Events: {event_summary}

# INPUT DATA 2: UPSTREAM STAGE 1 OUTPUT (FOOD WASTE ANALYST FINDINGS)
\"\"\"
{stage1_output}
\"\"\"

# CONSTRAINTS & RULES
1. Ground your recommendations directly in the historical sales aggregates and the Stage 1 waste diagnoses.
2. Consider overproduction patterns identified in Stage 1 and explain the reasoning behind each recommendation.
3. Transparently label suggested preparation quantities as operational recommendations, NOT statistically validated machine learning forecasts.
4. Do NOT invent historical observations or demand spikes.
5. State uncertainty clearly when observations are limited.
6. Emphasize a staged cooking / batching strategy (e.g., initial base prep + on-demand batching) to mitigate demand fluctuations.
7. Provide a structured result that can be passed to Stage 3 (Menu Optimizer).
8. CONCISENESS MANDATE: Keep the entire output strictly under 250 words total. Provide direct bulleted targets and clear integer portion recommendations.

# REQUIRED OUTPUT FORMAT

## 1. Baseline Demand Evaluation
Summarize the observed historical sales ranges, median demand, and Python baseline targets.

## 2. Integration with Stage 1 Waste Findings
Explain how the overproduction and high-waste items diagnosed in Stage 1 guide preparation adjustments.

## 3. Recommended Preparation Quantities
Provide specific recommended preparation quantities (or conservative ranges) for the key menu items.

## 4. Staged Cooking & Risk Management
Outline practical batch preparation schedules (e.g. 70% opening batch, 30% on-demand) to prevent surplus.

## 5. Uncertainty & Limitations
Identify planning uncertainties related to campus attendance, exam schedules, or weather.

## 6. Structured Demand Targets for Stage 3
Provide a clear summary of recommended batch limits and target portions to inform Stage 3 Menu Optimization.
"""
        return prompt.strip(), inputs_summary, [1]

    # ---------------- STAGE 3 ----------------
    elif stage_id == 3:
        stage1_output = pipeline_state[1]["raw_output"]
        stage2_output = pipeline_state[2]["raw_output"]
        inputs_summary = {
            "Stage 1 Output Used": "Yes (Waste diagnoses)",
            "Stage 2 Output Used": "Yes (Demand targets)",
            "Menu Items Evaluated": metrics["unique_menu_items_count"],
        }

        prompt = f"""# ROLE
You are a Canteen Menu Engineer and Food Waste Reduction Strategist.

# OBJECTIVE
Generate practical menu and preparation changes using the previous findings from Stage 1 (Waste Analysis) and Stage 2 (Demand Estimation).

# INPUT DATA 1: MENU PORTFOLIO SUMMARY (Calculated by Python)
{menu_table}

# INPUT DATA 2: UPSTREAM STAGE 1 OUTPUT (FOOD WASTE ANALYST)
\"\"\"
{stage1_output}
\"\"\"

# INPUT DATA 3: UPSTREAM STAGE 2 OUTPUT (DEMAND ESTIMATOR)
\"\"\"
{stage2_output}
\"\"\"

# CONSTRAINTS & RULES
1. Recommend adjustments specifically for high-waste menu items identified in Stages 1 and 2.
2. Balance waste reduction, demand stability, culinary variety, and kitchen practicality.
3. Avoid recommending the complete removal of a menu item based on a single observation or isolated day.
4. Clearly distinguish data-supported recommendations from culinary or operational assumptions.
5. Do NOT invent financial savings, cost metrics, or profit margins not supplied in the input.
6. Propose ingredient synergies (shared raw materials across dishes) to reduce perishable inventory spoilage.
7. Provide a structured result that can be passed to Stage 4 (Surplus Food Manager).
8. CONCISENESS MANDATE: Keep the entire output strictly under 250 words total. Provide actionable, concise bullet points for each heading.

# REQUIRED OUTPUT FORMAT

## 1. Menu Portfolio Optimization
Evaluate the performance balance: identify staple anchor dishes versus high-risk perishable items.

## 2. Recipe Batching & Sizing Adjustments
Recommend specific batch preparation adjustments combining Stage 2 demand targets with kitchen portioning.

## 3. Menu Scheduling & Rotation Strategy
Suggest calendar scheduling changes (e.g., swapping high-waste items off low-traffic exam days or rainy days).

## 4. Ingredient Synergy & Waste Prevention
Detail practical ways the kitchen can cross-utilize perishable ingredients across dishes.

## 5. Implementation Considerations & Trade-offs
Outline student dining preferences, kitchen workload trade-offs, and practical operational constraints.

## 6. Structured Menu Guidance for Stage 4
Summarize anticipated leftover profiles and dish characteristics to guide Stage 4 Surplus Management.
"""
        return prompt.strip(), inputs_summary, [1, 2]

    # ---------------- STAGE 4 ----------------
    elif stage_id == 4:
        stage1_output = pipeline_state[1]["raw_output"]
        stage2_output = pipeline_state[2]["raw_output"]
        stage3_output = pipeline_state[3]["raw_output"]
        inputs_summary = {
            "Total Surplus Portions": metrics["total_leftovers"],
            "Estimated Waste Weight": f"{metrics['total_waste_kg']} kg",
            "Stages 1, 2, 3 Outputs Used": "Yes",
        }

        prompt = f"""# ROLE
You are a Campus Food Recovery and Safe Surplus Redistribution Coordinator.

# OBJECTIVE
Recommend safe, responsible, and hygienic handling of canteen surplus food adhering strictly to food safety standards and the Food Recovery Hierarchy.

# CRITICAL FOOD SAFETY MANDATE
NEVER suggest donating, serving, or redistributing food that is spoiled, unhygienic, or held within the ambient temperature danger zone (>2 hours). Food safety takes absolute precedence over zero-hunger redistribution goals.

# INPUT DATA 1: SURPLUS VOLUMES & CONTEXT (Calculated by Python)
- Total Leftover Portions: {metrics['total_leftovers']:,}
- Total Estimated Waste Weight: {metrics['total_waste_kg']:.2f} kg
- Overall Waste Percentage: {metrics['waste_percentage']:.2f}%
- Operating Conditions: Weather = {weather_summary} | Exams = {exam_summary} | Events = {event_summary}

# INPUT DATA 2: UPSTREAM STAGE 1 (WASTE ANALYSIS)
\"\"\"
{stage1_output}
\"\"\"

# INPUT DATA 3: UPSTREAM STAGE 2 (DEMAND RECOMMENDATIONS)
\"\"\"
{stage2_output}
\"\"\"

# INPUT DATA 4: UPSTREAM STAGE 3 (MENU OPTIMIZATION)
\"\"\"
{stage3_output}
\"\"\"

# CONSTRAINTS & RULES
1. Prioritize preventing avoidable surplus at the source (referencing Stages 2 and 3).
2. For existing edible surplus, follow the Food Recovery Hierarchy: (1) Source reduction -> (2) Safe human redistribution -> (3) Safe repurposing -> (4) Composting/Disposal.
3. Suggest safe, timely handling: enforce the 2-hour ambient rule, rapid cooling, hot-holding, clean food-grade storage, and timestamped labelling.
4. Do NOT assume leftover food is automatically safe to donate without food-safety screening.
5. Do NOT invent external partner charity organizations or claim redistribution has already taken place.
6. Provide a structured result to pass to Stage 5 (Impact Analyst).
7. CONCISENESS MANDATE: Keep the entire output strictly under 250 words total. Provide direct bulleted protocols for each heading.

# REQUIRED OUTPUT FORMAT

## 1. Immediate Safety & Edibility Screening
Classify surplus items by spoilage risk and temperature sensitivity, specifying safety criteria for consumption.

## 2. Immediate Safe Storage & Handling Protocols
Detail concrete protocols for canteen staff (chilling, hot holding, sterile packaging, timestamping).

## 3. Safe Redistribution Pathways
Recommend realistic on-campus and immediate recovery options (e.g. campus evening student discount, verified community distribution) with strict time windows.

## 4. Secondary Repurposing or Safe Disposal
Specify kitchen repurposing options only if within hygiene guidelines; outline composting/organic waste segregation if food safety thresholds have expired.

## 5. Root Cause Prevention Feedback
Provide 2 operational feedback directives for kitchen managers to prevent this surplus tomorrow.

## 6. Structured Surplus Protocols for Stage 5
Summarize portion recovery potentials and safety constraints to feed into Stage 5 Impact Analysis.
"""
        return prompt.strip(), inputs_summary, [1, 2, 3]

    # ---------------- STAGE 5 ----------------
    elif stage_id == 5:
        stage1_output = pipeline_state[1]["raw_output"]
        stage2_output = pipeline_state[2]["raw_output"]
        stage3_output = pipeline_state[3]["raw_output"]
        stage4_output = pipeline_state[4]["raw_output"]
        inputs_summary = {
            "Python Summary Metrics": "Loaded",
            "Upstream Stages Integrated": "Stages 1, 2, 3, and 4",
            "Total Prepared": metrics["total_prepared"],
            "Total Sold": metrics["total_sold"],
            "Total Waste (kg)": f"{metrics['total_waste_kg']} kg",
            "Waste Percentage": f"{metrics['waste_percentage']}%",
            "Realized Revenue": f"₹{int(metrics['total_revenue_inr'])}",
        }

        prompt = f"""# ROLE
You are a Lead Sustainability & Impact Assessment Analyst specializing in food systems and UN Sustainable Development Goal 2 (Zero Hunger) and SDG 12.3 (Halving Food Waste).

# OBJECTIVE
Create the final consolidated report synthesizing the Python-calculated metrics and the findings from all four preceding stages (Waste Analysis, Demand Estimation, Menu Optimization, Surplus Management).

# INPUT DATA 1: DETERMINISTIC DATASET METRICS (Calculated by Python)
- Total Records Analyzed: {metrics['total_records']}
- Total Meals Prepared: {metrics['total_prepared']:,}
- Total Meals Sold: {metrics['total_sold']:,}
- Total Leftover Portions (Unserved): {metrics['total_leftovers']:,}
- Overall Waste Percentage: {metrics['waste_percentage']:.2f}%
- Total Estimated Waste Weight: {metrics['total_waste_kg']:.2f} kg
- Total Realized Revenue: ₹{int(metrics['total_revenue_inr']):,}
- Estimated Cost Value of Wasted Food: ₹{int(metrics['estimated_waste_cost_inr']):,}

# INPUT DATA 2: UPSTREAM STAGE 1 (FOOD WASTE ANALYST)
\"\"\"
{stage1_output}
\"\"\"

# INPUT DATA 3: UPSTREAM STAGE 2 (DEMAND ESTIMATOR)
\"\"\"
{stage2_output}
\"\"\"

# INPUT DATA 4: UPSTREAM STAGE 3 (MENU OPTIMIZER)
\"\"\"
{stage3_output}
\"\"\"

# INPUT DATA 5: UPSTREAM STAGE 4 (SURPLUS FOOD MANAGER)
\"\"\"
{stage4_output}
\"\"\"

# CONSTRAINTS & RULES
1. Consolidate earlier outputs into an executive-level institutional report without contradicting them.
2. Ground all numbers strictly in the provided Python metrics. Calculate NO numerical metrics independently.
3. Do NOT invent fictitious carbon/GHG reductions, water footprint multipliers, monetary savings, or meals donated.
4. Distinguish measured historical baseline results from expected qualitative benefits (label expectations as qualitative estimates).
5. Explicitly state the limitations of synthetic demonstration data.
6. Connect the multi-stage operational interventions directly to UN SDG 2 (Zero Hunger) and SDG 12.3.
7. CONCISENESS MANDATE: Keep the entire output strictly under 300 words total. Provide concise, high-impact bulleted takeaways.

# REQUIRED OUTPUT FORMAT

## 1. Executive Summary & Waste Footprint
Synthesize overall production volume, leftovers, waste percentages, and financial magnitude calculated by Python.

## 2. Multi-Stage Pipeline Synthesis
Briefly summarize how the pipeline progressed from Waste Diagnosis (Stage 1) -> Demand Estimation (Stage 2) -> Menu Engineering (Stage 3) -> Safe Surplus Handling (Stage 4).

## 3. Priority Actions & Implementation Roadmap
Detail 3 immediate high-priority actions for canteen management combining production, menu scheduling, and surplus protocols.

## 4. Measurable Indicators for Future Monitoring
Specify 4 quantitative KPIs (e.g. daily waste %, batch overproduction variance, surplus recovery rate) the canteen should track.

## 5. Expected Benefits (Qualitative Estimates)
Describe expected benefits on food security, operational resilience, and kitchen efficiency, clearly labelled as estimates.

## 6. Synthetic Data Limitations & Final Conclusion
Conclude with observations on synthetic dataset boundaries and the institutional value of the sequential LLM prompt pipeline.
"""
        return prompt.strip(), inputs_summary, [1, 2, 3, 4]

    raise ValueError(f"Invalid stage_id: {stage_id}. Must be 1 to 5.")


def execute_pipeline_stage(
    stage_id: int,
    df: pd.DataFrame,
    pipeline_state: Dict[int, Dict[str, Any]],
    ollama_client: OllamaClient,
    custom_prompt_text: Optional[str] = None,
    num_predict: Optional[int] = None,
    timeout: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute a single pipeline stage via Ollama with dependency enforcement and downstream invalidation.
    
    Returns the updated stage dictionary.
    """
    if num_predict is None and "num_predict" in kwargs:
        num_predict = kwargs["num_predict"]
    if timeout is None and "timeout" in kwargs:
        timeout = kwargs["timeout"]

    can_run, reason = can_run_stage(stage_id, pipeline_state)
    if not can_run:
        pipeline_state[stage_id]["status"] = "failed"
        pipeline_state[stage_id]["error"] = reason
        pipeline_state[stage_id]["status_message"] = reason
        return pipeline_state[stage_id]

    # Assemble user prompt
    assembled_prompt, inputs_summary, previous_stages_used = assemble_stage_user_prompt(
        stage_id=stage_id,
        df=df,
        pipeline_state=pipeline_state,
        user_override_prompt=custom_prompt_text or pipeline_state[stage_id]["custom_user_prompt"],
    )

    stage_entry = pipeline_state[stage_id]
    stage_entry["assembled_prompt"] = assembled_prompt
    stage_entry["inputs_summary"] = inputs_summary
    stage_entry["previous_stages_used"] = previous_stages_used
    stage_entry["is_custom"] = bool(custom_prompt_text or stage_entry["custom_user_prompt"])
    stage_entry["status"] = "running"
    stage_entry["status_message"] = "Querying local Qwen2.5 7B via Ollama..."
    stage_entry["error"] = None

    # Call Ollama
    res = ollama_client.generate(
        prompt=assembled_prompt,
        system_instruction=stage_entry["system_prompt"],
        num_predict=num_predict,
        timeout=timeout,
    )

    if res["success"]:
        stage_entry["status"] = "completed"
        stage_entry["status_message"] = f"Completed in {res['duration_seconds']}s"
        stage_entry["raw_output"] = res["response"]
        stage_entry["duration_seconds"] = res["duration_seconds"]
        stage_entry["eval_count"] = res.get("eval_count") or 0
        stage_entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stage_entry["error"] = None
        stage_entry["stale_reason"] = None

        # Downstream invalidation: any stage > stage_id is marked stale
        invalidate_downstream_stages(stage_id, pipeline_state)
    else:
        stage_entry["status"] = "failed"
        stage_entry["status_message"] = "Execution failed"
        stage_entry["error"] = res["error"]
        # Do not erase previous successful outputs of this stage if any existed, but keep error visible

    return stage_entry


def generate_consolidated_report(
    df: pd.DataFrame,
    pipeline_state: Dict[int, Dict[str, Any]],
) -> str:
    """Generate the comprehensive final consolidated markdown report from all 5 pipeline stages."""
    metrics = calculate_summary_metrics(df)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        "# 🍲 Zero Hunger Smart Canteen: Sequential LLM Pipeline Final Report",
        f"**Generated on**: {timestamp} | **Local Engine**: Qwen2.5 7B via Ollama",
        "",
        "---",
        "## 📊 Verified Python Dataset Metrics",
        f"- **Dataset Scope**: {metrics['date_min']} to {metrics['date_max']} ({metrics['total_records']} service records)",
        f"- **Total Meals Prepared**: {metrics['total_prepared']:,} portions",
        f"- **Total Meals Sold**: {metrics['total_sold']:,} portions ({(metrics['total_sold']/metrics['total_prepared']*100 if metrics['total_prepared']>0 else 0):.1f}% consumption rate)",
        f"- **Total Leftover Surplus**: {metrics['total_leftovers']:,} portions ({metrics['waste_percentage']:.2f}% waste rate)",
        f"- **Estimated Food Waste Weight**: {metrics['total_waste_kg']:.2f} kg",
        f"- **Total Realized Revenue**: ₹{int(metrics['total_revenue_inr']):,}",
        f"- **Unrecovered Surplus Value**: ₹{int(metrics['estimated_waste_cost_inr']):,}",
        "",
        "---",
    ]

    for s_id in range(1, 6):
        stage = pipeline_state.get(s_id, {})
        status = stage.get("status", "pending")
        output = stage.get("raw_output", "").strip()
        duration = stage.get("duration_seconds", 0.0)

        sections.append(f"## {stage.get('name', f'Stage {s_id}')}")
        sections.append(f"*Status: {status.upper()} | Execution Duration: {duration}s*")
        sections.append("")

        if status == "completed" and output:
            sections.append(output)
        elif status == "stale":
            sections.append(f"> ⚠️ **Warning**: Output is stale because upstream dependencies were rerun.\n\n{output}")
        else:
            sections.append(f"> ⏳ *Stage output not available (Status: {status}).*")

        sections.append("")
        sections.append("---")

    sections.append("## 🌍 UN Sustainable Development Goals Disclosure")
    sections.append(
        "This report was produced by a sequential prompt-engineering pipeline aligned with UN SDG 2 (Zero Hunger) "
        "and SDG 12.3 (Halving Food Waste). Deterministic metrics are calculated exclusively in Python, and "
        "qualitative operational recommendations are reasoned by local Qwen2.5 7B. The dataset represents synthetic demonstration data."
    )

    return "\n".join(sections)
