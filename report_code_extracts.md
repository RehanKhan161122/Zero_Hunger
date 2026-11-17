# Zero Hunger Smart Canteen — Project Implementation Code Extracts

This document contains the exact code extracts for Section 3 of the project report, ordered sequentially with labels, file references, and line numbers.

---

## 3.1 Application interface

- **File**: `app.py`
- **Exact Lines**: `312–423`

```python
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
```

---

## 3.2 Dataset upload and processing

- **File**: `app.py`
- **Exact Lines**: `337–380`

```python
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
```

---

## 3.3 Dataset validation

- **File**: `src/data_loader.py`
- **Exact Lines**: `88–153`

```python
def validate_and_process_canteen_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, apply column aliasing, cast datatypes, and format canteen data."""
    if raw_df is None or raw_df.empty:
        raise ValueError("The provided dataset is empty. Please provide a CSV with valid canteen records.")

    df = raw_df.copy()

    # Standardize column headers (lowercase, stripped)
    column_mapping = {}
    current_cols = [str(c).strip() for c in df.columns]
    df.columns = current_cols

    # Check for alias matches if required columns are not directly present
    for req_col in REQUIRED_COLUMNS:
        if req_col not in df.columns:
            # Look for alias
            for candidate in COLUMN_ALIASES.get(req_col, []):
                for col in df.columns:
                    if col.lower() == candidate.lower():
                        column_mapping[col] = req_col
                        break
                if req_col in column_mapping.values():
                    break

    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Check for missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_cols)}.\n"
            f"Detected columns: {list(df.columns)}.\n"
            f"Expected schema: {', '.join(REQUIRED_COLUMNS)}"
        )

    # Clean string columns
    for col in ["day_of_week", "menu_item", "weather", "exam_period", "college_event"]:
        df[col] = df[col].astype(str).str.strip()

    # Parse and validate dates
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    except Exception as e:
        raise ValueError(f"Error parsing 'date' column into valid dates: {e}") from e

    # Validate and cast numeric columns
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # If waste_percentage is 0 but leftovers > 0 and meals_prepared > 0, recalculate accurately
    mask = (df["waste_percentage"] == 0) & (df["meals_prepared"] > 0) & (df["leftover_portions"] > 0)
    if mask.any():
        df.loc[mask, "waste_percentage"] = (
            (df.loc[mask, "leftover_portions"] / df.loc[mask, "meals_prepared"]) * 100.0
        ).round(2)

    # If revenue_inr is 0, calculate as meals_sold * price_inr
    rev_mask = (df["revenue_inr"] == 0) & (df["meals_sold"] > 0) & (df["price_inr"] > 0)
    if rev_mask.any():
        df.loc[rev_mask, "revenue_inr"] = (df.loc[rev_mask, "meals_sold"] * df.loc[rev_mask, "price_inr"]).round(2)

    # Sort deterministically
    df = df.sort_values(by=["date", "menu_item"]).reset_index(drop=True)
    return df

```

---

## 3.4 Summary KPI calculations

- **File**: `src/analytics.py`
- **Exact Lines**: `12–79`

```python
def calculate_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate aggregate summary metrics for a given dataframe slice.
    
    Handles empty slices and zero division safely.
    """
    if df.empty:
        return {
            "total_records": 0,
            "total_prepared": 0,
            "total_sold": 0,
            "total_leftovers": 0,
            "total_waste_kg": 0.0,
            "waste_percentage": 0.0,
            "total_revenue_inr": 0.0,
            "estimated_waste_cost_inr": 0.0,
            "avg_prepared_per_record": 0.0,
            "avg_sold_per_record": 0.0,
            "avg_waste_kg_per_record": 0.0,
            "avg_waste_percentage": 0.0,
            "unique_menu_items_count": 0,
            "date_min": "N/A",
            "date_max": "N/A",
        }

    total_records = len(df)
    total_prepared = int(df["meals_prepared"].sum())
    total_sold = int(df["meals_sold"].sum())
    total_leftovers = int(df["leftover_portions"].sum())
    total_waste_kg = round(float(df["estimated_waste_kg"].sum()), 2)
    total_revenue_inr = round(float(df["revenue_inr"].sum()), 2)

    # Waste percentage calculated from aggregate leftovers vs aggregate prepared
    waste_percentage = (
        round((total_leftovers / total_prepared) * 100.0, 2)
        if total_prepared > 0
        else 0.0
    )

    # Estimated monetary value of food waste: sum(leftovers * price)
    waste_costs = df["leftover_portions"] * df["price_inr"]
    estimated_waste_cost_inr = round(float(waste_costs.sum()), 2)

    avg_prepared = round(float(df["meals_prepared"].mean()), 2)
    avg_sold = round(float(df["meals_sold"].mean()), 2)
    avg_waste_kg = round(float(df["estimated_waste_kg"].mean()), 2)
    avg_waste_pct = round(float(df["waste_percentage"].mean()), 2)

    unique_items = int(df["menu_item"].nunique())
    date_min = str(df["date"].min())
    date_max = str(df["date"].max())

    return {
        "total_records": total_records,
        "total_prepared": total_prepared,
        "total_sold": total_sold,
        "total_leftovers": total_leftovers,
        "total_waste_kg": total_waste_kg,
        "waste_percentage": waste_percentage,
        "total_revenue_inr": total_revenue_inr,
        "estimated_waste_cost_inr": estimated_waste_cost_inr,
        "avg_prepared_per_record": avg_prepared,
        "avg_sold_per_record": avg_sold,
        "avg_waste_kg_per_record": avg_waste_kg,
        "avg_waste_percentage": avg_waste_pct,
        "unique_menu_items_count": unique_items,
        "date_min": date_min,
        "date_max": date_max,
    }
```

---

## 3.5 Menu performance analysis

- **File**: `src/analytics.py`
- **Exact Lines**: `82–135`

```python
def calculate_menu_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance metrics per menu item.
    
    Includes sales volume, leftovers, waste kg, waste percentage, revenue, and waste cost.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "menu_item",
                "records_count",
                "total_prepared",
                "total_sold",
                "total_leftovers",
                "total_waste_kg",
                "waste_percentage",
                "total_revenue_inr",
                "estimated_waste_cost_inr",
                "avg_price_inr",
            ]
        )

    grouped = (
        df.groupby("menu_item")
        .apply(
            lambda g: pd.Series(
                {
                    "records_count": len(g),
                    "total_prepared": int(g["meals_prepared"].sum()),
                    "total_sold": int(g["meals_sold"].sum()),
                    "total_leftovers": int(g["leftover_portions"].sum()),
                    "total_waste_kg": round(float(g["estimated_waste_kg"].sum()), 2),
                    "waste_percentage": (
                        round(
                            (g["leftover_portions"].sum() / g["meals_prepared"].sum())
                            * 100.0,
                            2,
                        )
                        if g["meals_prepared"].sum() > 0
                        else 0.0
                    ),
                    "total_revenue_inr": round(float(g["revenue_inr"].sum()), 2),
                    "estimated_waste_cost_inr": round(
                        float((g["leftover_portions"] * g["price_inr"]).sum()), 2
                    ),
                    "avg_price_inr": round(float(g["price_inr"].mean()), 2),
                }
            )
        )
        .reset_index()
    )

    return grouped.sort_values(by="waste_percentage", ascending=False).reset_index(
        drop=True
    )
```

---

## 3.6 Demand baseline calculation

- **File**: `src/analytics.py`
- **Exact Lines**: `138–196`

```python
def calculate_demand_baseline(
    df: pd.DataFrame,
    menu_item: str,
    day_of_week: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate historical demand statistics for a menu item.
    
    Provides transparent arithmetic baseline targets (median demand) without claiming
    unverified ML statistical prediction.
    """
    subset = df[df["menu_item"] == menu_item]
    if day_of_week and not subset.empty:
        day_subset = subset[subset["day_of_week"] == day_of_week]
        if not day_subset.empty:
            subset = day_subset

    if subset.empty:
        return {
            "menu_item": menu_item,
            "day_of_week": day_of_week or "All",
            "historical_records_count": 0,
            "avg_meals_sold": 0.0,
            "median_meals_sold": 0.0,
            "min_meals_sold": 0,
            "max_meals_sold": 0,
            "std_meals_sold": 0.0,
            "avg_waste_percentage": 0.0,
            "calculated_baseline_target": 0,
            "baseline_strategy": "No historical observations available.",
        }

    records_count = len(subset)
    avg_sold = round(float(subset["meals_sold"].mean()), 1)
    median_sold = round(float(subset["meals_sold"].median()), 1)
    min_sold = int(subset["meals_sold"].min())
    max_sold = int(subset["meals_sold"].max())
    std_sold = round(float(subset["meals_sold"].std(ddof=0)), 1)
    avg_waste_pct = round(float(subset["waste_percentage"].mean()), 2)

    # Transparent baseline: Use median sales as robust central tendency,
    # plus a conservative 5% buffer rounded to nearest integer.
    calculated_baseline_target = int(round(median_sold * 1.05))

    return {
        "menu_item": menu_item,
        "day_of_week": day_of_week or "All Days",
        "historical_records_count": records_count,
        "avg_meals_sold": avg_sold,
        "median_meals_sold": median_sold,
        "min_meals_sold": min_sold,
        "max_meals_sold": max_sold,
        "std_meals_sold": std_sold,
        "avg_waste_percentage": avg_waste_pct,
        "calculated_baseline_target": calculated_baseline_target,
        "baseline_strategy": (
            f"Median historical sales ({median_sold} portions) with a conservative "
            f"5% buffer ({calculated_baseline_target} portions target)."
        ),
    }
```

---

## 3.7 Ollama integration

- **File**: `src/ollama_client.py`
- **Exact Lines**: `18–83, 85–149`

### Part 1 (Lines 18–83)

```python
class OllamaClient:
    """Client for querying a local Ollama instance."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        num_predict: Optional[int] = None,
    ):
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
        self.num_predict = num_predict if num_predict is not None else DEFAULT_NUM_PREDICT

    def check_health(self) -> Tuple[bool, str, List[str]]:
        """Check if local Ollama server is running and inspect loaded models.
        
        Returns:
            Tuple of (is_connected: bool, status_message: str, model_names: List[str])
        """
        tags_url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(tags_url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                if self.model in models or any(self.model in m for m in models):
                    return (
                        True,
                        f"Connected to Ollama. Model '{self.model}' is available.",
                        models,
                    )
                else:
                    return (
                        True,
                        f"Connected to Ollama, but model '{self.model}' was not found in installed models. Run: `ollama run {self.model}`",
                        models,
                    )
            else:
                return (
                    False,
                    f"Ollama server returned HTTP {resp.status_code}: {resp.text}",
                    [],
                )
        except requests.exceptions.ConnectionError:
            return (
                False,
                f"Cannot connect to Ollama at {self.base_url}. Make sure the Ollama service is running. Start it with `ollama serve` or open the Ollama app.",
                [],
            )
        except requests.exceptions.Timeout:
            return (
                False,
                f"Connection to Ollama at {self.base_url} timed out.",
                [],
            )
        except Exception as e:
            return (
                False,
                f"Ollama health check error: {str(e)}",
                [],
            )
```

### Part 2 (Lines 85–149)

```python
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        num_predict: Optional[int] = None,
        timeout: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send prompt to local Ollama /api/generate with stream=False.
        
        Returns structured dictionary with success status, response text, timing, and error details.
        """
        target_model = model or self.model
        generate_url = f"{self.base_url}/api/generate"
        effective_timeout = timeout if timeout is not None else self.timeout
        effective_num_predict = num_predict if num_predict is not None else self.num_predict

        gen_options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": effective_num_predict,
            "num_ctx": DEFAULT_NUM_CTX,
        }
        if options:
            gen_options.update(options)

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": gen_options,
        }

        if system_instruction:
            payload["system"] = system_instruction

        start_time = time.time()
        try:
            response = requests.post(
                generate_url,
                json=payload,
                timeout=effective_timeout,
            )
            duration_s = round(time.time() - start_time, 2)

            if response.status_code == 200:
                result_json = response.json()
                generated_text = result_json.get("response", "").strip()
                eval_count = result_json.get("eval_count")
                total_duration_ms = (
                    round(result_json.get("total_duration", 0) / 1e6, 2)
                    if "total_duration" in result_json
                    else None
                )

                return {
                    "success": True,
                    "response": generated_text,
                    "model": target_model,
                    "duration_seconds": duration_s,
                    "total_duration_ms": total_duration_ms,
                    "eval_count": eval_count,
                    "error": None,
                }
```

---

## 3.8 Prompt template management

- **File**: `src/prompt_manager.py`
- **Exact Lines**: `50–71`

```python
def get_available_modules() -> List[str]:
    """Return the list of prompt module identifiers."""
    return list(MODULE_METADATA.keys())


def load_template(module_name: str) -> str:
    """Load the raw text template from the prompts directory."""
    template_path = PROMPTS_DIR / f"{module_name}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template file not found at: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _summarize_condition(series: pd.Series) -> str:
    """Helper to summarize distribution of categorical conditions."""
    if series.empty:
        return "Not available"
    counts = series.value_counts()
    items = [f"{val}: {count}" for val, count in counts.items()]
    return ", ".join(items)


```

---

## 3.9 Prompt assembly

- **File**: `src/prompt_manager.py`
- **Exact Lines**: `72–296`

```python
def assemble_food_waste_prompt(
    df: pd.DataFrame,
    scope_desc: str = "Current Selection",
    additional_context: str = "Standard college operating day",
) -> str:
    """Assemble the Food Waste Analyst prompt using calculated metrics."""
    template = load_template("food_waste_analyst")
    metrics = calculate_summary_metrics(df)

    if df.empty:
        return template.format(
            analysis_scope=scope_desc,
            date_range="None",
            menu_items="None",
            total_prepared=0,
            total_sold=0,
            leftover_portions=0,
            estimated_waste_kg=0.0,
            waste_percentage=0.0,
            weather_summary="No data",
            exam_period_summary="No data",
            college_event_summary="No data",
            additional_context="Empty dataset selected.",
        )

    menu_items_str = ", ".join(df["menu_item"].unique())
    date_range_str = (
        f"{metrics['date_min']} to {metrics['date_max']}"
        if metrics["date_min"] != metrics["date_max"]
        else metrics["date_min"]
    )

    weather_str = _summarize_condition(df["weather"])
    exam_str = _summarize_condition(df["exam_period"])
    event_str = _summarize_condition(df["college_event"])

    return template.format(
        analysis_scope=scope_desc,
        date_range=date_range_str,
        menu_items=menu_items_str,
        total_prepared=metrics["total_prepared"],
        total_sold=metrics["total_sold"],
        leftover_portions=metrics["total_leftovers"],
        estimated_waste_kg=metrics["total_waste_kg"],
        waste_percentage=metrics["waste_percentage"],
        weather_summary=weather_str,
        exam_period_summary=exam_str,
        college_event_summary=event_str,
        additional_context=additional_context or "None noted",
    )


def assemble_demand_estimator_prompt(
    df_full: pd.DataFrame,
    menu_item: str,
    target_day: str = "Monday",
    weather: str = "Sunny",
    exam_period: str = "No",
    college_event: str = "No",
    additional_context: str = "",
) -> str:
    """Assemble Demand Estimator prompt using historical aggregates for the item."""
    template = load_template("demand_estimator")
    baseline = calculate_demand_baseline(df_full, menu_item, day_of_week=target_day)

    return template.format(
        menu_item=menu_item,
        target_day=f"{target_day} (historical reference)",
        historical_records_count=baseline["historical_records_count"],
        avg_meals_sold=baseline["avg_meals_sold"],
        median_meals_sold=baseline["median_meals_sold"],
        min_meals_sold=baseline["min_meals_sold"],
        max_meals_sold=baseline["max_meals_sold"],
        avg_waste_percentage=baseline["avg_waste_percentage"],
        calculated_baseline_target=baseline["calculated_baseline_target"],
        weather_summary=weather,
        exam_period_summary=f"Exams in progress: {exam_period}",
        college_event_summary=f"Event scheduled: {college_event}",
        additional_context=(
            f"{baseline['baseline_strategy']} | "
            + (additional_context or "General campus routine")
        ),
    )


def assemble_menu_optimizer_prompt(
    df: pd.DataFrame,
    scope_desc: str = "Filtered Selection",
    additional_context: str = "",
) -> str:
    """Assemble Menu Optimizer prompt using aggregated menu performance."""
    template = load_template("menu_optimizer")
    metrics = calculate_summary_metrics(df)
    menu_perf = calculate_menu_performance(df)

    if menu_perf.empty:
        summary_table = "No data available."
        highest_waste = "None"
        top_revenue = "None"
    else:
        summary_table = format_menu_summary_table(menu_perf, max_rows=10)

        # Top 3 waste percentage
        top_waste_df = menu_perf.sort_values(
            by="waste_percentage", ascending=False
        ).head(3)
        highest_waste = "; ".join(
            f"{row['menu_item']} ({row['waste_percentage']}% waste, {row['total_waste_kg']} kg)"
            for _, row in top_waste_df.iterrows()
        )

        # Top 3 revenue
        top_rev_df = menu_perf.sort_values(
            by="total_revenue_inr", ascending=False
        ).head(3)
        top_revenue = "; ".join(
            f"{row['menu_item']} (₹{int(row['total_revenue_inr'])}, {int(row['total_sold'])} sold)"
            for _, row in top_rev_df.iterrows()
        )

    date_range_str = (
        f"{metrics['date_min']} to {metrics['date_max']}"
        if metrics["date_min"] != "N/A"
        else "N/A"
    )
    operating_context = (
        f"Weather: {_summarize_condition(df['weather'])} | "
        f"Exams: {_summarize_condition(df['exam_period'])} | "
        f"Events: {_summarize_condition(df['college_event'])}"
        if not df.empty
        else "N/A"
    )

    return template.format(
        analysis_scope=scope_desc,
        date_range=date_range_str,
        menu_summary_table=summary_table,
        highest_waste_items=highest_waste,
        top_revenue_items=top_revenue,
        operating_context=operating_context,
        additional_context=additional_context or "Standard kitchen capacity and storage",
    )


def assemble_surplus_food_prompt(
    df: pd.DataFrame,
    scope_desc: str = "End of Service Surplus",
    service_phase: str = "End of Lunch Service (within 1 hour)",
    logistical_context: str = "On-campus cooling and local partner shelter nearby",
    additional_context: str = "",
) -> str:
    """Assemble Surplus Food Manager prompt adhering strictly to food safety."""
    template = load_template("surplus_food_manager")
    metrics = calculate_summary_metrics(df)
    menu_perf = calculate_menu_performance(df)

    if menu_perf.empty:
        items_breakdown = "No surplus recorded."
    else:
        # Items with leftovers > 0
        surplus_df = menu_perf[menu_perf["total_leftovers"] > 0]
        if surplus_df.empty:
            items_breakdown = "Zero leftovers recorded in this selection."
        else:
            lines = []
            for _, row in surplus_df.iterrows():
                lines.append(
                    f"- {row['menu_item']}: {int(row['total_leftovers'])} portions ({row['total_waste_kg']} kg approx)"
                )
            items_breakdown = "\n".join(lines)

    weather_str = _summarize_condition(df["weather"]) if not df.empty else "Normal"
    service_window_str = (
        f"{metrics['date_min']} to {metrics['date_max']}"
        if metrics["date_min"] != "N/A"
        else "Today's service"
    )

    return template.format(
        analysis_scope=scope_desc,
        service_window=service_window_str,
        surplus_items_breakdown=items_breakdown,
        total_surplus_portions=metrics["total_leftovers"],
        total_surplus_kg=metrics["total_waste_kg"],
        weather_summary=weather_str,
        service_phase=service_phase,
        logistical_context=logistical_context,
    )


def assemble_impact_analyst_prompt(
    df: pd.DataFrame,
    scope_desc: str = "Current Scope Period",
    context_note: str = "Demonstration synthetic canteen dataset representing student dining hall operations",
) -> str:
    """Assemble Impact Analyst prompt summarizing computed metrics."""
    template = load_template("impact_analyst")
    metrics = calculate_summary_metrics(df)

    date_range_str = (
        f"{metrics['date_min']} to {metrics['date_max']}"
        if metrics["date_min"] != "N/A"
        else "N/A"
    )

    weather_str = _summarize_condition(df["weather"]) if not df.empty else "N/A"
    exam_str = _summarize_condition(df["exam_period"]) if not df.empty else "N/A"
    event_str = _summarize_condition(df["college_event"]) if not df.empty else "N/A"

    return template.format(
        analysis_scope=scope_desc,
        date_range=date_range_str,
        total_records=metrics["total_records"],
        total_prepared=metrics["total_prepared"],
        total_sold=metrics["total_sold"],
        total_leftovers=metrics["total_leftovers"],
        waste_percentage=metrics["waste_percentage"],
        estimated_waste_kg=metrics["total_waste_kg"],
        total_revenue_inr=int(metrics["total_revenue_inr"]),
        estimated_waste_cost_inr=int(metrics["estimated_waste_cost_inr"]),
        exam_period_summary=exam_str,
        college_event_summary=event_str,
        weather_summary=weather_str,
        context_note=context_note,
    )
```

---

## 3.10 Pipeline dependencies

- **File**: `src/pipeline_engine.py`
- **Exact Lines**: `20–66, 99–124`

### Part 1 (Lines 20–66)

```python
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
```

### Part 2 (Lines 99–124)

```python
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
```

---

## 3.11 Sequential prompt assembly

- **File**: `src/pipeline_engine.py`
- **Exact Lines**: `149–240`

```python
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

```

---

## 3.12 Stage execution

- **File**: `src/pipeline_engine.py`
- **Exact Lines**: `531–602`

```python
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


```

---

## 3.13 Consolidated reporting

- **File**: `src/pipeline_engine.py`
- **Exact Lines**: `603–655`

```python
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
```

---

## 3.14 Prompt inspection interface

- **File**: `app.py`
- **Exact Lines**: `596–729`

```python
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
```

---

## 3.15 Report download

- **File**: `app.py`
- **Exact Lines**: `730–759`

```python
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
```

---

