
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# 1. Set up the page and explain the business purpose.
# ---------------------------------------------------------
st.set_page_config(
    page_title="How Much Should I Bake?",
    layout="wide"
)

st.title("How Much Should I Bake?")
st.caption("Explore bakery sales and predict sales for individual products.")

st.write(
    "This model predicts daily bakery sales to support production planning; "
    "overestimates may cause food waste, while underestimates may lead "
    "to stockouts and lost sales."
)

# Use readable labels while keeping the original CSV column names.
labels = {
    "quantity": "Daily units sold",
    "median_unit_price_eur": "Median unit price (€)",
    "estimated_revenue": "Estimated daily revenue (€)",
    "units_sold_7_days_ago": "Units sold seven days earlier",
    "previous_7_day_avg": "Average units sold over preceding seven days"
}

# ---------------------------------------------------------
# 2. Upload and validate the prepared CSV.
# Data preparation remains in the separate notebook.
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload bakery_prepared_top37.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload your prepared CSV to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read this CSV: {error}")
    st.stop()

required_columns = [
    "date",
    "article",
    "is_closed",
    "product_label_needs_review",
    *labels.keys()
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error("Missing required columns: " + ", ".join(missing_columns))
    st.stop()

# Convert dates and validate product labels.
df["date"] = pd.to_datetime(df["date"], errors="coerce")

if df.empty or df["date"].isna().any() or df["article"].isna().any():
    st.error("The file is empty or contains missing products or invalid dates.")
    st.stop()

# Read True/False flags consistently.
for column in ["is_closed", "product_label_needs_review"]:
    values = df[column].astype(str).str.strip().str.lower()

    if not values.isin(["true", "false"]).all():
        st.error(f"'{column}' must contain True or False.")
        st.stop()

    df[column] = values.eq("true")

# Convert the five analysis columns to numbers.
# Blanks remain missing; unexpected text produces a clear error.
for column in labels:
    converted = pd.to_numeric(df[column], errors="coerce")
    invalid = df[column].notna() & converted.isna()

    if invalid.any():
        st.error(f"'{column}' contains nonnumeric values.")
        st.stop()

    df[column] = converted

# Treat infinite numeric values as missing.
df[list(labels)] = df[list(labels)].replace(
    [float("inf"), float("-inf")],
    float("nan")
)

# ---------------------------------------------------------
# 3. Preview the file and select one product.
# ---------------------------------------------------------
st.subheader("1. Preview your data")

st.write(
    f"**{df['article'].nunique()} products** · "
    f"**{len(df):,} calendar rows**"
)

st.dataframe(df.head(20), use_container_width=True)

# Exclude unidentified labels from selection without modifying the file.
available = df.loc[~df["product_label_needs_review"]].copy()
products = sorted(available["article"].unique().tolist())

if not products:
    st.warning("No identifiable products are available.")
    st.stop()

default_product = (
    products.index("BAGUETTE") if "BAGUETTE" in products else 0
)

product = st.selectbox(
    "Choose a bakery product",
    options=products,
    index=default_product
)

# Always filter to one product before plotting or fitting a model.
product_df = (
    available.loc[available["article"] == product]
    .sort_values("date")
    .copy()
)

numeric_columns = list(labels)

with st.expander("How to interpret this dataset"):
    st.write(
        "The file contains 37 products selected during data preparation "
        "for their usable sales history. It retains calendar rows that "
        "are not suitable for modeling."
    )
    st.write(
        "Listed closures have zero sales. Missing records on other days "
        "remain unknown. Historical averages include closure days and "
        "exclude the day being predicted."
    )
    st.write(
        "Revenue is estimated using median prices in euros. Recorded "
        "sales may understate demand when products sell out."
    )

# ---------------------------------------------------------
# 4. Explore a column using a line, scatter, or histogram chart.
# ---------------------------------------------------------
st.subheader("2. Explore sales")

chart_column = st.selectbox(
    "Column to explore",
    options=numeric_columns,
    format_func=lambda column: labels[column],
    key="chart_column"
)

chart_type = st.selectbox(
    "Chart type",
    options=["Line", "Scatter", "Histogram"]
)

if chart_type == "Line":
    figure = px.line(
        product_df,
        x="date",
        y=chart_column,
        labels=labels,
        title=f"{product}: {labels[chart_column]} over time"
    )

    # Display gaps where values are unknown.
    figure.update_traces(connectgaps=False)

elif chart_type == "Scatter":
    horizontal_column = st.selectbox(
        "Horizontal-axis column",
        options=numeric_columns,
        index=numeric_columns.index("units_sold_7_days_ago"),
        format_func=lambda column: labels[column],
        key="scatter_x"
    )

    figure = px.scatter(
        product_df,
        x=horizontal_column,
        y=chart_column,
        hover_data=["date"],
        labels=labels,
        title=f"{product}: relationship between selected columns"
    )

else:
    figure = px.histogram(
        product_df,
        x=chart_column,
        nbins=30,
        labels=labels,
        title=f"{product}: distribution of {labels[chart_column]}"
    )

st.plotly_chart(figure, use_container_width=True)
st.caption(
    "Charts include listed closure days. Unknown values are not plotted."
)

# ---------------------------------------------------------
# 5. Choose a target and two distinct regression inputs.
# ---------------------------------------------------------
st.subheader("3. Predict sales")

st.info(
    "Suggested setup: predict Daily units sold using Units sold seven "
    "days earlier and Average units sold over preceding seven days."
)

target = st.selectbox(
    "Target: what do you want to predict?",
    options=numeric_columns,
    format_func=lambda column: labels[column],
    key="target"
)

# Exclude the target from the possible inputs.
input_options = [
    column for column in numeric_columns if column != target
]

preferred_inputs = [
    "units_sold_7_days_ago",
    "previous_7_day_avg"
]

# Place the historical-sales columns first in the input selectors.
input_options = (
    [column for column in preferred_inputs if column in input_options]
    + [column for column in input_options if column not in preferred_inputs]
)

input_1 = st.selectbox(
    "Input 1",
    options=input_options,
    format_func=lambda column: labels[column],
    key="input_1"
)

# Exclude Input 1 so the user cannot select the same input twice.
input_2 = st.selectbox(
    "Input 2",
    options=[column for column in input_options if column != input_1],
    format_func=lambda column: labels[column],
    key="input_2"
)

inputs = [input_1, input_2]

# Explain a common source of misleading results.
if (
    target == "quantity" and "estimated_revenue" in inputs
) or (
    target == "estimated_revenue" and "quantity" in inputs
):
    st.warning(
        "Estimated revenue is calculated from quantity and price. "
        "Using one to predict the other explores an arithmetic relationship, "
        "rather than forecasting unknown future sales."
    )

# ---------------------------------------------------------
# 6. Prepare training rows for the CURRENT column selection.
# Do not use the model_ready flag here: it was calculated
# specifically for the default historical-sales inputs.
# ---------------------------------------------------------
training_data = (
    product_df.loc[
        ~product_df["is_closed"],
        ["date", *inputs, target]
    ]
    .dropna(subset=inputs + [target])
    .sort_values("date")
)

st.write(
    f"Complete open-day records for this selection: "
    f"**{len(training_data):,}**"
)

# This classroom guardrail avoids fitting to just a few observations.
# It does not guarantee that the resulting model will be reliable.
if len(training_data) < 10:
    st.warning(
        "At least 10 complete records are needed in this app. "
        "Choose another product or different columns."
    )
    st.stop()

# Median price is constant within a product and is not a useful input.
constant_inputs = [
    column for column in inputs
    if training_data[column].nunique() < 2
]

if constant_inputs:
    names = ", ".join(labels[column] for column in constant_inputs)
    st.warning(
        f"These inputs do not vary for this product: {names}. "
        "Choose different inputs."
    )
    st.stop()

if training_data[target].nunique() < 2:
    st.warning(
        "The target does not vary for this product. Choose another target."
    )
    st.stop()

# ---------------------------------------------------------
# 7. Fit linear regression using the two selected inputs.
# ---------------------------------------------------------
X = training_data[inputs]
y = training_data[target]

model = LinearRegression()
model.fit(X, y)

st.caption(
    "The model is fitted to the available historical records. "
    "Its accuracy on future dates has not yet been evaluated."
)

# ---------------------------------------------------------
# 8. Accept user-entered values and display a prediction.
# ---------------------------------------------------------
with st.form("prediction_form"):
    st.write("Enter values for your prediction:")

    entered_values = {}

    for column in inputs:
        # Use the historical median as a reasonable starting value.
        entered_values[column] = st.number_input(
            labels[column],
            min_value=0.0,
            value=max(0.0, float(training_data[column].median())),
            format="%.2f",
            key=f"value_{product}_{column}"
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    # Preserve exactly the same column names and order as training.
    prediction_inputs = pd.DataFrame(
        [[entered_values[column] for column in inputs]],
        columns=inputs
    )

    prediction = float(model.predict(prediction_inputs)[0])

    st.metric(
        label=f"Predicted {labels[target]} — {product}",
        value=f"{prediction:,.2f}"
    )

    # Identify input values outside the observed historical range.
    outside_range = [
        column for column in inputs
        if not (
            training_data[column].min()
            <= entered_values[column]
            <= training_data[column].max()
        )
    ]

    if outside_range:
        names = ", ".join(labels[column] for column in outside_range)
        st.warning(
            f"Outside the historical input range: {names}. "
            "Predictions at these values may be less dependable."
        )

    # Do not silently hide impossible negative predictions.
    if prediction < 0:
        st.warning(
            "Linear regression returned a negative value. "
            "This is not a usable sales or revenue estimate."
        )
