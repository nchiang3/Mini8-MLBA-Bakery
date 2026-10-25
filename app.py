import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="How Much Should I Bake Tomorrow?",
    layout="wide"
)

st.title("How Much Should I Bake Tomorrow?")

st.write(
    "Explore historical bakery sales and predict "
    "how many units to produce tomorrow using "
    "linear regression."
)

st.info(
    "This model helps estimate tomorrow's bakery demand. "
    "If the prediction is too high, unsold products may "
    "be wasted. If it is too low, the bakery may sell "
    "out and lose potential sales."
)

WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

LABELS = {
    "units_sold_today": "Units sold today",
    "last_7_day_avg": "Last 7-day sales average",
    "units_sold_tomorrow": "Units sold tomorrow"
}


# --------------------------------------------------
# MODEL FEATURE ENGINEERING
# --------------------------------------------------

def build_features(data):
    """
    Prepare the numeric inputs and weekday
    interaction terms for linear regression.

    Weekday interactions allow different
    relationships between today's sales and
    tomorrow's sales for each weekday.
    """

    X = data[
        ["units_sold_today", "last_7_day_avg"]
    ].copy()

    weekdays = data["date"].dt.day_name()

    for day in WEEKDAYS:
        flag = (weekdays == day).astype(int)

        X[f"day_{day}"] = flag

        X[f"sales_x_day_{day}"] = (
            data["units_sold_today"] * flag
        )

    return X


def make_prediction_input(
    today_sales, seven_day_avg, weekday, columns
):
    """
    Create a prediction row with the same
    feature columns used during training.
    """

    new_X = pd.DataFrame(
        0.0,
        index=[0],
        columns=columns
    )

    new_X["units_sold_today"] = today_sales
    new_X["last_7_day_avg"] = seven_day_avg

    new_X[f"day_{weekday}"] = 1
    new_X[f"sales_x_day_{weekday}"] = today_sales

    return new_X


# --------------------------------------------------
# 1. UPLOAD DATA
# --------------------------------------------------

st.header("1. Upload Bakery Sales Data")

uploaded_file = st.file_uploader(
    "Upload your prepared bakery sales CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

required_columns = [
    "date",
    "article",
    "units_sold_today",
    "last_7_day_avg",
    "units_sold_tomorrow"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    st.error(
        f"The uploaded CSV is missing: {missing}"
    )
    st.stop()

df["date"] = pd.to_datetime(
    df["date"], errors="coerce"
)

for col in LABELS:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

df = df.dropna(subset=["date", "article"])

st.write("Preview of uploaded data:")
st.dataframe(
    df.head(10),
    use_container_width=True
)

products = sorted(df["article"].unique())

default_index = (
    products.index("BAGUETTE")
    if "BAGUETTE" in products else 0
)

product = st.selectbox(
    "Choose a bakery product",
    products,
    index=default_index
)

product_df = (
    df[df["article"] == product]
    .sort_values("date")
    .copy()
)


# --------------------------------------------------
# 2. EXPLORE HISTORICAL SALES
# --------------------------------------------------

st.header("2. Explore Historical Sales")

numeric_columns = list(LABELS.keys())

col1, col2 = st.columns(2)

with col1:
    chart_column = st.selectbox(
        "Choose a numeric column",
        numeric_columns,
        format_func=lambda x: LABELS[x]
    )

with col2:
    chart_type = st.selectbox(
        "Choose a chart type",
        ["Line", "Scatter", "Histogram"]
    )

if chart_type == "Line":

    fig = px.line(
        product_df,
        x="date",
        y=chart_column,
        title=f"{product}: Sales Over Time"
    )

elif chart_type == "Scatter":

    x_options = [
        c for c in numeric_columns
        if c != chart_column
    ]

    x_column = st.selectbox(
        "Choose the X-axis",
        x_options,
        format_func=lambda x: LABELS[x]
    )

    fig = px.scatter(
        product_df,
        x=x_column,
        y=chart_column,
        title=f"{product}: Sales Relationship"
    )

else:

    fig = px.histogram(
        product_df,
        x=chart_column,
        nbins=20,
        title=f"{product}: Sales Distribution"
    )

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 3. TRAIN REGRESSION MODEL
# --------------------------------------------------

st.header("3. Predict Tomorrow's Production")

st.write(
    "This forecast uses today's sales, the recent "
    "7-day sales average, and historical weekday "
    "patterns to estimate tomorrow's demand."
)

data = (
    product_df.dropna(
        subset=[
            "units_sold_today",
            "last_7_day_avg",
            "units_sold_tomorrow"
        ]
    )
    .sort_values("date")
    .reset_index(drop=True)
)

if len(data) < 25:
    st.warning(
        "There are not enough complete historical "
        "observations to train this model."
    )
    st.stop()

# Prepare the regression inputs.
X = build_features(data)
y = data["units_sold_tomorrow"]

# Chronological train/test split.
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Fit linear regression.
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on unseen data.
train_r2 = model.score(X_train, y_train)

test_predictions = model.predict(X_test)

test_r2 = r2_score(
    y_test, test_predictions
)

test_mae = mean_absolute_error(
    y_test, test_predictions
)


# --------------------------------------------------
# 4. USER FORECAST INPUTS
# --------------------------------------------------

st.subheader("Enter Today's Sales Information")

with st.form("forecast_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        today_sales = st.number_input(
            "Units sold today",
            min_value=0,
            value=30,
            step=1
        )

    with col2:
        seven_day_avg = st.number_input(
            "Last 7-day sales average",
            min_value=0.0,
            value=28.0,
            step=1.0
        )

    with col3:
        weekday = st.selectbox(
            "Today's day of week",
            WEEKDAYS
        )

    predict_button = st.form_submit_button(
        "Calculate Tomorrow's Production",
        type="primary"
    )


# --------------------------------------------------
# 5. FORECAST RESULTS
# --------------------------------------------------

if predict_button:

    new_X = make_prediction_input(
        today_sales,
        seven_day_avg,
        weekday,
        X.columns
    )

    raw_prediction = model.predict(new_X)[0]

    # Production cannot be negative.
    recommended_units = max(
        0, round(float(raw_prediction))
    )

    # Identify tomorrow's weekday.
    tomorrow_index = (
        WEEKDAYS.index(weekday) + 1
    ) % 7

    tomorrow = WEEKDAYS[tomorrow_index]

    st.divider()

    st.subheader("Your Production Recommendation")

    st.write(
        "Based on historical sales patterns "
        "and your input data:"
    )

    st.markdown(
        f"### You should make approximately "
        f"{recommended_units:,} {product.lower()} "
        f"tomorrow ({tomorrow})."
    )

    st.write(
        f"Historical predictions have been off by "
        f"an average of **{test_mae:.2f} units**."
    )

    st.divider()

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Test R²",
            f"{test_r2:.3f}"
        )

        st.caption(
            "Measures how well the model explains "
            "sales variation in the test dataset."
        )

    with col2:
        st.metric(
            "Average Prediction Error (MAE)",
            f"±{test_mae:.2f} units"
        )

        st.caption(
            "Average absolute difference between "
            "predicted and actual sales in the "
            "test dataset. This is not a "
            "guaranteed prediction interval."
        )

    with st.expander("More about this regression"):

        st.write(
            f"Product: {product}"
        )

        st.write(
            f"Training observations: {len(X_train)}"
        )

        st.write(
            f"Testing observations: {len(X_test)}"
        )

        st.write(
            f"Training R²: {train_r2:.3f}"
        )

        st.write(
            "Model: Linear regression using today's "
            "sales, the seven-day average, and "
            "weekday interaction variables."
        )

        st.write(
            "The model uses 80% of historical "
            "observations for training and the "
            "most recent 20% for testing."
        )

    st.caption(
        "Forecasts are estimates based on historical "
        "sales. Actual demand may differ, and "
        "historical sales may understate demand "
        "when products have sold out."
    )
