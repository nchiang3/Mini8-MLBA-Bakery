import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Set up the page and explain its purpose.
st.set_page_config(page_title="How Much Should I Bake Tomorrow?")

st.title("How Much Should I Bake Tomorrow?")
st.caption("Explore bakery sales and predict sales for individual products.")

st.write(
    "This model predicts daily bakery sales to support production planning; "
    "overestimates may cause food waste, while underestimates may lead "
    "to stockouts and lost sales."
)

st.write(
    "Use historical sales patterns and your recent sales information to "
    "estimate tomorrow's sales and help plan tomorrow morning's baking. "
    "The default forecast assumes the bakery will be open tomorrow."
)

# Give the CSV columns readable labels and explanations.
labels = {
    "units_sold_today": "Units sold today",
    "last_7_day_avg": "Average daily units sold over the last seven days",
    "units_sold_tomorrow": "Units sold tomorrow"
}

tips = {
    "units_sold_today": (
        "The selected product's total sales for today, after sales are complete."
    ),
    "last_7_day_avg": (
        "Add this product's sales for today and the previous six calendar "
        "days, then divide by seven. Include known closure days as zero."
    ),
    "units_sold_tomorrow": (
        "The following calendar day's sales. This is normally the target "
        "you want to predict."
    )
}

# Upload and preview the prepared data.
uploaded_file = st.file_uploader(
    "Upload bakery_app_data.csv",
    type=["csv"],
    help="Use the finished CSV from your Bakery Data Prep notebook."
)

if uploaded_file is None:
    st.info("Upload your CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

required_columns = ["date", "article"] + list(labels)
if not set(required_columns).issubset(df.columns):
    st.error("Please upload bakery_app_data.csv with the five prepared columns.")
    st.stop()

df["date"] = pd.to_datetime(df["date"])

st.subheader("1. Preview your data")
st.dataframe(df.head(10))

# Select one product for both charts and regression.
products = sorted(df["article"].dropna().unique())

if not products:
    st.error("The file contains no products.")
    st.stop()

product = st.selectbox(
    "Choose a product",
    products,
    index=products.index("BAGUETTE") if "BAGUETTE" in products else 0,
    help="Charts and predictions use only this product's history."
)

product_df = (
    df.loc[df["article"] == product]
    .sort_values("date")
    .copy()
)

numeric_columns = list(labels)

# Let users choose a column and chart type.
st.subheader("2. Explore sales")

chart_column = st.selectbox(
    "Column to explore",
    numeric_columns,
    format_func=lambda column: labels[column],
    help="Choose the sales measure to display."
)

chart_type = st.selectbox(
    "Chart type",
    ["Line", "Scatter", "Histogram"],
    help=(
        "Line: changes over time. Scatter: relationships between columns. "
        "Histogram: how frequently different values occur."
    )
)

if chart_type == "Line":
    # Insert blank dates so excluded dates appear as gaps.
    chart_data = (
        product_df.set_index("date")[[chart_column]]
        .asfreq("D")
        .reset_index()
    )

    fig = px.line(
        chart_data,
        x="date",
        y=chart_column,
        labels=labels
    )
    fig.update_traces(connectgaps=False)

elif chart_type == "Scatter":
    x_column = st.selectbox(
        "Horizontal-axis column",
        [column for column in numeric_columns if column != chart_column],
        format_func=lambda column: labels[column],
        help="Compare this column with the measure selected above."
    )

    fig = px.scatter(
        product_df,
        x=x_column,
        y=chart_column,
        labels=labels,
        hover_data=["date"]
    )

else:
    fig = px.histogram(
        product_df,
        x=chart_column,
        nbins=20,
        labels=labels
    )

st.plotly_chart(fig)
st.caption(
    "Charts show only dates retained during preparation. "
    "Date refers to 'today' in each row; tomorrow's sales occur one day later."
)

# Let users choose the target and two different inputs.
st.subheader("3. Make a prediction")

target = st.selectbox(
    "Target: what do you want to predict?",
    numeric_columns,
    index=2,
    format_func=lambda column: labels[column],
    help="Choose Units sold tomorrow for the intended baking forecast."
)

input_options = [column for column in numeric_columns if column != target]

input_1 = st.selectbox(
    "Input 1",
    input_options,
    format_func=lambda column: labels[column],
    help="The first piece of information available when making a prediction."
)

input_2 = st.selectbox(
    "Input 2",
    [column for column in input_options if column != input_1],
    format_func=lambda column: labels[column],
    help="The second piece of information available when making a prediction."
)

inputs = [input_1, input_2]

if target != "units_sold_tomorrow":
    st.info(
        "This selection explores a historical relationship. "
        "To forecast tomorrow, choose Units sold tomorrow as the target."
    )

# Fit multiple linear regression using the selected columns.
data = product_df[inputs + [target]].dropna()

if len(data) < 5:
    st.warning("There are too few complete rows to fit this model.")
    st.stop()

X = data[inputs]
y = data[target]

model = LinearRegression()
model.fit(X, y)

st.metric("Training R²", f"{model.score(X, y):.2f}")
st.caption(
    "This measures fit to the historical training data, "
    "not accuracy on future sales."
)

# Collect values using tooltips specific to each input.
with st.form("prediction"):
    values = {}

    for column in inputs:
        values[column] = st.number_input(
            labels[column],
            min_value=0.0,
            value=max(0.0, float(data[column].median())),
            help=tips[column],
            key=f"{product}_{column}"
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    # Match the training column names AND their order.
    new_data = pd.DataFrame(
        [[values[column] for column in inputs]],
        columns=inputs
    )

    prediction = model.predict(new_data)[0]

    st.metric(
        f"Predicted {labels[target].lower()}",
        f"{prediction:.1f} units"
    )

    if prediction < 0:
        st.warning("A negative sales estimate is not usable for baking plans.")

    if any(
        values[column] < data[column].min()
        or values[column] > data[column].max()
        for column in inputs
    ):
        st.warning(
            "An input is outside the historical range. "
            "Treat this prediction with extra caution."
        )

    st.caption(
        "Predicted sales help guide production; they are not a guaranteed "
        "quantity needed. Past sales may understate demand when items sold out."
    )
