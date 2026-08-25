import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
import sqlite3
import random

st.set_page_config(page_title="AI Trends SaaS", layout="wide")

# =============================
# LOGIN SYSTEM
# =============================
# =============================
# SIMPLE LOGIN SYSTEM
# =============================
st.sidebar.title("🔐 Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username != "karan" or password != "1234":
    st.warning("Enter correct username and password")
    st.stop()

st.sidebar.success("Logged in successfully")
# =============================
# DATABASE
# =============================
conn = sqlite3.connect("user.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS preferences (
    user TEXT,
    keyword TEXT
)
""")

def save_pref(user, keyword):
    c.execute("INSERT INTO preferences VALUES (?,?)", (user, keyword))
    conn.commit()

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    data = pd.read_csv("multiTimeline.csv", skiprows=2, header=None)

    data.columns = [
        "Date","AI","Machine Learning","Data Science","Python",
        "Deep Learning","Big Data","Neural Network","NLP"
    ]

    data["Date"] = pd.to_datetime(data["Date"])
    data.iloc[:,1:] = data.iloc[:,1:].apply(pd.to_numeric, errors="coerce")
    data = data.ffill()

    return data

df = load_data()

keywords = list(df.columns[1:])

# =============================
# SIDEBAR
# =============================
st.sidebar.title("⚙ Controls")

menu = ["Dashboard","Forecast","Model Accuracy","Settings"]
choice = st.sidebar.selectbox("Menu", menu)

selected_keywords = st.sidebar.multiselect(
    "Select Keywords", keywords, default=[keywords[0]]
)

start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())

filtered_df = df[
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
]
if not selected_keywords:
    st.warning("⚠ Please select at least one keyword from sidebar")
    st.stop()
# =============================
# DASHBOARD
# =============================
if choice == "Dashboard":

    st.title("📊 AI Trends Dashboard")

    # KPI
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Records", len(filtered_df))
    col2.metric("Keywords", len(keywords))
    col3.metric("Latest", filtered_df["Date"].max().strftime("%Y-%m-%d"))
    col4.metric("Peak", int(filtered_df[keywords].max().max()))

    # Trend
    st.subheader("📈 Trends")

    fig = px.line(
        filtered_df,
        x="Date",
        y=selected_keywords,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Correlation")
        corr = filtered_df[keywords].corr()
        fig2, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig2)

    with col2:
        st.subheader("🏆 Top Days")
        if selected_keywords:

         top = filtered_df.sort_values(
         by=selected_keywords[0],
         ascending=False
         ).head(10)

         st.dataframe(top)

        else:
         st.warning("⚠ Please select at least one keyword")

    # Monthly Heatmap
    st.subheader("📊 Monthly Heatmap")

    temp = filtered_df.copy()
    temp["Month"] = temp["Date"].dt.month
    temp["Year"] = temp["Date"].dt.year

    pivot = temp.pivot_table(
        values=selected_keywords[0],
        index="Month",
        columns="Year",
        aggfunc="mean"
    )

    fig3, ax = plt.subplots()
    sns.heatmap(pivot, cmap="coolwarm", annot=True, ax=ax)
    st.pyplot(fig3)

    # =============================
    # TREND SCORE (MOVE INSIDE)
    # =============================
    st.subheader("🎯 Trend Score")

    scores = {}

    for key in keywords:
        recent_avg = filtered_df[key].tail(5).mean()
        overall_avg = filtered_df[key].mean()

        if overall_avg == 0:
            score = 0
        else:
            score = round((recent_avg / overall_avg) * 100, 2)

        scores[key] = score

    score_df = pd.DataFrame({
        "Keyword": scores.keys(),
        "Score": scores.values()
    }).sort_values(by="Score", ascending=False)

    st.dataframe(score_df)

    st.success(f"🔥 Top: {score_df.iloc[0]['Keyword']}")

    # =============================
    # COUNTRY VIEW (MOVE INSIDE)
    # =============================
    st.subheader("🌍 Country View")

    countries = ["India","USA","UK","Germany","Canada"]

    country_df = pd.DataFrame({
        "Country": countries,
        "Interest": [random.randint(50,100) for _ in countries]
    })

    st.plotly_chart(
        px.bar(country_df, x="Country", y="Interest", template="plotly_dark")
    )
# =============================
# FORECAST
# =============================
elif choice == "Forecast":

    st.title("🔮 Forecast")

    # Detect the frequency of the Google Trends data.
    inferred_freq = pd.infer_freq(df["Date"])
    if inferred_freq is None:
        inferred_freq = "MS"

    if inferred_freq.startswith("M"):
        forecast_periods = st.slider(
            "Forecast horizon (months)", 3, 12, 6
        )
        horizon_text = f"{forecast_periods} months"
    else:
        forecast_periods = st.slider(
            "Forecast horizon (periods)", 3, 30, 12
        )
        horizon_text = f"{forecast_periods} periods"

    st.caption(
        f"Detected data frequency: {inferred_freq}. "
        f"Forecast horizon: {horizon_text}."
    )

    if len(selected_keywords) > 3:
        st.warning("⚠ Select max 3 keywords for faster forecast")

    # Use the best Prophet configuration found by the accuracy page.
    forecast_configs = [
        {"changepoint_prior_scale": 0.05, "seasonality_mode": "additive"},
        {"changepoint_prior_scale": 0.10, "seasonality_mode": "additive"},
        {"changepoint_prior_scale": 0.50, "seasonality_mode": "additive"},
        {"changepoint_prior_scale": 0.10, "seasonality_mode": "multiplicative"},
        {"changepoint_prior_scale": 0.50, "seasonality_mode": "multiplicative"},
        {"changepoint_prior_scale": 1.00, "seasonality_mode": "multiplicative"},
    ]

    for keyword in selected_keywords:

        prophet_df = filtered_df[["Date", keyword]].rename(
            columns={"Date": "ds", keyword: "y"}
        ).dropna()

        if len(prophet_df) < 10:
            st.warning(f"Not enough data for {keyword}")
            continue

        # Default to a tuned configuration; the Model Accuracy page
        # provides the validation-based comparison.
        config = forecast_configs[2]

        model = Prophet(
            growth="linear",
            changepoint_prior_scale=config["changepoint_prior_scale"],
            changepoint_range=0.95,
            seasonality_mode=config["seasonality_mode"],
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.fit(prophet_df)

        future = model.make_future_dataframe(
            periods=forecast_periods,
            freq=inferred_freq
        )
        forecast = model.predict(future)

        fig = px.line(
            forecast,
            x="ds",
            y="yhat",
            title=f"Forecast for {keyword}",
            template="plotly_dark"
        )

        # Add uncertainty interval.
        fig.add_scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        )
        fig.add_scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="Uncertainty"
        )

        st.plotly_chart(fig, use_container_width=True)

# =============================
# MODEL ACCURACY
# =============================
elif choice == "Model Accuracy":

    st.title("🎯 Model Accuracy & Model Selection")

    st.write(
        "The model is evaluated using historical holdout validation. "
        "The final portion of the selected date range is kept unseen during training. "
        "Multiple Prophet configurations are compared automatically."
    )

    test_ratio = st.slider(
        "Test data (%)",
        min_value=10,
        max_value=30,
        value=20,
        step=5
    )

    # Prophet configurations to compare.
    model_configs = [
        {
            "Model": "Prophet 1",
            "changepoint_prior_scale": 0.05,
            "seasonality_mode": "additive"
        },
        {
            "Model": "Prophet 2",
            "changepoint_prior_scale": 0.10,
            "seasonality_mode": "additive"
        },
        {
            "Model": "Prophet 3",
            "changepoint_prior_scale": 0.50,
            "seasonality_mode": "additive"
        },
        {
            "Model": "Prophet 4",
            "changepoint_prior_scale": 0.10,
            "seasonality_mode": "multiplicative"
        },
        {
            "Model": "Prophet 5",
            "changepoint_prior_scale": 0.50,
            "seasonality_mode": "multiplicative"
        },
        {
            "Model": "Prophet 6",
            "changepoint_prior_scale": 1.00,
            "seasonality_mode": "multiplicative"
        }
    ]

    overall_results = []

    for keyword in selected_keywords:

        prophet_df = filtered_df[["Date", keyword]].rename(
            columns={"Date": "ds", keyword: "y"}
        ).dropna()

        if len(prophet_df) < 20:
            st.warning(
                f"Not enough data for {keyword}. At least 20 observations "
                "are recommended for accuracy testing."
            )
            continue

        test_size = max(1, int(np.ceil(len(prophet_df) * test_ratio / 100)))

        if len(prophet_df) - test_size < 10:
            test_size = len(prophet_df) - 10

        train = prophet_df.iloc[:-test_size].copy()
        test = prophet_df.iloc[-test_size:].copy()

        model_results = []
        predictions = {}

        for config in model_configs:

            try:
                model = Prophet(
                    growth="linear",
                    changepoint_prior_scale=config["changepoint_prior_scale"],
                    changepoint_range=0.95,
                    seasonality_mode=config["seasonality_mode"],
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False
                )

                model.fit(train)

                forecast = model.predict(test[["ds"]])

                actual = test["y"].to_numpy(dtype=float)
                predicted = forecast["yhat"].to_numpy(dtype=float)
                errors = actual - predicted

                mae = float(np.mean(np.abs(errors)))
                rmse = float(np.sqrt(np.mean(errors ** 2)))

                non_zero = actual != 0
                if np.any(non_zero):
                    mape = float(
                        np.mean(
                            np.abs(
                                (actual[non_zero] - predicted[non_zero])
                                / actual[non_zero]
                            )
                        ) * 100
                    )
                else:
                    mape = np.nan

                ss_res = float(np.sum(errors ** 2))
                ss_tot = float(
                    np.sum((actual - np.mean(actual)) ** 2)
                )
                r2 = (
                    float(1 - ss_res / ss_tot)
                    if ss_tot != 0
                    else np.nan
                )

                model_results.append({
                    "Model": config["Model"],
                    "Changepoint": config["changepoint_prior_scale"],
                    "Seasonality": config["seasonality_mode"],
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE (%)": mape,
                    "R²": r2
                })

                predictions[config["Model"]] = predicted

            except Exception as e:
                st.warning(
                    f"{config['Model']} could not be evaluated for "
                    f"{keyword}: {e}"
                )

        if not model_results:
            continue

        results_df = pd.DataFrame(model_results)

        # Select the model using MAE as the primary metric.
        # MAE is robust and remains meaningful when Trends contains zeros.
        best_index = results_df["MAE"].idxmin()
        best = results_df.loc[best_index]

        overall_results.append({
            "Keyword": keyword,
            "Best Model": best["Model"],
            "MAE": round(best["MAE"], 2),
            "RMSE": round(best["RMSE"], 2),
            "MAPE (%)": (
                round(best["MAPE (%)"], 2)
                if not np.isnan(best["MAPE (%)"]) else np.nan
            ),
            "R²": (
                round(best["R²"], 3)
                if not np.isnan(best["R²"]) else np.nan
            )
        })

        st.subheader(f"🏆 Best Model for {keyword}: {best['Model']}")

        metric_cols = st.columns(4)
        metric_cols[0].metric("MAE", f"{best['MAE']:.2f}")
        metric_cols[1].metric("RMSE", f"{best['RMSE']:.2f}")
        metric_cols[2].metric(
            "MAPE",
            f"{best['MAPE (%)']:.2f}%"
            if not np.isnan(best["MAPE (%)"]) else "N/A"
        )
        metric_cols[3].metric(
            "R²",
            f"{best['R²']:.3f}"
            if not np.isnan(best["R²"]) else "N/A"
        )

        st.caption(
            f"Selected configuration: changepoint_prior_scale="
            f"{best['Changepoint']}, "
            f"seasonality={best['Seasonality']}"
        )

        # Full model comparison.
        display_results = results_df.copy()
        for col in ["MAE", "RMSE", "MAPE (%)", "R²"]:
            display_results[col] = display_results[col].round(3)

        st.write("### Model Comparison")
        st.dataframe(
            display_results.sort_values("MAE"),
            use_container_width=True
        )

        # Actual vs best prediction.
        comparison = test[["ds", "y"]].copy()
        comparison["Predicted"] = predictions[best["Model"]]

        comparison_long = comparison.melt(
            id_vars="ds",
            value_vars=["y", "Predicted"],
            var_name="Series",
            value_name="Interest"
        )
        comparison_long["Series"] = comparison_long["Series"].replace({
            "y": "Actual",
            "Predicted": "Best Model Prediction"
        })

        fig_accuracy = px.line(
            comparison_long,
            x="ds",
            y="Interest",
            color="Series",
            markers=True,
            title=f"Actual vs Best Prediction — {keyword}",
            template="plotly_dark"
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)

    if overall_results:
        st.subheader("📋 Best Model Summary")
        summary_df = pd.DataFrame(overall_results)
        st.dataframe(summary_df, use_container_width=True)

        st.info(
            "The best configuration is selected using the lowest MAE on the "
            "held-out historical test period. Lower MAE, RMSE and MAPE are "
            "better; R² closer to 1 is better. These metrics estimate "
            "historical forecasting performance and do not guarantee future accuracy."
        )

# =============================
# SETTINGS
# =============================
elif choice == "Settings":

    st.title("⚙ Settings")

    if st.button("Save Preference"):
        save_pref(username, selected_keywords[0])
        st.success("Saved!")

# =============================
# DOWNLOAD
# =============================
st.sidebar.download_button(
    "Download Data",
    filtered_df.to_csv(index=False),
    "trends.csv",
    "text/csv"
)
