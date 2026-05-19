import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet   # Prophet imported here

# -------------------
# Page settings
# -------------------
st.set_page_config(
    page_title="AI Finance Tracker",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI Personal Finance Tracker")
st.caption("Track expenses • Predict spending • Improve savings")

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main{
    background-color:#0e1117;
}
h1{
    color:#00c4ff;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
page = st.sidebar.radio("Navigation", ["Dashboard","Prediction"])
uploaded_file = st.file_uploader("Upload expense CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ==================
    # DASHBOARD
    # ==================
    if page == "Dashboard":

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df["Rolling_Avg"] = df["Amount"].rolling(window=3).mean()

        st.subheader("📈 Spending Trends")
        st.line_chart(df.set_index("Date")[["Amount","Rolling_Avg"]])

        category_threshold = st.number_input("Set category alert threshold (₹)", value=2000)
        alerts = df.groupby("Category")["Amount"].sum()
        for cat, amt in alerts.items():
            if amt > category_threshold:
                st.warning(f"⚠️ {cat} exceeded threshold: ₹{amt}")

        planned_savings = st.number_input("Planned Savings (₹)", value=5000)

        total = df["Amount"].sum()
        average = df["Amount"].mean()
        highest = df["Amount"].max()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Spending", f"₹{total}")
        col2.metric("Average Expense", f"₹{average:.0f}")
        col3.metric("Highest Expense", f"₹{highest}")

        budget = st.number_input("Monthly Budget", value=1000)
        actual_savings = budget - total
        st.metric("Savings Gap", f"₹{planned_savings-actual_savings}")

        if total > budget:
            st.error(f"Exceeded by ₹{total-budget}")
        else:
            st.success(f"Remaining ₹{budget-total}")

        st.subheader("Expense Distribution")
        pie = px.pie(
            df,
            names="Category",
            values="Amount",
            hole=0.6,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        pie.update_layout(title="Expense Categories", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(pie, use_container_width=True)

        # --- Monthly Breakdown Bar Chart ---
        monthly_expenses = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum().reset_index()
        monthly_expenses["Date"] = monthly_expenses["Date"].dt.to_timestamp()

        st.subheader("📅 Monthly Expense Totals")
        fig_bar = px.bar(
            monthly_expenses,
            x="Date",
            y="Amount",
            text="Amount",
            color="Amount",
            color_continuous_scale="Blues",
            title="Monthly Expenses"
        )
        fig_bar.update_traces(texttemplate="₹%{text}", textposition="outside")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(monthly_expenses)

        st.dataframe(df)

    # ==================
    # PREDICTION PAGE
    # ==================
    elif page == "Prediction":

        st.subheader("🔮 Monthly Expense Forecast with Prophet")

        # Convert to datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Aggregate monthly totals
        monthly = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum().reset_index()
        monthly["Date"] = monthly["Date"].dt.to_timestamp()

        # Prepare for Prophet
        df_prophet = monthly.rename(columns={"Date":"ds","Amount":"y"})

        # Fit model
        model = Prophet()
        model.fit(df_prophet)

        # Forecast next 3 months (use 'ME' for month end)
        future = model.make_future_dataframe(periods=3, freq="ME")
        forecast = model.predict(future)

        # Show prediction
        st.success(f"Predicted next month expense ₹{forecast.iloc[-1]['yhat']:.0f}")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_prophet["ds"], y=df_prophet["y"],
            mode="lines+markers", name="Actual"
        ))
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            mode="lines+markers", name="Predicted"
        ))
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat_lower"],
            mode="lines", line=dict(dash="dot", color="gray"),
            name="Lower Bound"
        ))
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat_upper"],
            mode="lines", line=dict(dash="dot", color="gray"),
            name="Upper Bound"
        ))

        st.plotly_chart(fig, use_container_width=True)


