import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Linear Regression Explorer", layout="wide")

st.title("Linear Regression Visualizer")
st.markdown("Interactively explore how **linear regression** fits data by tuning the parameters below.")
st.divider()

# Sidebar controls
st.sidebar.header("Data Settings")
n_samples = st.sidebar.slider("Number of samples", 20, 200, 80)
noise = st.sidebar.slider("Noise level", 0.0, 10.0, 2.0, step=0.5)
true_slope = st.sidebar.number_input("True slope", value=2.0, step=0.5)
true_intercept = st.sidebar.number_input("True intercept", value=5.0, step=0.5)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.divider()
show_residuals = st.sidebar.toggle("Show residuals", value=True)
show_equation = st.sidebar.toggle("Show equation", value=True)

st.sidebar.divider()
st.sidebar.header("Custom Point")
enable_custom = st.sidebar.toggle("Add custom point", value=False)
custom_x = st.sidebar.number_input("Custom X", value=5.0, step=0.5)
custom_y = st.sidebar.number_input("Custom Y", value=15.0, step=0.5)

# Progress bar while generating data
st.caption("Generating data and fitting model...")
bar = st.progress(0)

# Generate base data
np.random.seed(int(seed))
X_base = np.random.uniform(0, 10, n_samples)
y_base = true_slope * X_base + true_intercept + np.random.normal(0, noise, n_samples)
bar.progress(30)

# Session state for clicked points
if "clicked_points" not in st.session_state:
    st.session_state.clicked_points = []

# Combine base + clicked points
clicked = st.session_state.clicked_points
X_all = np.concatenate([X_base, [p[0] for p in clicked]]) if clicked else X_base
y_all = np.concatenate([y_base, [p[1] for p in clicked]]) if clicked else y_base
bar.progress(50)

# Fit model
model = LinearRegression()
model.fit(X_all.reshape(-1, 1), y_all)
y_pred_all = model.predict(X_all.reshape(-1, 1))
bar.progress(70)

r2 = r2_score(y_all, y_pred_all)
rmse = np.sqrt(mean_squared_error(y_all, y_pred_all))
bar.progress(90)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fitted Slope", f"{model.coef_[0]:.3f}", f"{model.coef_[0] - true_slope:+.3f} vs true")
col2.metric("Fitted Intercept", f"{model.intercept_:.3f}", f"{model.intercept_ - true_intercept:+.3f} vs true")
col3.metric("R² Score", f"{r2:.4f}")
col4.metric("RMSE", f"{rmse:.4f}")
bar.progress(100)
bar.empty()

st.divider()

tab1, tab2, tab3 = st.tabs(["Regression Plot", "Data Table", "Model Info"])

with tab1:
    st.caption("Click anywhere on the chart to add a data point. The regression line will update instantly.")

    x_line = np.linspace(0, 10, 200)
    y_line_fit = model.coef_[0] * x_line + model.intercept_
    y_line_true = true_slope * x_line + true_intercept

    fig = go.Figure()

    # Base data points
    fig.add_trace(go.Scatter(
        x=X_base, y=y_base, mode="markers",
        marker=dict(color="steelblue", size=7, opacity=0.7),
        name="Data points"
    ))

    # Clicked points
    if clicked:
        cx, cy = zip(*clicked)
        fig.add_trace(go.Scatter(
            x=list(cx), y=list(cy), mode="markers",
            marker=dict(color="limegreen", size=10, symbol="star"),
            name="Clicked points"
        ))

    # Residual lines
    if show_residuals:
        for xi, yi, ypi in zip(X_all, y_all, y_pred_all):
            fig.add_shape(type="line", x0=xi, x1=xi, y0=yi, y1=ypi,
                          line=dict(color="orange", width=1, dash="dot"))

    # Fitted line
    fig.add_trace(go.Scatter(x=x_line, y=y_line_fit, mode="lines",
                             line=dict(color="crimson", width=2), name="Fitted line"))

    # True line
    fig.add_trace(go.Scatter(x=x_line, y=y_line_true, mode="lines",
                             line=dict(color="green", width=2, dash="dash"), name="True line"))

    # Custom point
    if enable_custom:
        custom_y_pred = model.coef_[0] * custom_x + model.intercept_
        fig.add_trace(go.Scatter(
            x=[custom_x], y=[custom_y], mode="markers",
            marker=dict(color="magenta", size=12, symbol="circle"),
            name=f"Custom point ({custom_x}, {custom_y})"
        ))
        fig.add_trace(go.Scatter(
            x=[custom_x], y=[custom_y_pred], mode="markers",
            marker=dict(color="magenta", size=8, symbol="diamond"),
            name=f"Predicted y = {custom_y_pred:.3f}"
        ))
        fig.add_shape(type="line", x0=custom_x, x1=custom_x, y0=custom_y, y1=custom_y_pred,
                      line=dict(color="magenta", width=1.5, dash="dot"))

    fig.update_layout(
        xaxis_title="X", yaxis_title="y",
        title="Linear Regression Fit  —  click the plot to add a point",
        clickmode="event",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="lr_plot")

    # Handle click event
    if event and event.selection and event.selection.get("points"):
        pt = event.selection["points"][0]
        new_point = (round(pt["x"], 3), round(pt["y"], 3))
        if new_point not in st.session_state.clicked_points:
            st.session_state.clicked_points.append(new_point)
            st.rerun()

    # Clicked points controls
    if clicked:
        st.success(f"{len(clicked)} custom point(s) added via clicks.")
        if st.button("Clear clicked points"):
            st.session_state.clicked_points = []
            st.rerun()

    # Custom point metrics
    if enable_custom:
        custom_y_pred = model.coef_[0] * custom_x + model.intercept_
        residual = custom_y - custom_y_pred
        c1, c2, c3 = st.columns(3)
        c1.metric("Custom X", f"{custom_x}")
        c2.metric("Predicted y", f"{custom_y_pred:.3f}")
        c3.metric("Residual (actual - predicted)", f"{residual:.3f}")

with tab2:
    df = pd.DataFrame({
        "X": X_all, "y (actual)": y_all, "y (predicted)": y_pred_all, "residual": y_all - y_pred_all,
        "source": ["base"] * len(X_base) + ["clicked"] * len(clicked)
    })
    st.dataframe(df.round(4), use_container_width=True)

with tab3:
    if show_equation:
        st.subheader("Fitted Equation")
        st.latex(rf"y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")
        st.subheader("True Equation")
        st.latex(rf"y = {true_slope}x + {true_intercept} + \epsilon, \quad \epsilon \sim \mathcal{{N}}(0, {noise}^2)")

    with st.expander("What is Linear Regression?"):
        st.markdown("""
Linear regression models the relationship between a dependent variable **y** and an independent variable **X** as:

> **y = β₀ + β₁X + ε**

- **β₀** — intercept (value of y when X = 0)
- **β₁** — slope (change in y per unit change in X)
- **ε** — random noise

The model is fit by minimizing the **sum of squared residuals** (OLS).
        """)
# page config added
# sidebar sliders added
# numpy data generation
# sklearn model fit
# r2 and rmse metrics
# columns layout
# tabs added
# plotly scatter
# fitted and true lines
# residual lines
# equation toggle
# latex equations
# custom point sidebar
# custom point plot
# custom point metrics
