# 💼 Salary Prediction using Linear Regression
# Author: K. Siddhartha

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction | Linear Regression",
    layout="centered"
)

st.title("💼 Salary Prediction based on Skills")
st.write(
    "Interactive Machine Learning app that predicts salary using an "
    "**Explainable Linear Regression model**."
)

# --------------------------------------------------
# Synthetic Dataset (Deterministic)
# --------------------------------------------------
np.random.seed(42)

skills = np.array([[1], [2], [4], [6], [8], [9]])
salary = np.array([10, 20, 40, 60, 80, 90])

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    skills,
    salary,
    test_size=0.3,
    random_state=42
)

# --------------------------------------------------
# Model Training
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluation Metrics (TRUE model performance)
# --------------------------------------------------
y_pred_test = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# --------------------------------------------------
# User Input
# --------------------------------------------------
st.subheader("🎯 Predict Salary")

skill_input = st.slider(
    "Select Experience / Skill Level",
    min_value=1,
    max_value=10,
    value=3
)

predicted_salary = model.predict([[skill_input]])[0]

st.success(f"Predicted Salary: {predicted_salary:.2f} thousand")

# --------------------------------------------------
# Model Performance
# --------------------------------------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

col1.metric("Mean Squared Error", f"{mse:.2f}")
col2.metric("R² Score", f"{r2:.4f}")

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.subheader("📈 Regression Visualization")

skills_sorted = np.sort(skills, axis=0)
salary_line = model.predict(skills_sorted)

fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(
    skills,
    salary,
    s=80,
    label="Training Data"
)

ax.plot(
    skills_sorted,
    salary_line,
    linewidth=3,
    label="Regression Line"
)

ax.set_xlabel("Experience / Skill Level")
ax.set_ylabel("Salary (in thousands)")
ax.set_title("Salary Prediction using Linear Regression")
ax.legend()

st.pyplot(fig)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "Built with **Python**, **Scikit-learn**, and **Streamlit** by K. Siddhartha"
)
