import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("💼 Salary Prediction Based on Skills")
st.write("Predict salary (in thousands) using Linear Regression.")

# skills in coding languages based on percentage
skills = np.array([[1], [2], [4], [6], [8], [9]])

# salary in thousands
salary = np.array([10, 20, 40, 60, 80, 90])

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    skills, salary, test_size=0.3, random_state=42
)

# train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# user input
skill_input = st.slider("Select your skill level", 1, 10, 3)
predicted_salary = model.predict([[skill_input]])[0]

st.subheader("📈 Prediction")
st.success(f"Predicted salary: {predicted_salary:.2f} thousand")

# evaluate the model
y_pred = model.predict(X_test)

st.subheader("📊 Model Performance")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# visualization (no dotted lines)
skills_sorted = np.sort(skills, axis=0)
salary_line = model.predict(skills_sorted)

fig, ax = plt.subplots()
ax.scatter(skills, salary, color="orange", label="my coding skills")
ax.plot(skills_sorted, salary_line, color="green", label="predicting the salary")
ax.set_title("salary based on skills")
ax.set_xlabel("skills")
ax.set_ylabel("salary in thousands")
ax.legend()

st.pyplot(fig)
