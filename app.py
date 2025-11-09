import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib

# Title
st.title('Hybrid Model Deployment with Streamlit')

# Load the trained models
decision_tree_model = joblib.load("DecisionTree.pkl")
random_forest_model = joblib.load("RandomForest.pkl")
multi_task_model = joblib.load("MultiTaskModel.pkl")
bagging_model = joblib.load("BaggingRegressor.pkl")

# Load data
data = pd.read_csv("data.csv")

# Data preprocessing steps


# Drop unnecessary columns
data = data.drop("index", axis=1)

# Define X and y variables
y = data[["compressor decay", "turbine decay"]]
X = data.drop(["compressor decay", "turbine decay"], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalization
normalizer = Normalizer()
X_train_normalized = normalizer.fit_transform(X_train_scaled)
X_test_normalized = normalizer.transform(X_test_scaled)

# Define available models
models = {
    "Decision Tree Regressor": decision_tree_model,
    "Random Forest Regressor": random_forest_model,
    "Multi-Task Decision Tree Regressor": multi_task_model,
    "Bagging Regressor (Decision Trees)": bagging_model
}

# Model selection dropdown
selected_model = st.selectbox('Select Model', list(models.keys()))

# Calculate predictions based on selected model
if selected_model in models:
    # Perform predictions using the selected model
    model = models[selected_model]
    predictions = model.predict(X_test_normalized)
    
    # Calculate performance metrics for the selected model
    mse = mean_squared_error(predictions, y_test)
    mae = mean_absolute_error(predictions, y_test)
    mape = mean_absolute_percentage_error(predictions, y_test)
    r2 = r2_score(predictions, y_test)
    # Display performance metrics
    st.subheader(f'{selected_model} Performance Metrics:')
    st.write(f"MSE: {mse:.8f}")
    st.write(f"MAE: {mae:.5f}")
    st.write(f"MAPE: {mape:.5f}")
    st.write(f"r2 Score: {r2:.4f}")
    
else:
    st.write("Selected model not found.")