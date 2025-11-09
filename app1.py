import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib

st.title('Predict Compressor Decay and Turbine Decay')

decision_tree_model = joblib.load("DecisionTree.pkl")
random_forest_model = joblib.load("RandomForest.pkl")
multi_task_model = joblib.load("MultiTaskModel.pkl")
bagging_model = joblib.load("BaggingRegressor.pkl")
grid_search = joblib.load("GridSearch.pkl")

models = {
    "Decision Tree Regressor": decision_tree_model,
    "Random Forest Regressor": random_forest_model,
    "Multi-Task Decision Tree Regressor": multi_task_model,
    "Bagging Regressor (Decision Trees)": bagging_model,
    "Grid Search Model": grid_search
}

selected_model = st.sidebar.selectbox('Select Model', list(models.keys()))

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

def preprocess_input_data(df):
    df = df.drop("index", axis=1)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    normalizer = Normalizer()
    df_normalized = normalizer.transform(df_scaled)
    return df_normalized

def predict(selected_model, input_data):
    model = models[selected_model]
    predictions = model.predict(input_data)
    return predictions

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_data = preprocess_input_data(input_df)
    predictions = predict(selected_model, input_data)
    predictions_df = pd.DataFrame(predictions, columns=["Predicted Compressor Decay", "Predicted Turbine Decay"])
    result_df = pd.concat([input_df, predictions_df], axis=1)
    st.subheader('Predictions for Compressor Decay and Turbine Decay:')
    st.write(result_df)
else:
    st.info('Please upload a CSV file.')

data = pd.read_csv("data.csv")
data = data.drop("index", axis=1)
y = data[["compressor decay", "turbine decay"]]
X = data.drop(["compressor decay", "turbine decay"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

normalizer = Normalizer()
X_train_normalized = normalizer.fit_transform(X_train_scaled)
X_test_normalized = normalizer.transform(X_test_scaled)

if selected_model in models:
    model = models[selected_model]
    predictions = model.predict(X_test_normalized)
    mse = mean_squared_error(predictions, y_test)
    mae = mean_absolute_error(predictions, y_test)
    mape = mean_absolute_percentage_error(predictions, y_test)
    r2 = r2_score(predictions, y_test)
    st.subheader(f'{selected_model} Performance Metrics:')
    st.write(f"MSE: {mse:.8f}")
    st.write(f"MAE: {mae:.5f}")
    st.write(f"MAPE: {mape:.5f}")
    st.write(f"r2 Score: {r2:.4f}")
else:
    st.write("Selected model not found.")
