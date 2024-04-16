import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib
import plotly.express as px

st.title('Predict Compressor Decay and Turbine Decay')

decision_tree_model = joblib.load("DecisionTree.pkl")
random_forest_model = joblib.load("RandomForest.pkl")
multi_task_model = joblib.load("MultiTaskModel.pkl")
bagging_model = joblib.load("BaggingRegressor.pkl")
grid_search_model = joblib.load("GridSearch.pkl")

models = {
    "Decision Tree Regressor": decision_tree_model,
    "Random Forest Regressor": random_forest_model,
    "Multi-Task Decision Tree Regressor": multi_task_model,
    "Bagging Regressor (Decision Trees)": bagging_model,
    "Grid Search Model": grid_search_model
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

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_data = preprocess_input_data(input_df)
    predictions = predict(selected_model, input_data)
    predictions_df = pd.DataFrame(predictions, columns=["Predicted Compressor Decay", "Predicted Turbine Decay"])
    result_df = pd.concat([input_df, predictions_df], axis=1)
    
    st.subheader('Select section to view:')
    section = st.selectbox('', ['Predictions for Compressor and Turbine Decay', 'Performance Metrics', 'Graphs'])
    
    if section == 'Predictions for Compressor and Turbine Decay':
        st.subheader('Predictions for Compressor Decay and Turbine Decay:')
        st.write(result_df)
    
    elif section == 'Performance Metrics':
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
    
    elif section == 'Graphs':
        
        st.subheader('Trend Analysis')
        fig = px.line(result_df, x=result_df.index, y=['Predicted Compressor Decay', 'Predicted Turbine Decay'], title='Trend Analysis')
        fig.update_layout(xaxis_title='Index', yaxis_title='Predicted Values', legend_title='Components')
        st.plotly_chart(fig)
        
        st.subheader('Histogram of Compressor Decay and Turbine Decay')
        fig_hist = px.histogram(result_df, x=['Predicted Compressor Decay', 'Predicted Turbine Decay'], marginal='rug')
        st.plotly_chart(fig_hist)
        
        st.subheader('Pairplot')
        fig_pairplot = px.scatter_matrix(result_df, dimensions=['Predicted Compressor Decay', 'Predicted Turbine Decay'], title='Pairplot')
        st.plotly_chart(fig_pairplot)
        
        st.subheader('Summary Statistics')
        st.write(result_df.describe())

else:
    st.info('Please upload a CSV file.')
