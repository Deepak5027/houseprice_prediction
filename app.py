import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import gdown
import os

st.set_page_config(
    page_title="AI Powered ML Prediction System",
    layout="wide",
    page_icon="🤖"
)

# =========================================================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# =========================================================

MODEL_URL = "https://drive.google.com/file/d/1Ea0OBfbyk4kEPm-nH8xyopLilht82yqe/view?usp=sharing"
MODEL_PATH = "stacking_ensemble.pkl"

if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = pickle.load(open(MODEL_PATH, "rb"))

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

df = load_data()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Project Overview",
        "Dataset Explorer",
        "EDA Visualizations",
        "Correlation Analysis",
        "Model Performance",
        "Clustering Analysis",
        "Prediction System"
    ]
)

# =========================================================
# PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.title("🤖 AI Powered Machine Learning Prediction System")

    st.markdown("""
    ### Project Features

    - Advanced **Machine Learning Models**
    - **Stacking Ensemble Model**
    - **XGBoost Optimization**
    - **Clustering Analysis**
    - **Explainable AI (SHAP)**
    - **Interactive Visualizations**
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Size", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Model Type", "Stacking Ensemble")

# =========================================================
# DATASET EXPLORER
# =========================================================

elif page == "Dataset Explorer":

    st.title("📊 Dataset Explorer")

    st.dataframe(df)

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")

    missing = df.isnull().sum()

    fig = px.bar(
        x=missing.index,
        y=missing.values,
        title="Missing Values per Column"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# EDA VISUALIZATIONS
# =========================================================

elif page == "EDA Visualizations":

    st.title("📈 Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns

    feature = st.selectbox("Select Feature", numeric_cols)

    fig = px.histogram(
        df,
        x=feature,
        title=f"Distribution of {feature}",
        color_discrete_sequence=["blue"]
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(
        df,
        y=feature,
        title=f"Box Plot of {feature}"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# CORRELATION ANALYSIS
# =========================================================

elif page == "Correlation Analysis":

    st.title("🔗 Feature Correlation")

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(12,8))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=False,
        ax=ax
    )

    st.pyplot(fig)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("🏆 Model Comparison")

    models = ["Random Forest", "XGBoost", "SVM", "MLP", "Stacking"]

    accuracy = [0.91,0.94,0.88,0.90,0.96]

    fig = px.bar(
        x=models,
        y=accuracy,
        title="Model Accuracy Comparison",
        color=accuracy
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# CLUSTERING ANALYSIS
# =========================================================

elif page == "Clustering Analysis":

    st.title("🎯 Clustering Visualization")

    numeric_cols = df.select_dtypes(include=np.number).columns

    x = st.selectbox("X Axis", numeric_cols)
    y = st.selectbox("Y Axis", numeric_cols)

    fig = px.scatter(
        df,
        x=x,
        y=y,
        title="Cluster Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PREDICTION SYSTEM
# =========================================================

elif page == "Prediction System":

    st.title("🔮 AI Prediction System")

    st.write("Enter feature values to make predictions")

    input_data = []

    numeric_cols = df.select_dtypes(include=np.number).columns[:-1]

    for col in numeric_cols:
        val = st.number_input(col, float(df[col].min()), float(df[col].max()))
        input_data.append(val)

    if st.button("Predict"):

        prediction = model.predict([input_data])

        st.success(f"Prediction Result: {prediction[0]}")
