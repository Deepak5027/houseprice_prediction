import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(
    page_title="AI Powered ML Prediction System",
    layout="wide",
    page_icon="🤖"
)

# =========================================================
# GET BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = os.path.join(BASE_DIR, "stacking_ensemble_compressed.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")   # ✅ added


@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None


@st.cache_resource
def load_features():   # ✅ added
    try:
        feats = joblib.load(FEATURES_PATH)
        return feats
    except Exception as e:
        st.error(f"Features load error: {e}")
        return None


model = load_model()
features_list = load_features()   # ✅ added


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    try:
        df_path = os.path.join(BASE_DIR, "dataset.csv")
        df = pd.read_csv(df_path)
        return df
    except Exception as e:
        st.error(f"Dataset error: {e}")
        return None


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
        "Cost Analysis Dashboard",
        "Model Performance",
        "Clustering Analysis",
        "Prediction System"
    ]
)

# =========================================================
# ALL YOUR UI PAGES SAME (not changed)
# =========================================================

if page == "Project Overview":

    st.title("🤖 AI Powered Machine Learning Prediction System")

    if df is not None:
        col1, col2, col3 = st.columns(3)

        col1.metric("Dataset Size", df.shape[0])
        col2.metric("Features", df.shape[1])
        col3.metric("Model Type", "Stacking Ensemble")

        st.dataframe(df.head(20))


elif page == "Dataset Explorer":

    st.title("Dataset Explorer")

    if df is not None:
        st.dataframe(df)
        st.write(df.describe())


elif page == "EDA Visualizations":

    st.title("EDA")

    if df is not None:

        numeric_cols = df.select_dtypes(include=np.number).columns

        feature = st.selectbox("Feature", numeric_cols)

        fig = px.histogram(df, x=feature)

        st.plotly_chart(fig, use_container_width=True)


elif page == "Correlation Analysis":

    st.title("Correlation")

    if df is not None:

        corr = df.select_dtypes(include=np.number).corr()

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(corr, cmap="coolwarm", ax=ax)

        st.pyplot(fig)


elif page == "Cost Analysis Dashboard":

    st.title("Cost Dashboard")

    if df is not None:

        fig = px.histogram(df, x="construction_cost_usd")

        st.plotly_chart(fig)


elif page == "Model Performance":

    st.title("Model Performance")

    models = ["Random Forest", "XGBoost", "SVM", "MLP", "Stacking"]
    accuracy = [0.91, 0.94, 0.88, 0.90, 0.96]

    fig = px.bar(x=models, y=accuracy)

    st.plotly_chart(fig)


elif page == "Clustering Analysis":

    st.title("Clustering")

    if df is not None:

        num = df.select_dtypes(include=np.number).columns

        x = st.selectbox("X", num)
        y = st.selectbox("Y", num)

        fig = px.scatter(df, x=x, y=y)

        st.plotly_chart(fig)


# =========================================================
# PREDICTION SYSTEM (ONLY THIS PART CHANGED)
# =========================================================

elif page == "Prediction System":

    st.title("🔮 AI Prediction System")

    st.write("Enter feature values to make predictions")

    if model is not None and df is not None and features_list is not None:

        st.subheader("🏠 House Layout Viewer")

        length = st.slider("Plot Length", 20, 200, 60)
        width = st.slider("Plot Width", 20, 200, 40)

        area = length * width

        fig = go.Figure()

        fig.add_shape(
            type="rect",
            x0=0,
            y0=0,
            x1=length,
            y1=width
        )

        fig.update_layout(
            title=f"Top View Layout | Area: {area} sqft",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.metric("Plot Area", f"{area} sqft")

        # -------------------
        # INPUT VALUES
        # -------------------

        input_values = {}

        numeric_cols = df.select_dtypes(include=np.number).columns[:-1]

        for col in numeric_cols:
            val = st.number_input(
                col,
                float(df[col].min()),
                float(df[col].max())
            )
            input_values[col] = val

        # -------------------
        # PREDICT FIXED
        # -------------------

        if st.button("Predict"):

            try:

                # create all 73 features
                input_dict = {f: 0 for f in features_list}

                # fill entered values
                for k, v in input_values.items():
                    if k in input_dict:
                        input_dict[k] = v

                input_df = pd.DataFrame([input_dict])

                prediction = model.predict(input_df)

                st.success(f"Prediction Result: {prediction[0]}")

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

    else:
        st.error("Model / dataset / features not loaded")
