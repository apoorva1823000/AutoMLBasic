import streamlit as st
import pandas as pd
import os

# Import profiling capability
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# ML processing
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("https://img.icons8.com/?size=100&id=ij6f4GUUwLE8&format=png&color=000000")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info(
        "This application allows you to build an automated machine learning pipeline using Streamlit, "
        "Pandas Profiling and PyCaret")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload your Data for Modelling")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        # do something here
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = ProfileReport(df, title="Profiling Report")
    st_profile_report(profile_report)
if choice == "ML":
    st.title("Machine Learning Process")
    target = st.selectbox("Select your target", df.columns)
    if st.button("Train model"):
        setup(df, target=target, verbose=False)
        setup_df = pull()
        st.info("This is the ML Experiment Settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, "best_model")

if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the file", f, "trained_model.pkl")
    # pass
