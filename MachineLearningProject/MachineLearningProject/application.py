import streamlit as st
from pred_page import show_predict_page
from explore_page import show_explore_page

page = st.sidebar.selectbox("Predict or About the Model", ("Predict", "About the Model"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()



