import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report

def show_explore_page():
    st.title("About the Model")
    st.markdown("&nbsp;")
    df = pd.read_csv("Invistico_Airline.csv")
    st.write("Dataset Example: ")
    st.write(df.head())
    
    # Load y_test and y_pred
    y_test = np.load('y_test.npy', allow_pickle=True)
    y_pred = np.load('y_pred.npy', allow_pickle=True)

    # Compute confusion matrix
    matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(matrix.T, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")

    # Display the plot in Streamlit
    st.markdown("&nbsp;")
    st.title("Confusion Matrix")
    st.pyplot(fig)

    # Classification report
    report = classification_report(y_test, y_pred)
    st.markdown("&nbsp;")
    st.title("Classification Report")
    st.text(report)