import streamlit as st
import joblib
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

st.title("Customer Segmentation using Hierarchical Clustering")
st.markdown("Enter customer details to predict their segment.")

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")   
training_data=joblib.load("scaled_data.pkl")
clusters=joblib.load("clusters.pkl")

cluster_labels = {
    0: "Budget Customers",          
    1: "Premium Customers",         
    2: "Careful High Earners",      
    3: "Value Seekers",             
}

income = st.number_input("Income", min_value=1, max_value=200, value=20)
spending = st.number_input("Spending Score", min_value=1, max_value=200, value=15)

if st.button("Predict Cluster"):

    
    new_data = pd.DataFrame([[income, spending]], columns=["Income", "Spending"])
  
    scaled = scaler.transform(new_data)
    
    
    distances = cdist(scaled, training_data)  
    nearest_index = np.argmin(distances, axis=1)
    assigned_cluster = clusters[nearest_index[0]]


    st.success(f"Predicted Cluster: {assigned_cluster} - {cluster_labels.get(assigned_cluster, 'Unknown')}")
