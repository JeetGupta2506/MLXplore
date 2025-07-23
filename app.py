import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

st.title("ML Playground")

# Sidebar controls
dataset = st.sidebar.selectbox("Dataset", [
    "Iris", "Wine", "Breast Cancer", "Digits", "Moons", "Blobs", "Circles", "Gaussian Quantiles"
])
model = st.sidebar.selectbox("Model", [
    "Logistic Regression", "SVM", "KNN", "Decision Tree",
    "Random Forest", "AdaBoost", "Naive Bayes", "MLP (Neural Net)"
])

params = {}
if model == "Logistic Regression":
    params["C"] = st.sidebar.slider("C (Inverse Regularization)", 0.01, 10.0, 1.0)
elif model == "SVM":
    params["C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0)
    params["kernel"] = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
elif model == "KNN":
    params["n_neighbors"] = st.sidebar.slider("Neighbors", 1, 15, 5)
elif model == "Decision Tree":
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 10, 3)
elif model == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("Trees", 10, 200, 100)
elif model == "AdaBoost":
    params["n_estimators"] = st.sidebar.slider("Estimators", 10, 100, 50)
elif model == "MLP (Neural Net)":
    params["hidden_layer_sizes"] = st.sidebar.text_input("Hidden Layers (tuple)", "(100,)")
    params["activation"] = st.sidebar.selectbox("Activation", ["relu", "tanh", "logistic"])
    try:
        params["hidden_layer_sizes"] = eval(params["hidden_layer_sizes"])
    except:
        params["hidden_layer_sizes"] = (100,)

if st.button("Train & Visualize"):
    with st.spinner("Training..."):
        res = requests.post(
            "http://localhost:8000/train",
            json={"dataset": dataset, "model": model, "params": params}
        )
        data = res.json()
        if "error" in data:
            st.error(data["error"])
        else:
            st.success(f"Test Accuracy: {data['score']:.2f}")
            # Plot
            fig, ax = plt.subplots()
            ax.contourf(np.array(data["xx"]), np.array(data["yy"]), np.array(data["Z"]), alpha=0.3)
            scatter = ax.scatter(np.array(data["X"])[:,0], np.array(data["X"])[:,1], c=data["y"], edgecolor='k')
            ax.set_title("Decision Boundary")
            st.pyplot(fig)
