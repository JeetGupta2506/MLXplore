import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

st.title("ML Playground")

# Task selector
task = st.sidebar.selectbox("Task", ["Classification", "Regression", "Clustering"])

dataset_options = {
    "Classification": ["Iris", "Wine", "Breast Cancer", "Digits", "Moons", "Blobs", "Circles", "Gaussian Quantiles"],
    "Regression": ["Diabetes", "Synthetic"],
    "Clustering": ["Blobs", "Moons", "Circles", "Gaussian Quantiles", "Iris"]
}
model_options = {
    "Classification": [
        "Logistic Regression", "SVM", "KNN", "Decision Tree",
        "Random Forest", "AdaBoost", "Naive Bayes", "MLP (Neural Net)"
    ],
    "Regression": ["Linear Regression", "SVR"],
    "Clustering": ["KMeans", "DBSCAN"]
}

dataset = st.sidebar.selectbox("Dataset", dataset_options[task])
model = st.sidebar.selectbox("Model", model_options[task])
params = {}
noise = None
if task == "Classification":
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
elif task == "Regression":
    if model == "Linear Regression":
        pass
    elif model == "SVR":
        params["C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        params["kernel"] = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
    if dataset == "Synthetic":
        noise = st.sidebar.slider("Noise", 0, 50, 10)
elif task == "Clustering":
    if model == "KMeans":
        params["n_clusters"] = st.sidebar.slider("Clusters", 2, 6, 3)
    elif model == "DBSCAN":
        params["eps"] = st.sidebar.slider("Epsilon (eps)", 0.05, 1.0, 0.3, step=0.05)

# Button to run
if st.button("Train & Visualize"):
    with st.spinner("Training..."):
        payload = {"task": task.lower(), "dataset": dataset, "model": model, "params": params}
        if noise is not None:
            payload["noise"] = noise
        res = requests.post(
            "http://localhost:8000/train",
            json=payload
        )
        data = res.json()
        if "error" in data:
            st.error(data["error"])
        else:
            if data.get("task") == "classification":
                st.success(f"Test Accuracy: {data['score']:.2f}")
                fig, ax = plt.subplots()
                ax.contourf(np.array(data["xx"]), np.array(data["yy"]), np.array(data["Z"]), alpha=0.3)
                ax.scatter(np.array(data["X_train"])[:,0], np.array(data["X_train"])[:,1], 
                           c=data["y_train"], marker='o', edgecolor='k', label='Train', alpha=0.8)
                ax.scatter(np.array(data["X_test"])[:,0], np.array(data["X_test"])[:,1], 
                           c=data["y_test"], marker='^', edgecolor='k', label='Test', alpha=1.0)
                ax.set_title("Decision Boundary & Train/Test Split")
                ax.legend()
                st.pyplot(fig)
            elif data.get("task") == "regression":
                st.success(f"R²: {data['r2']:.2f} | MSE: {data['mse']:.2f}")
                fig, ax = plt.subplots()
                ax.scatter(data["y_test"], data["y_pred"], c='blue', label='Test', alpha=0.7)
                ax.plot([min(data["y_test"]), max(data["y_test"])], [min(data["y_test"]), max(data["y_test"])], 'r--', label='Ideal')
                ax.set_xlabel("True Value")
                ax.set_ylabel("Predicted Value")
                ax.set_title("Regression: True vs. Predicted (Test Set)")
                ax.legend()
                st.pyplot(fig)
            elif data.get("task") == "clustering":
                st.success("Clustering complete!")
                fig, ax = plt.subplots()
                X = np.array(data["X"])
                labels = np.array(data["labels"])
                scatter = ax.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', edgecolor='k')
                ax.set_title("Clustering Result")
                st.pyplot(fig) 