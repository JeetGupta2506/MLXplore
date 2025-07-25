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
        params["max_iter"] = st.sidebar.slider("Max Iterations", 50, 1000, 100)
        params["solver"] = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
    elif model == "SVM":
        params["C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        params["kernel"] = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        if params["kernel"] in ["rbf", "poly", "sigmoid"]:
            params["gamma"] = st.sidebar.selectbox("Gamma", ["scale", "auto"])
        if params["kernel"] == "poly":
            params["degree"] = st.sidebar.slider("Degree", 2, 5, 3)
    elif model == "KNN":
        params["n_neighbors"] = st.sidebar.slider("Neighbors", 1, 15, 5)
        params["weights"] = st.sidebar.selectbox("Weights", ["uniform", "distance"])
        params["p"] = st.sidebar.slider("Power (p)", 1, 5, 2)
    elif model == "Decision Tree":
        params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 3)
        params["min_samples_split"] = st.sidebar.slider("Min Samples Split", 2, 10, 2)
        params["criterion"] = st.sidebar.selectbox("Criterion", ["gini", "entropy", "log_loss"])
    elif model == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("Trees", 10, 200, 100)
        params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)
        params["min_samples_split"] = st.sidebar.slider("Min Samples Split", 2, 10, 2)
        params["criterion"] = st.sidebar.selectbox("Criterion", ["gini", "entropy", "log_loss"])
    elif model == "AdaBoost":
        params["n_estimators"] = st.sidebar.slider("Estimators", 10, 100, 50)
        params["learning_rate"] = st.sidebar.slider("Learning Rate", 0.01, 2.0, 1.0)
    elif model == "MLP (Neural Net)":
        params["hidden_layer_sizes"] = st.sidebar.text_input("Hidden Layers (tuple)", "(100,)")
        params["activation"] = st.sidebar.selectbox("Activation", ["relu", "tanh", "logistic"])
        params["solver"] = st.sidebar.selectbox("Solver", ["adam", "sgd", "lbfgs"])
        params["alpha"] = st.sidebar.slider("Alpha (L2 penalty)", 0.0001, 0.01, 0.0001)
        params["learning_rate_init"] = st.sidebar.slider("Learning Rate Init", 0.0001, 0.1, 0.001)
        params["max_iter"] = st.sidebar.slider("Max Iterations", 50, 1000, 200)
        try:
            params["hidden_layer_sizes"] = eval(params["hidden_layer_sizes"])
        except:
            params["hidden_layer_sizes"] = (100,)
elif task == "Regression":
    if model == "SVR":
        params["C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        params["kernel"] = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        if params["kernel"] in ["rbf", "poly", "sigmoid"]:
            params["gamma"] = st.sidebar.selectbox("Gamma", ["scale", "auto"])
        if params["kernel"] == "poly":
            params["degree"] = st.sidebar.slider("Degree", 2, 5, 3)
        params["epsilon"] = st.sidebar.slider("Epsilon", 0.01, 1.0, 0.1)
elif task == "Clustering":
    if model == "KMeans":
        params["n_clusters"] = st.sidebar.slider("Clusters", 2, 6, 3)
        params["init"] = st.sidebar.selectbox("Init", ["k-means++", "random"])
        params["max_iter"] = st.sidebar.slider("Max Iterations", 50, 500, 300)
    elif model == "DBSCAN":
        params["eps"] = st.sidebar.slider("Epsilon (eps)", 0.05, 1.0, 0.3, step=0.05)
        params["min_samples"] = st.sidebar.slider("Min Samples", 2, 10, 5)

# Dataset preview
if dataset:
    preview_payload = {"task": task.lower(), "dataset": dataset}
    try:
        preview_res = requests.post("http://localhost:8000/preview", json=preview_payload)
        preview_data = preview_res.json()
        if "X" in preview_data:
            X = np.array(preview_data["X"])
            fig, ax = plt.subplots()
            if task == "classification" and "y" in preview_data:
                y = np.array(preview_data["y"])
                scatter = ax.scatter(X[:,0], X[:,1], c=y, cmap='tab10', edgecolor='k')
                ax.set_title("Dataset Preview (colored by class)")
            elif task == "regression" and "y" in preview_data:
                y = np.array(preview_data["y"])
                scatter = ax.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k')
                ax.set_title("Dataset Preview (colored by target)")
            else:
                scatter = ax.scatter(X[:,0], X[:,1], edgecolor='k')
                ax.set_title("Dataset Preview")
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not load dataset preview: {e}")

run = st.sidebar.button("Train & Visualize")

if run:
    with st.spinner("Training..."):
        payload = {"task": task.lower(), "dataset": dataset, "model": model, "params": params}
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