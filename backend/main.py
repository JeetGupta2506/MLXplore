from fastapi import FastAPI
from pydantic import BaseModel
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, make_blobs, load_wine, load_breast_cancer, load_digits, make_circles, make_gaussian_quantiles, load_diabetes, make_regression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
# --- Add for image generation ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
# ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    task: str  # "classification", "regression", "clustering"
    dataset: str
    model: str
    params: dict

class PreviewRequest(BaseModel):
    task: str
    dataset: str

@app.post("/preview")
def preview(request: PreviewRequest):
    if request.task == "classification":
        if request.dataset == "Iris":
            data = datasets.load_iris()
            X, y = data.data[:, :2], data.target
        elif request.dataset == "Wine":
            data = load_wine()
            X, y = data.data[:, :2], data.target
        elif request.dataset == "Breast Cancer":
            data = load_breast_cancer()
            X, y = data.data[:, :2], data.target
        elif request.dataset == "Digits":
            data = load_digits()
            X, y = data.data[:, :2], data.target
        elif request.dataset == "Moons":
            X, y = make_moons(noise=0.3, random_state=0)
        elif request.dataset == "Blobs":
            X, y = make_blobs(n_samples=150, centers=3, random_state=0, cluster_std=1.0)
        elif request.dataset == "Circles":
            X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
        elif request.dataset == "Gaussian Quantiles":
            X, y = make_gaussian_quantiles(n_samples=150, n_features=2, n_classes=3, random_state=1)
        else:
            return {"error": "Unknown dataset"}
        # --- Matplotlib image generation ---
        fig, ax = plt.subplots()
        scatter = ax.scatter(X[:,0], X[:,1], c=y, cmap='tab10', edgecolor='k')
        ax.set_title("Dataset Preview (colored by class/target)")
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        # ---
        return {"X": X.tolist(), "y": y.tolist(), "image": img_base64}
    elif request.task == "regression":
        if request.dataset == "Diabetes":
            data = load_diabetes()
            X, y = data.data[:, :2], data.target
        elif request.dataset == "Synthetic":
            X, y = make_regression(n_samples=200, n_features=2, noise=10, random_state=42)
        else:
            return {"error": f"Unknown regression dataset: {request.dataset}"}
        # --- Matplotlib image generation ---
        fig, ax = plt.subplots()
        scatter = ax.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k')
        ax.set_title("Dataset Preview (colored by target)")
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        # ---
        return {"X": X.tolist(), "y": y.tolist(), "image": img_base64}
    elif request.task == "clustering":
        if request.dataset == "Blobs":
            X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
        elif request.dataset == "Moons":
            X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
        elif request.dataset == "Circles":
            X, _ = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
        elif request.dataset == "Gaussian Quantiles":
            X, _ = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=42)
        elif request.dataset == "Iris":
            data = datasets.load_iris()
            X = data.data[:, :2]
        else:
            return {"error": "Unknown clustering dataset"}
        # --- Matplotlib image generation ---
        fig, ax = plt.subplots()
        scatter = ax.scatter(X[:,0], X[:,1], edgecolor='k')
        ax.set_title("Dataset Preview")
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        # ---
        return {"X": X.tolist(), "image": img_base64}
    else:
        return {"error": "Unknown task"}

@app.post("/train")
def train(request: TrainRequest):
    if request.task == "classification":
        # Dataset selection
        if request.dataset == "Iris":
            data = datasets.load_iris()
            X, y = data.data[:, :2], data.target
        elif request.dataset == "Wine":
            data = load_wine()
            X, y = data.data[:, :2], data.target
        elif request.dataset == "Breast Cancer":
            data = load_breast_cancer()
            X, y = data.data[:, :2], data.target
        elif request.dataset == "Digits":
            data = load_digits()
            X, y = data.data[:, :2], data.target
        elif request.dataset == "Moons":
            X, y = make_moons(noise=0.3, random_state=0)
        elif request.dataset == "Blobs":
            X, y = make_blobs(n_samples=150, centers=3, random_state=0, cluster_std=1.0)
        elif request.dataset == "Circles":
            X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
        elif request.dataset == "Gaussian Quantiles":
            X, y = make_gaussian_quantiles(n_samples=150, n_features=2, n_classes=3, random_state=1)
        else:
            return {"error": "Unknown dataset"}
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Model selection
        if request.model == "Logistic Regression":
            model = LogisticRegression(**request.params)
        elif request.model == "SVM":
            model = SVC(probability=True, **request.params)
        elif request.model == "KNN":
            model = KNeighborsClassifier(**request.params)
        elif request.model == "Decision Tree":
            model = DecisionTreeClassifier(**request.params)
        elif request.model == "Random Forest":
            model = RandomForestClassifier(**request.params)
        elif request.model == "AdaBoost":
            model = AdaBoostClassifier(**request.params)
        elif request.model == "Naive Bayes":
            model = GaussianNB(**request.params)
        elif request.model == "MLP (Neural Net)":
            if "max_iter" not in request.params:
                request.params["max_iter"] = 1000
            model = MLPClassifier(**request.params)
        else:
            return {"error": "Unknown model"}
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        # For decision boundary
        h = .02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # --- Plot decision boundary and points ---
        fig, ax = plt.subplots()
        ax.contourf(xx, yy, Z, alpha=0.3)
        scatter1 = ax.scatter(X_train[:,0], X_train[:,1], c=y_train, marker='o', edgecolor='k', label='Train', alpha=0.8)
        scatter2 = ax.scatter(X_test[:,0], X_test[:,1], c=y_test, marker='^', edgecolor='k', label='Test', alpha=1.0)
        ax.set_title("Decision Boundary & Train/Test Split")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        # ---
        return {
            "score": score,
            "X_train": X_train.tolist(),
            "y_train": y_train.tolist(),
            "X_test": X_test.tolist(),
            "y_test": y_test.tolist(),
            "xx": xx.tolist(),
            "yy": yy.tolist(),
            "Z": Z.tolist(),
            "classes": np.unique(y).tolist(),
            "task": "classification",
            "image": img_base64
        }
    elif request.task == "regression":
        # Dataset selection (case-sensitive)
        if request.dataset == "Diabetes":
            data = load_diabetes()
            X, y = data.data[:, :2], data.target
        elif request.dataset == "Synthetic":
            X, y = make_regression(n_samples=200, n_features=2, noise=10, random_state=42)
        else:
            return {"error": f"Unknown regression dataset: {request.dataset}"}
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Model selection
        if request.model == "Linear Regression":
            model = LinearRegression(**request.params)
        elif request.model == "SVR":
            model = SVR(**request.params)
        else:
            return {"error": f"Unknown regression model: {request.model}"}
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        # --- Plot true vs predicted ---
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, c='blue', label='Test', alpha=0.7)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal')
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        ax.set_title("Regression: True vs. Predicted (Test Set)")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        # ---
        return {
            "r2": r2,
            "mse": mse,
            "X_train": X_train.tolist(),
            "y_train": y_train.tolist(),
            "X_test": X_test.tolist(),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "task": "regression",
            "image": img_base64
        }
    elif request.task == "clustering":
        # Dataset selection (no y)
        if request.dataset == "Blobs":
            X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
        elif request.dataset == "Moons":
            X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
        elif request.dataset == "Circles":
            X, _ = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
        elif request.dataset == "Gaussian Quantiles":
            X, _ = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=42)
        elif request.dataset == "Iris":
            data = datasets.load_iris()
            X = data.data[:, :2]
        else:
            return {"error": "Unknown clustering dataset"}
        # Model selection
        if request.model == "KMeans":
            model = KMeans(n_clusters=request.params.get('n_clusters', 3), random_state=42)
        elif request.model == "DBSCAN":
            model = DBSCAN(eps=request.params.get('eps', 0.3))
        else:
            return {"error": "Unknown clustering model"}
        labels = model.fit_predict(X)
        # --- Plot clustering result ---
        fig, ax = plt.subplots()
        scatter = ax.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', edgecolor='k')
        ax.set_title("Clustering Result")
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        # ---
        return {
            "X": X.tolist(),
            "labels": labels.tolist(),
            "task": "clustering",
            "image": img_base64
        }
    else:
        return {"error": "Unknown task"} 