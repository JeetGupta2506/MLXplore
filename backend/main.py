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

app = FastAPI()

class TrainRequest(BaseModel):
    task: str  # "classification", "regression", "clustering"
    dataset: str
    model: str
    params: dict

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
            "task": "classification"
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
        return {
            "r2": r2,
            "mse": mse,
            "X_train": X_train.tolist(),
            "y_train": y_train.tolist(),
            "X_test": X_test.tolist(),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "task": "regression"
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
        return {
            "X": X.tolist(),
            "labels": labels.tolist(),
            "task": "clustering"
        }
    else:
        return {"error": "Unknown task"} 