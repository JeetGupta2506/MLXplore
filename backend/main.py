from fastapi import FastAPI
from pydantic import BaseModel
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, make_blobs, load_wine, load_breast_cancer, load_digits, make_circles, make_gaussian_quantiles
import numpy as np

app = FastAPI()

class TrainRequest(BaseModel):
    dataset: str
    model: str
    params: dict

@app.post("/train")
def train(request: TrainRequest):
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
        # Ensure max_iter is set for MLP
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
        "X": X.tolist(),
        "y": y.tolist(),
        "xx": xx.tolist(),
        "yy": yy.tolist(),
        "Z": Z.tolist(),
        "classes": np.unique(y).tolist()
    }
