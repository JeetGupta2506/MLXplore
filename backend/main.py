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
from sklearn.datasets import make_moons, make_blobs, load_wine, load_breast_cancer, make_circles, make_gaussian_quantiles, load_diabetes, make_regression, fetch_california_housing
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os
# --- Add for image generation ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
# ---

app = FastAPI()

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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

class TuneRequest(BaseModel):
    task: str
    dataset: str
    model: str
    param_grid: dict
    search_type: str  # "grid" or "random"
    cv_folds: int = 5
    n_iter: int = 10  # for random search
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
        elif request.dataset == "Moons":
            X, y = make_moons(n_samples=100, noise=0, random_state=0)
        elif request.dataset == "Blobs":
            X, y = make_blobs(n_samples=150, centers=3, random_state=0, cluster_std=0.0)
        elif request.dataset == "Circles":
            X, y = make_circles(noise=0, factor=0.5, random_state=1)
        elif request.dataset == "Gaussian Quantiles":
            X, y = make_gaussian_quantiles(n_samples=150, n_features=2, n_classes=3, random_state=1)
        else:
            return {"error": "Unknown dataset"}
        # --- Matplotlib image generation ---
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        scatter = ax.scatter(X[:,0], X[:,1], c=y, cmap='tab10', edgecolor='k', s=30)
        ax.set_title("Dataset Preview (colored by class/target)", fontsize=12)
        ax.set_xlabel("Feature 1", fontsize=10)
        ax.set_ylabel("Feature 2", fontsize=10)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        # ---
        return {"X": X.tolist(), "y": y.tolist(), "image": img_base64}
    elif request.task == "regression":
        if request.dataset == "California Housing":
            data = fetch_california_housing()
            X, y = data.data[:, :2], data.target
        elif request.dataset == "Synthetic":
            X, y = make_regression(n_samples=200, n_features=2, noise=0, random_state=42)
        else:
            return {"error": f"Unknown regression dataset: {request.dataset}"}
        # --- Matplotlib image generation ---
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        scatter = ax.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k', s=30)
        ax.set_title("Dataset Preview (colored by target)", fontsize=12)
        ax.set_xlabel("Feature 1", fontsize=10)
        ax.set_ylabel("Feature 2", fontsize=10)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        # ---
        return {"X": X.tolist(), "y": y.tolist(), "image": img_base64}
    elif request.task == "clustering":
        if request.dataset == "Blobs":
            X, _ = make_blobs(n_samples=200, centers=3, random_state=42, cluster_std=0.0)
        elif request.dataset == "Moons":
            X, _ = make_moons(n_samples=200, noise=0, random_state=42)
        elif request.dataset == "Circles":
            X, _ = make_circles(n_samples=200, noise=0, factor=0.5, random_state=42)
        elif request.dataset == "Gaussian Quantiles":
            X, _ = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=42)
        elif request.dataset == "Iris":
            data = datasets.load_iris()
            X = data.data[:, :2]
        else:
            return {"error": "Unknown clustering dataset"}
        # --- Matplotlib image generation ---
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        scatter = ax.scatter(X[:,0], X[:,1], edgecolor='k', s=30)
        ax.set_title("Dataset Preview", fontsize=12)
        ax.set_xlabel("Feature 1", fontsize=10)
        ax.set_ylabel("Feature 2", fontsize=10)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        # ---
        return {"X": X.tolist(), "image": img_base64}
    else:
        return {"error": "Unknown task"}

@app.post("/tune")
def tune_hyperparameters(request: TuneRequest):
    try:
        # Get dataset
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
            elif request.dataset == "Moons":
                X, y = make_moons(n_samples=200, noise=0.1, random_state=0)
            elif request.dataset == "Blobs":
                X, y = make_blobs(n_samples=200, centers=3, random_state=0, cluster_std=1.0)
            elif request.dataset == "Circles":
                X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=1)
            elif request.dataset == "Gaussian Quantiles":
                X, y = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=3, random_state=1)
            else:
                return {"error": "Unknown dataset"}
        elif request.task == "regression":
            if request.dataset == "California Housing":
                data = fetch_california_housing()
                X, y = data.data[:, :2], data.target
            elif request.dataset == "Synthetic":
                X, y = make_regression(n_samples=200, n_features=2, noise=10, random_state=42)
            else:
                return {"error": f"Unknown regression dataset: {request.dataset}"}
        else:
            return {"error": "Hyperparameter tuning not supported for clustering"}

        # Get base model
        if request.model == "Logistic Regression":
            base_model = LogisticRegression(random_state=42)
        elif request.model == "SVM":
            base_model = SVC(random_state=42)
        elif request.model == "KNN":
            base_model = KNeighborsClassifier()
        elif request.model == "Decision Tree":
            base_model = DecisionTreeClassifier(random_state=42)
        elif request.model == "Random Forest":
            base_model = RandomForestClassifier(random_state=42)
        elif request.model == "AdaBoost":
            base_model = AdaBoostClassifier(random_state=42)
        elif request.model == "MLP (Neural Net)":
            base_model = MLPClassifier(random_state=42, max_iter=1000)
        elif request.model == "Linear Regression":
            base_model = LinearRegression()
        elif request.model == "SVR":
            base_model = SVR()
        else:
            return {"error": "Unknown model"}

        # Perform hyperparameter tuning
        if request.search_type == "grid":
            search = GridSearchCV(
                base_model, 
                request.param_grid, 
                cv=request.cv_folds, 
                scoring='accuracy' if request.task == 'classification' else 'r2',
                n_jobs=-1
            )
        else:  # random search
            search = RandomizedSearchCV(
                base_model, 
                request.param_grid, 
                cv=request.cv_folds, 
                scoring='accuracy' if request.task == 'classification' else 'r2',
                n_iter=request.n_iter,
                random_state=42,
                n_jobs=-1
            )

        search.fit(X, y)

        # Get results
        results = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": {
                "mean_test_score": search.cv_results_['mean_test_score'].tolist(),
                "std_test_score": search.cv_results_['std_test_score'].tolist(),
                "params": [str(p) for p in search.cv_results_['params']]
            },
            "search_type": request.search_type,
            "cv_folds": request.cv_folds
        }

        # Create visualization of results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Best parameters performance
        scores = search.cv_results_['mean_test_score']
        ax1.bar(range(len(scores)), scores)
        ax1.set_xlabel('Parameter Combination')
        ax1.set_ylabel('CV Score')
        ax1.set_title(f'{request.search_type.title()} Search Results')
        ax1.axhline(y=search.best_score_, color='r', linestyle='--', label=f'Best: {search.best_score_:.3f}')
        ax1.legend()

        # Plot 2: Train best model and show decision boundary/regression
        best_model = search.best_estimator_
        if request.task == "classification":
            # Decision boundary
            h = .02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax2.contourf(xx, yy, Z, alpha=0.3)
            ax2.scatter(X[:,0], X[:,1], c=y, marker='o', edgecolor='k', alpha=0.8)
            ax2.set_title('Best Model Decision Boundary')
        else:  # regression
            X_sorted = np.sort(X[:,0])
            X_plot = np.column_stack([X_sorted, np.full_like(X_sorted, X[:,1].mean())])
            y_pred_plot = best_model.predict(X_plot)
            ax2.scatter(X[:,0], y, c='blue', alpha=0.7, label='Data')
            ax2.plot(X_sorted, y_pred_plot, 'r-', linewidth=2, label='Best Model')
            ax2.set_xlabel("Feature 1")
            ax2.set_ylabel("Target")
            ax2.set_title('Best Model Fit')
            ax2.legend()
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        results["image"] = img_base64
        return results
    except Exception as e:
        return {"error": f"Hyperparameter tuning failed: {str(e)}"}
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
        elif request.dataset == "Moons":
            X, y = make_moons(n_samples=100, noise=0, random_state=0)
        elif request.dataset == "Blobs":
            X, y = make_blobs(n_samples=150, centers=3, random_state=0, cluster_std=0.0)
        elif request.dataset == "Circles":
            X, y = make_circles(noise=0, factor=0.5, random_state=1)
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
        # For decision boundary - use full dataset
        h = .02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # --- Plot decision boundary and full dataset ---
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.contourf(xx, yy, Z, alpha=0.3)
        scatter = ax.scatter(X[:,0], X[:,1], c=y, marker='o', edgecolor='k', alpha=0.8, s=30)
        ax.set_title("Decision Boundary & Full Dataset", fontsize=12)
        ax.set_xlabel("Feature 1", fontsize=10)
        ax.set_ylabel("Feature 2", fontsize=10)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
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
        if request.dataset == "California Housing":
            data = fetch_california_housing()
            X, y = data.data[:, :2], data.target
        elif request.dataset == "Synthetic":
            X, y = make_regression(n_samples=200, n_features=2, noise=0, random_state=42)
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
        # --- Plot regression on full dataset ---
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        # Plot full dataset
        ax.scatter(X[:,0], y, c='blue', alpha=0.7, label='Data', s=30)
        # Plot regression line/curve - use both features
        X_sorted = np.sort(X[:,0])
        # Create 2D array with both features for prediction
        X_plot = np.column_stack([X_sorted, np.full_like(X_sorted, X[:,1].mean())])  # Use mean of second feature
        y_pred_plot = model.predict(X_plot)
        ax.plot(X_sorted, y_pred_plot, 'r-', linewidth=2, label='Regression Line')
        ax.set_xlabel("Feature 1", fontsize=10)
        ax.set_ylabel("Target", fontsize=10)
        ax.set_title("Regression: Model Fit on Full Dataset", fontsize=12)
        ax.legend(fontsize=7)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
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
            X, _ = make_blobs(n_samples=200, centers=3, random_state=42, cluster_std=0.0)
        elif request.dataset == "Moons":
            X, _ = make_moons(n_samples=200, noise=0, random_state=42)
        elif request.dataset == "Circles":
            X, _ = make_circles(n_samples=200, noise=0, factor=0.5, random_state=42)
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
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        scatter = ax.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', edgecolor='k', s=30)
        ax.set_title("Clustering Result", fontsize=12)
        ax.set_xlabel("Feature 1", fontsize=10)
        ax.set_ylabel("Feature 2", fontsize=10)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
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