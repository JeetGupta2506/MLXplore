import React from 'react';
import {
  Box,
  Button,
  Stack,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert
} from '@mui/material';
import { ExpandMore, Download, Share, Code } from '@mui/icons-material';

function ExportResults({ trainResult, model, params, dataset, task }) {
  const exportAsJSON = () => {
    const data = {
      experiment: {
        task,
        dataset,
        model,
        parameters: params,
        timestamp: new Date().toISOString()
      },
      results: trainResult
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ml_experiment_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportAsCSV = () => {
    if (!trainResult) return;
    
    let csvContent = '';
    
    if (trainResult.task === 'classification') {
      csvContent = 'X1,X2,y_true,y_pred\n';
      trainResult.X_test.forEach((point, i) => {
        csvContent += `${point[0]},${point[1]},${trainResult.y_test[i]},predicted\n`;
      });
    } else if (trainResult.task === 'regression') {
      csvContent = 'y_true,y_pred\n';
      trainResult.y_test.forEach((actual, i) => {
        csvContent += `${actual},${trainResult.y_pred[i]}\n`;
      });
    }
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ml_results_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const generateCode = () => {
    const codeTemplate = `
# Generated ML experiment code
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import ${getDatasetImport()}
${getModelImport()}

# Load dataset
${getDatasetCode()}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = ${getModelCode()}

# Train model
model.fit(X_train, y_train)

# Evaluate
${getEvaluationCode()}
`;

    const blob = new Blob([codeTemplate], { type: 'text/python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ml_experiment_${Date.now()}.py`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getDatasetImport = () => {
    const imports = {
      'Iris': 'load_iris',
      'Wine': 'load_wine',
      'Breast Cancer': 'load_breast_cancer',
      'Digits': 'load_digits',
      'California Housing': 'fetch_california_housing',
      'Moons': 'make_moons',
      'Blobs': 'make_blobs',
      'Circles': 'make_circles',
      'Gaussian Quantiles': 'make_gaussian_quantiles',
      'Synthetic': 'make_regression'
    };
    return imports[dataset] || 'load_iris';
  };

  const getModelImport = () => {
    const imports = {
      'Logistic Regression': 'from sklearn.linear_model import LogisticRegression',
      'SVM': 'from sklearn.svm import SVC',
      'KNN': 'from sklearn.neighbors import KNeighborsClassifier',
      'Decision Tree': 'from sklearn.tree import DecisionTreeClassifier',
      'Random Forest': 'from sklearn.ensemble import RandomForestClassifier',
      'AdaBoost': 'from sklearn.ensemble import AdaBoostClassifier',
      'Naive Bayes': 'from sklearn.naive_bayes import GaussianNB',
      'MLP (Neural Net)': 'from sklearn.neural_network import MLPClassifier',
      'Linear Regression': 'from sklearn.linear_model import LinearRegression',
      'SVR': 'from sklearn.svm import SVR',
      'KMeans': 'from sklearn.cluster import KMeans',
      'DBSCAN': 'from sklearn.cluster import DBSCAN'
    };
    return imports[model] || '';
  };

  const getDatasetCode = () => {
    if (dataset === 'Iris') return 'data = load_iris()\nX, y = data.data[:, :2], data.target';
    if (dataset === 'Synthetic') return 'X, y = make_regression(n_samples=200, n_features=2, noise=0, random_state=42)';
    return `# Load ${dataset} dataset\nX, y = load_data()  # Replace with actual loading code`;
  };

  const getModelCode = () => {
    const paramStr = Object.keys(params).length > 0 
      ? JSON.stringify(params).replace(/"/g, '').replace(/:/g, '=')
      : '';
    
    const modelMap = {
      'Logistic Regression': 'LogisticRegression',
      'SVM': 'SVC',
      'KNN': 'KNeighborsClassifier',
      'Decision Tree': 'DecisionTreeClassifier',
      'Random Forest': 'RandomForestClassifier',
      'AdaBoost': 'AdaBoostClassifier',
      'Naive Bayes': 'GaussianNB',
      'MLP (Neural Net)': 'MLPClassifier',
      'Linear Regression': 'LinearRegression',
      'SVR': 'SVR'
    };
    
    const modelClass = modelMap[model] || 'Model';
    return `${modelClass}(${paramStr})`;
  };

  const getEvaluationCode = () => {
    if (task === 'classification') {
      return 'score = model.score(X_test, y_test)\nprint(f"Accuracy: {score:.4f}")';
    } else if (task === 'regression') {
      return `from sklearn.metrics import r2_score, mean_squared_error
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R²: {r2:.4f}, MSE: {mse:.4f}")`;
    }
    return 'print("Model trained successfully")';
  };

  if (!trainResult) {
    return (
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Download sx={{ color: 'text.disabled' }} />
            <Typography variant="h6" color="text.disabled">Export Results</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Alert severity="info">
            Train a model first to export results
          </Alert>
        </AccordionDetails>
      </Accordion>
    );
  }

  return (
    <Accordion>
      <AccordionSummary expandIcon={<ExpandMore />}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Download sx={{ color: 'success.main' }} />
          <Typography variant="h6">Export Results</Typography>
        </Box>
      </AccordionSummary>
      <AccordionDetails>
        <Stack spacing={2}>
          <Typography variant="body2" color="text.secondary">
            Export your experiment results and code for further analysis
          </Typography>
          
          <Stack direction="row" spacing={1} flexWrap="wrap">
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={exportAsJSON}
              size="small"
            >
              Export JSON
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<Share />}
              onClick={exportAsCSV}
              size="small"
              disabled={task === 'clustering'}
            >
              Export CSV
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<Code />}
              onClick={generateCode}
              size="small"
            >
              Generate Code
            </Button>
          </Stack>
        </Stack>
      </AccordionDetails>
    </Accordion>
  );
}

export default ExportResults;