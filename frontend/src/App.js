import React, { useState } from 'react';
import {
  Box,
  Drawer,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Slider,
  Button,
  CircularProgress,
  Alert,
  Stack,
  Divider,
  Card,
  CardContent,
  IconButton,
  ThemeProvider,
  createTheme
} from '@mui/material';
import { Brightness4, Brightness7 } from '@mui/icons-material';

const drawerWidth = 320;

const TASKS = [
  'classification',
  'regression',
  'clustering',
];

const DATASETS = {
  classification: [
    'Iris', 'Wine', 'Breast Cancer', 'Digits', 'Moons', 'Blobs', 'Circles', 'Gaussian Quantiles'
  ],
  regression: [
    'California Housing', 'Synthetic'
  ],
  clustering: [
    'Blobs', 'Moons', 'Circles', 'Gaussian Quantiles', 'Iris'
  ]
};

const MODELS = {
  classification: [
    'Logistic Regression', 'SVM', 'KNN', 'Decision Tree', 'Random Forest', 'AdaBoost', 'Naive Bayes', 'MLP (Neural Net)'
  ],
  regression: [
    'Linear Regression', 'SVR'
  ],
  clustering: [
    'KMeans', 'DBSCAN'
  ]
};

// Parameter configs for dynamic controls (min, max, step, default, type, options)
const PARAM_CONFIG = {
  // Classification
  'Logistic Regression': {
    C: { type: 'slider', min: 0.01, max: 10, step: 0.01, default: 1, label: 'C (Inverse Regularization)' },
    max_iter: { type: 'slider', min: 50, max: 1000, step: 10, default: 100, label: 'Max Iterations' },
    solver: { type: 'select', options: ['lbfgs', 'liblinear', 'saga'], default: 'lbfgs', label: 'Solver' }
  },
  SVM: {
    C: { type: 'slider', min: 0.01, max: 10, step: 0.01, default: 1, label: 'C' },
    kernel: { type: 'select', options: ['rbf', 'linear', 'poly', 'sigmoid'], default: 'rbf', label: 'Kernel' },
    gamma: { type: 'select', options: ['scale', 'auto'], default: 'scale', label: 'Gamma', dependsOn: { kernel: ['rbf', 'poly', 'sigmoid'] } },
    degree: { type: 'slider', min: 2, max: 5, step: 1, default: 3, label: 'Degree', dependsOn: { kernel: ['poly'] } }
  },
  KNN: {
    n_neighbors: { type: 'slider', min: 1, max: 15, step: 1, default: 5, label: 'Neighbors' },
    weights: { type: 'select', options: ['uniform', 'distance'], default: 'uniform', label: 'Weights' },
    p: { type: 'slider', min: 1, max: 5, step: 1, default: 2, label: 'Power (p)' }
  },
  'Decision Tree': {
    max_depth: { type: 'slider', min: 1, max: 20, step: 1, default: 3, label: 'Max Depth' },
    min_samples_split: { type: 'slider', min: 2, max: 10, step: 1, default: 2, label: 'Min Samples Split' },
    criterion: { type: 'select', options: ['gini', 'entropy', 'log_loss'], default: 'gini', label: 'Criterion' }
  },
  'Random Forest': {
    n_estimators: { type: 'slider', min: 10, max: 200, step: 1, default: 100, label: 'Trees' },
    max_depth: { type: 'slider', min: 1, max: 20, step: 1, default: 5, label: 'Max Depth' },
    min_samples_split: { type: 'slider', min: 2, max: 10, step: 1, default: 2, label: 'Min Samples Split' },
    criterion: { type: 'select', options: ['gini', 'entropy', 'log_loss'], default: 'gini', label: 'Criterion' }
  },
  AdaBoost: {
    n_estimators: { type: 'slider', min: 10, max: 100, step: 1, default: 50, label: 'Estimators' },
    learning_rate: { type: 'slider', min: 0.01, max: 2, step: 0.01, default: 1, label: 'Learning Rate' }
  },
  'Naive Bayes': {},
  'MLP (Neural Net)': {
    hidden_layer_sizes: { type: 'text', default: '(100,)', label: 'Hidden Layers (tuple)' },
    activation: { type: 'select', options: ['relu', 'tanh', 'logistic'], default: 'relu', label: 'Activation' },
    solver: { type: 'select', options: ['adam', 'sgd', 'lbfgs'], default: 'adam', label: 'Solver' },
    alpha: { type: 'slider', min: 0.0001, max: 0.01, step: 0.0001, default: 0.0001, label: 'Alpha (L2 penalty)' },
    learning_rate_init: { type: 'slider', min: 0.0001, max: 0.1, step: 0.0001, default: 0.001, label: 'Learning Rate Init' },
    max_iter: { type: 'slider', min: 50, max: 1000, step: 10, default: 200, label: 'Max Iterations' }
  },
  // Regression
  SVR: {
    C: { type: 'slider', min: 0.01, max: 10, step: 0.01, default: 1, label: 'C' },
    kernel: { type: 'select', options: ['rbf', 'linear', 'poly', 'sigmoid'], default: 'rbf', label: 'Kernel' },
    gamma: { type: 'select', options: ['scale', 'auto'], default: 'scale', label: 'Gamma', dependsOn: { kernel: ['rbf', 'poly', 'sigmoid'] } },
    degree: { type: 'slider', min: 2, max: 5, step: 1, default: 3, label: 'Degree', dependsOn: { kernel: ['poly'] } },
    epsilon: { type: 'slider', min: 0.01, max: 1, step: 0.01, default: 0.1, label: 'Epsilon' }
  },
  'Linear Regression': {},
  // Clustering
  KMeans: {
    n_clusters: { type: 'slider', min: 2, max: 6, step: 1, default: 3, label: 'Clusters' },
    init: { type: 'select', options: ['k-means++', 'random'], default: 'k-means++', label: 'Init' },
    max_iter: { type: 'slider', min: 50, max: 500, step: 10, default: 300, label: 'Max Iterations' }
  },
  DBSCAN: {
    eps: { type: 'slider', min: 0.05, max: 1, step: 0.05, default: 0.3, label: 'Epsilon (eps)' },
    min_samples: { type: 'slider', min: 2, max: 10, step: 1, default: 5, label: 'Min Samples' }
  }
};

function getParamFields(model, params, setParams, extra) {
  const config = PARAM_CONFIG[model] || {};
  return Object.entries(config).map(([key, cfg]) => {
    // Handle conditional display (dependsOn)
    if (cfg.dependsOn) {
      const [depKey, depVals] = Object.entries(cfg.dependsOn)[0];
      if (!params[depKey] || !depVals.includes(params[depKey])) return null;
    }
    if (cfg.type === 'slider') {
      return (
        <Box key={key} sx={{ my: 1 }}>
          <Typography gutterBottom>{cfg.label}</Typography>
          <Slider
            value={params[key] !== undefined ? params[key] : cfg.default}
            min={cfg.min}
            max={cfg.max}
            step={cfg.step}
            onChange={(_, val) => setParams(p => ({ ...p, [key]: val }))}
            valueLabelDisplay="auto"
          />
        </Box>
      );
    }
    if (cfg.type === 'select') {
      return (
        <FormControl fullWidth key={key} sx={{ my: 1 }}>
          <InputLabel>{cfg.label}</InputLabel>
          <Select
            value={params[key] !== undefined ? params[key] : cfg.default}
            label={cfg.label}
            onChange={e => setParams(p => ({ ...p, [key]: e.target.value }))}
          >
            {cfg.options.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
            )
            }
          </Select>
        </FormControl>
      );
    }
    if (cfg.type === 'text') {
      return (
        <TextField
          key={key}
          label={cfg.label}
          name={key}
          value={params[key] !== undefined ? params[key] : cfg.default}
          onChange={e => setParams(p => ({ ...p, [key]: e.target.value }))}
          margin="normal"
          fullWidth
        />
      );
    }
    return null;
  });
}

function App() {
  const [task, setTask] = useState('classification');
  const [dataset, setDataset] = useState(DATASETS['classification'][0]);
  const [model, setModel] = useState(MODELS['classification'][0]);
  const [params, setParams] = useState({});
  const [preview, setPreview] = useState(null);
  const [trainResult, setTrainResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [darkMode, setDarkMode] = useState(false);
  const [tuningMode, setTuningMode] = useState(false);
  const [tuningParams, setTuningParams] = useState({});
  const [tuningResult, setTuningResult] = useState(null);
  const [searchType, setSearchType] = useState('grid');
  const [cvFolds, setCvFolds] = useState(5);
  const [nIter, setNIter] = useState(10);

  // Create theme based on darkMode state
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#1976d2',
      },
      secondary: {
        main: '#dc004e',
      },
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1e1e1e' : '#ffffff',
      },
    },
  });

  const toggleTheme = () => {
    setDarkMode(!darkMode);
  };

  // Reset params on model change
  React.useEffect(() => {
    setParams({});
  }, [model, task]);

  const handleTaskChange = (e) => {
    const newTask = e.target.value;
    setTask(newTask);
    setDataset(DATASETS[newTask][0]);
    setModel(MODELS[newTask][0]);
    setParams({});
    setPreview(null);
    setTrainResult(null);
    setError(null);
  };

  const handleDatasetChange = (e) => {
    setDataset(e.target.value);
    setPreview(null);
    setTrainResult(null);
    setError(null);
  };

  const handleModelChange = (e) => {
    setModel(e.target.value);
    setParams({});
    setTrainResult(null);
    setTuningResult(null);
    setError(null);
  };

  const handlePreview = async () => {
    setLoading(true);
    setError(null);
    setPreview(null);
    try {
      const res = await fetch('http://localhost:8000/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, dataset })
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        return;
      }
      setPreview(data);
    } catch (err) {
      setError(`Failed to fetch preview: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    setTrainResult(null);
    try {
      let sendParams = { ...params };
      // Convert hidden_layer_sizes string to tuple for MLP
      if (model === 'MLP (Neural Net)' && typeof sendParams.hidden_layer_sizes === 'string') {
        try {
          sendParams.hidden_layer_sizes = eval(sendParams.hidden_layer_sizes);
        } catch {
          sendParams.hidden_layer_sizes = [100];
        }
      }
      const res = await fetch('http://localhost:8000/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, dataset, model, params: sendParams })
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        return;
      }
      setTrainResult(data);
    } catch (err) {
      setError(`Failed to train model: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Prepare metrics for train result
  let trainMetrics = null;
  if (trainResult) {
    if (trainResult.task === 'classification') {
      trainMetrics = <Alert severity="success" sx={{ mb: 2 }}>Test Accuracy: {trainResult.score?.toFixed(2)}</Alert>;
    } else if (trainResult.task === 'regression') {
      trainMetrics = <Alert severity="success" sx={{ mb: 2 }}>R²: {trainResult.r2?.toFixed(2)} | MSE: {trainResult.mse?.toFixed(2)}</Alert>;
    } else if (trainResult.task === 'clustering') {
      trainMetrics = <Alert severity="success" sx={{ mb: 2 }}>Clustering complete!</Alert>;
    }
  }

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ display: 'flex', bgcolor: 'background.default', minHeight: '100vh' }}>
        {/* Sidebar Drawer */}
        <Drawer
          variant="permanent"
          anchor="left"
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: { 
              width: drawerWidth, 
              boxSizing: 'border-box', 
              p: 2,
              bgcolor: 'background.paper'
            }
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h5">MLPlay Controls</Typography>
            <IconButton onClick={toggleTheme} color="inherit">
              {darkMode ? <Brightness7 /> : <Brightness4 />}
            </IconButton>
          </Box>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Task</InputLabel>
            <Select value={task} label="Task" onChange={handleTaskChange}>
              {TASKS.map(t => <MenuItem key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</MenuItem>)}
              )
              }
            </Select>
          </FormControl>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Dataset</InputLabel>
            <Select value={dataset} label="Dataset" onChange={handleDatasetChange}>
              {DATASETS[task].map(d => <MenuItem key={d} value={d}>{d}</MenuItem>)}
              )
              }
            </Select>
          </FormControl>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Model</InputLabel>
            <Select value={model} label="Model" onChange={handleModelChange}>
              {MODELS[task].map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}
              )
              }
            </Select>
          </FormControl>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle1" sx={{ mb: 1 }}>Parameters</Typography>
          {getParamFields(model, params, setParams)}
          <Stack direction="row" spacing={2} sx={{ mt: 3 }}>
            <Button variant="contained" color="primary" onClick={handlePreview} disabled={loading}>Preview</Button>
            <Button variant="contained" color="secondary" onClick={handleTrain} disabled={loading}>Train & Visualize</Button>
          </Stack>
        </Drawer>
        {/* Main Content */}
        <Box sx={{ flexGrow: 1, p: 3, ml: `${drawerWidth}px`, bgcolor: 'background.default' }}>
          <Typography variant="h3" gutterBottom align="center" sx={{ color: 'text.primary' }}>MLPlay: ML Playground</Typography>
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
              <CircularProgress />
            </Box>
          )}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
          )}
          {/* Preview Section: Only Matplotlib image preview */}
          {preview && preview.image && (
            <Card variant="outlined" sx={{ mb: 3, bgcolor: 'background.paper' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ color: 'text.primary' }}>Dataset Preview</Typography>
                <img src={`data:image/png;base64,${preview.image}`} alt="Preview Visualization" style={{maxWidth: '100%'}} />
              </CardContent>
            </Card>
          )}
          {/* Train Result Section: Only Matplotlib image and metrics */}
          {trainMetrics}
          {trainResult && trainResult.image && (
            <Card variant="outlined" sx={{ bgcolor: 'background.paper' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ color: 'text.primary' }}>Result Visualization</Typography>
                <img src={`data:image/png;base64,${trainResult.image}`} alt="Result Visualization" style={{maxWidth: '100%'}} />
              </CardContent>
            </Card>
          )}
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
