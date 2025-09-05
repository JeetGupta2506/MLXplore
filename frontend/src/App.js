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
  createTheme,
  Paper,
  Chip,
  Fade,
  Zoom,
  LinearProgress
} from '@mui/material';
import { 
  Brightness4, 
  Brightness7, 
  PlayArrow, 
  Visibility, 
  Science, 
  Dataset, 
  ModelTraining,
  Settings,
  TrendingUp,
  Analytics
} from '@mui/icons-material';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const drawerWidth = 300;

const TASKS = [
  'classification',
  'regression',
  'clustering',
];

const DATASETS = {
  classification: [
    'Iris', 'Wine', 'Breast Cancer', 'Moons', 'Blobs', 'Circles', 'Gaussian Quantiles'
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

  // Create enhanced theme with better colors and gradients
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#6366f1',
        light: '#818cf8',
        dark: '#4f46e5',
      },
      secondary: {
        main: '#ec4899',
        light: '#f472b6',
        dark: '#db2777',
      },
      background: {
        default: darkMode ? '#0f0f23' : '#f8fafc',
        paper: darkMode ? '#1a1a2e' : '#ffffff',
      },
      text: {
        primary: darkMode ? '#e2e8f0' : '#1e293b',
        secondary: darkMode ? '#94a3b8' : '#64748b',
      },
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h3: {
        fontWeight: 700,
        background: darkMode 
          ? 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)'
          : 'linear-gradient(135deg, #1e293b 0%, #6366f1 100%)',
        backgroundClip: 'text',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
      },
      h5: {
        fontWeight: 600,
      },
      h6: {
        fontWeight: 600,
      },
    },
    shape: {
      borderRadius: 12,
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            textTransform: 'none',
            fontWeight: 600,
            padding: '10px 24px',
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 16,
            boxShadow: darkMode 
              ? '0 4px 20px rgba(0, 0, 0, 0.3)'
              : '0 4px 20px rgba(0, 0, 0, 0.08)',
          },
        },
      },
      MuiSlider: {
        styleOverrides: {
          root: {
            color: '#6366f1',
          },
          thumb: {
            width: 20,
            height: 20,
          },
        },
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
    setError(null);
  };

  const handlePreview = async () => {
    setLoading(true);
    setError(null);
    setPreview(null);
    try {
      const res = await fetch(`${API_BASE}/preview`, {
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
      const res = await fetch(`${API_BASE}/train`, {
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
      trainMetrics = (
        <Fade in={true} timeout={800}>
          <Alert 
            severity="success" 
            sx={{ 
              mb: 2, 
              borderRadius: 2,
              background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
              color: 'white',
              '& .MuiAlert-icon': { color: 'white' }
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Test Accuracy: {(trainResult.score * 100).toFixed(1)}%
            </Typography>
          </Alert>
        </Fade>
      );
    } else if (trainResult.task === 'regression') {
      trainMetrics = (
        <Fade in={true} timeout={800}>
          <Alert 
            severity="success" 
            sx={{ 
              mb: 2, 
              borderRadius: 2,
              background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
              color: 'white',
              '& .MuiAlert-icon': { color: 'white' }
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              R²: {trainResult.r2?.toFixed(3)} | MSE: {trainResult.mse?.toFixed(3)}
            </Typography>
          </Alert>
        </Fade>
      );
    } else if (trainResult.task === 'clustering') {
      trainMetrics = (
        <Fade in={true} timeout={800}>
          <Alert 
            severity="success" 
            sx={{ 
              mb: 2, 
              borderRadius: 2,
              background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
              color: 'white',
              '& .MuiAlert-icon': { color: 'white' }
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Clustering Complete! 🎉
            </Typography>
          </Alert>
        </Fade>
      );
    }
  }

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ 
        display: 'flex', 
        bgcolor: 'background.default', 
        minHeight: '100vh',
        background: darkMode 
          ? 'linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%)'
          : 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)'
      }}>
        {/* Enhanced Sidebar Drawer */}
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
              bgcolor: 'background.paper',
              background: darkMode 
                ? 'linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)'
                : 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
              borderRight: `1px solid ${darkMode ? '#2d3748' : '#e2e8f0'}`,
              boxShadow: darkMode 
                ? '2px 0 10px rgba(0, 0, 0, 0.2)'
                : '2px 0 10px rgba(0, 0, 0, 0.05)'
            }
          }}
        >
          {/* Header with enhanced styling */}
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            mb: 2,
            p: 1.5,
            borderRadius: 2,
            background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
            color: 'white'
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Science sx={{ fontSize: 20 }} />
              <Typography variant="h6" sx={{ fontWeight: 700, fontSize: '1rem' }}>
                MLPlay Controls
              </Typography>
            </Box>
            <IconButton onClick={toggleTheme} sx={{ color: 'white', p: 0.5 }}>
              {darkMode ? <Brightness7 /> : <Brightness4 />}
            </IconButton>
          </Box>

          {/* Task Selection */}
          <Paper elevation={0} sx={{ p: 1.5, mb: 1.5, borderRadius: 2, bgcolor: 'rgba(99, 102, 241, 0.05)' }}>
            <FormControl fullWidth size="small">
              <InputLabel>Task</InputLabel>
              <Select value={task} label="Task" onChange={handleTaskChange}>
                {TASKS.map(t => <MenuItem key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</MenuItem>)}
              </Select>
            </FormControl>
          </Paper>

          {/* Dataset Selection */}
          <Paper elevation={0} sx={{ p: 1.5, mb: 1.5, borderRadius: 2, bgcolor: 'rgba(236, 72, 153, 0.05)' }}>
            <FormControl fullWidth size="small">
              <InputLabel>Dataset</InputLabel>
              <Select value={dataset} label="Dataset" onChange={handleDatasetChange}>
                {DATASETS[task].map(d => <MenuItem key={d} value={d}>{d}</MenuItem>)}
              </Select>
            </FormControl>
          </Paper>

          {/* Model Selection */}
          <Paper elevation={0} sx={{ p: 1.5, mb: 1.5, borderRadius: 2, bgcolor: 'rgba(16, 185, 129, 0.05)' }}>
            <FormControl fullWidth size="small">
              <InputLabel>Model</InputLabel>
              <Select value={model} label="Model" onChange={handleModelChange}>
                {MODELS[task].map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}
              </Select>
            </FormControl>
          </Paper>

          {/* Parameters Section */}
          <Paper elevation={0} sx={{ p: 1.5, mb: 2, borderRadius: 2, bgcolor: 'rgba(245, 158, 11, 0.05)' }}>
            {getParamFields(model, params, setParams)}
          </Paper>

          {/* Action Buttons */}
          <Stack direction="column" spacing={1}>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={handlePreview} 
              disabled={loading}
              startIcon={<Visibility />}
              sx={{ py: 1.5, fontWeight: 600 }}
            >
              Preview Dataset
            </Button>
            <Button 
              variant="contained" 
              color="secondary" 
              onClick={handleTrain} 
              disabled={loading}
              startIcon={<PlayArrow />}
              sx={{ py: 1.5, fontWeight: 600 }}
            >
              Train & Visualize
            </Button>
          </Stack>

        </Drawer>

        {/* Enhanced Main Content */}
        <Box sx={{ 
          flexGrow: 1, 
          p: 3, 
          ml: 0, 
          bgcolor: 'transparent',
          position: 'relative'
        }}>
          {/* Header with gradient background */}
          <Box sx={{ 
            mb: 3, 
            p: 3, 
            borderRadius: 3,
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%)',
            border: `1px solid ${darkMode ? 'rgba(99, 102, 241, 0.2)' : 'rgba(99, 102, 241, 0.1)'}`,
            textAlign: 'center'
          }}>
            <Typography variant="h3" gutterBottom sx={{ mb: 1, fontSize: '2.5rem' }}>
              MLXplore: ML Playground
            </Typography>
            <Typography variant="h6" sx={{ color: 'text.secondary', fontWeight: 400 }}>
              Interactive Machine Learning Experimentation Platform
            </Typography>
          </Box>

          {/* Loading State */}
          {loading && (
            <Fade in={true} timeout={300}>
              <Box sx={{ mb: 3 }}>
                <LinearProgress 
                  sx={{ 
                    height: 8, 
                    borderRadius: 4,
                    background: 'rgba(99, 102, 241, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
                      borderRadius: 4
                    }
                  }} 
                />
                <Typography variant="body2" sx={{ mt: 1, textAlign: 'center', color: 'text.secondary' }}>
                  Processing your request...
                </Typography>
              </Box>
            </Fade>
          )}

          {/* Error State */}
          {error && (
            <Fade in={true} timeout={300}>
              <Alert 
                severity="error" 
                sx={{ 
                  mb: 3, 
                  borderRadius: 2,
                  background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
                  color: 'white',
                  '& .MuiAlert-icon': { color: 'white' }
                }}
              >
                {error}
              </Alert>
            </Fade>
          )}

          {/* Preview Section */}
          {preview && preview.image && (
            <Zoom in={true} timeout={500}>
              <Card sx={{ 
                mb: 3, 
                overflow: 'hidden',
                border: `1px solid ${darkMode ? '#2d3748' : '#e2e8f0'}`,
                '&:hover': {
                  transform: 'translateY(-2px)',
                  transition: 'transform 0.3s ease-in-out'
                }
              }}>
                <CardContent sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                    <Visibility sx={{ color: 'primary.main' }} />
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      Dataset Preview
                    </Typography>
                  </Box>
                  <Box sx={{ 
                    borderRadius: 2, 
                    overflow: 'hidden',
                    border: `1px solid ${darkMode ? '#2d3748' : '#e2e8f0'}`,
                    maxWidth: '100%',
                    display: 'flex',
                    justifyContent: 'center'
                  }}>
                    <img 
                      src={`data:image/png;base64,${preview.image}`} 
                      alt="Preview Visualization" 
                      style={{
                        maxWidth: '100%',
                        width: 'auto',
                        height: 'auto',
                        maxHeight: '400px',
                        display: 'block',
                        objectFit: 'contain'
                      }} 
                    />
                  </Box>
                </CardContent>
              </Card>
            </Zoom>
          )}

          {/* Train Result Section */}
          {trainMetrics}
          {trainResult && trainResult.image && (
            <Zoom in={true} timeout={500}>
              <Card sx={{ 
                overflow: 'hidden',
                border: `1px solid ${darkMode ? '#2d3748' : '#e2e8f0'}`,
                '&:hover': {
                  transform: 'translateY(-2px)',
                  transition: 'transform 0.3s ease-in-out'
                }
              }}>
                <CardContent sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                    <TrendingUp sx={{ color: 'secondary.main' }} />
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      Result Visualization
                    </Typography>
                  </Box>
                  <Box sx={{ 
                    borderRadius: 2, 
                    overflow: 'hidden',
                    border: `1px solid ${darkMode ? '#2d3748' : '#e2e8f0'}`,
                    maxWidth: '100%',
                    display: 'flex',
                    justifyContent: 'center'
                  }}>
                    <img 
                      src={`data:image/png;base64,${trainResult.image}`} 
                      alt="Result Visualization" 
                      style={{
                        maxWidth: '100%',
                        width: 'auto',
                        height: 'auto',
                        maxHeight: '400px',
                        display: 'block',
                        objectFit: 'contain'
                      }} 
                    />
                  </Box>
                </CardContent>
              </Card>
            </Zoom>
          )}

          {/* Empty State */}
          {!preview && !trainResult && !loading && !error && (
            <Fade in={true} timeout={800}>
              <Box sx={{ 
                textAlign: 'center', 
                py: 6,
                px: 3
              }}>
                <Science sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h5" sx={{ mb: 2, fontWeight: 600 }}>
                  Ready to Experiment?
                </Typography>
                <Typography variant="body1" sx={{ color: 'text.secondary', maxWidth: 500, mx: 'auto' }}>
                  Select your task, dataset, and model parameters from the control panel, then click "Preview Dataset" to see your data or "Train & Visualize" to run your experiment.
                </Typography>
              </Box>
            </Fade>
          )}
          
          {/* Hyperparameter Tuning Results */}
          {tuningResult && (
            <Card variant="outlined" sx={{ mb: 3, bgcolor: 'background.paper' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ color: 'text.primary' }}>
                  Hyperparameter Tuning Results
                </Typography>
                <Alert severity="success" sx={{ mb: 2 }}>
                  Best {tuningResult.search_type} search score: {tuningResult.best_score?.toFixed(3)} 
                  (CV={tuningResult.cv_folds} folds)
                </Alert>
                <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.primary' }}>
                  Best Parameters:
                </Typography>
                <Box sx={{ mb: 2, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
                  <pre style={{ margin: 0, fontSize: '0.875rem' }}>
                    {JSON.stringify(tuningResult.best_params, null, 2)}
                  </pre>
                </Box>
                {tuningResult.image && (
                  <img 
                    src={`data:image/png;base64,${tuningResult.image}`} 
                    alt="Hyperparameter Tuning Results" 
                    style={{maxWidth: '100%'}} 
                  />
                )}
              </CardContent>
            </Card>
          )}
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
