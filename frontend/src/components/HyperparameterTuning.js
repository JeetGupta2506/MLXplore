import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  Alert,
  Chip,
  Stack,
  Divider,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid
} from '@mui/material';
import { ExpandMore, Tune, AutoFixHigh } from '@mui/icons-material';

const PARAM_GRIDS = {
  'Logistic Regression': {
    C: [0.01, 0.1, 1, 10],
    solver: ['lbfgs', 'liblinear', 'saga'],
    max_iter: [100, 500, 1000]
  },
  SVM: {
    C: [0.1, 1, 10],
    kernel: ['rbf', 'linear', 'poly'],
    gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
  },
  'Random Forest': {
    n_estimators: [50, 100, 200],
    max_depth: [3, 5, 10, null],
    min_samples_split: [2, 5, 10]
  },
  KNN: {
    n_neighbors: [3, 5, 7, 9],
    weights: ['uniform', 'distance'],
    p: [1, 2]
  }
};

function HyperparameterTuning({ task, dataset, model, onTuningComplete }) {
  const [paramGrid, setParamGrid] = useState(PARAM_GRIDS[model] || {});
  const [searchType, setSearchType] = useState('grid');
  const [cvFolds, setCvFolds] = useState(5);
  const [nIter, setNIter] = useState(10);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleTune = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/tune', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task,
          dataset,
          model,
          param_grid: paramGrid,
          search_type: searchType,
          cv_folds: cvFolds,
          n_iter: nIter
        })
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
        onTuningComplete(data.best_params);
      }
    } catch (err) {
      setError(`Tuning failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const updateParamGrid = (param, values) => {
    setParamGrid(prev => ({
      ...prev,
      [param]: values.split(',').map(v => {
        const trimmed = v.trim();
        if (trimmed === 'null') return null;
        if (!isNaN(trimmed)) return Number(trimmed);
        return trimmed;
      })
    }));
  };

  return (
    <Accordion>
      <AccordionSummary expandIcon={<ExpandMore />}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Tune sx={{ color: 'primary.main' }} />
          <Typography variant="h6">Hyperparameter Tuning</Typography>
        </Box>
      </AccordionSummary>
      <AccordionDetails>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Search Type</InputLabel>
              <Select
                value={searchType}
                label="Search Type"
                onChange={(e) => setSearchType(e.target.value)}
              >
                <MenuItem value="grid">Grid Search</MenuItem>
                <MenuItem value="random">Random Search</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6} md={3}>
            <TextField
              label="CV Folds"
              type="number"
              value={cvFolds}
              onChange={(e) => setCvFolds(Number(e.target.value))}
              size="small"
              fullWidth
              inputProps={{ min: 2, max: 10 }}
            />
          </Grid>
          {searchType === 'random' && (
            <Grid item xs={6} md={3}>
              <TextField
                label="Iterations"
                type="number"
                value={nIter}
                onChange={(e) => setNIter(Number(e.target.value))}
                size="small"
                fullWidth
                inputProps={{ min: 5, max: 100 }}
              />
            </Grid>
          )}
        </Grid>

        <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
          Parameter Grid:
        </Typography>
        {Object.entries(paramGrid).map(([param, values]) => (
          <TextField
            key={param}
            label={param}
            value={Array.isArray(values) ? values.join(', ') : values}
            onChange={(e) => updateParamGrid(param, e.target.value)}
            fullWidth
            size="small"
            sx={{ mb: 1 }}
            helperText="Comma-separated values"
          />
        ))}

        <Button
          variant="contained"
          onClick={handleTune}
          disabled={loading || task === 'clustering'}
          startIcon={<AutoFixHigh />}
          sx={{ mt: 2 }}
          fullWidth
        >
          {loading ? 'Tuning...' : 'Start Hyperparameter Tuning'}
        </Button>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        {result && (
          <Card sx={{ mt: 2, bgcolor: 'success.light', color: 'success.contrastText' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Best Score: {result.best_score?.toFixed(4)}
              </Typography>
              <Typography variant="subtitle2" gutterBottom>
                Best Parameters:
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {Object.entries(result.best_params).map(([key, value]) => (
                  <Chip
                    key={key}
                    label={`${key}: ${value}`}
                    size="small"
                    sx={{ bgcolor: 'success.dark', color: 'white' }}
                  />
                ))}
              </Stack>
              {result.image && (
                <Box sx={{ mt: 2, textAlign: 'center' }}>
                  <img
                    src={`data:image/png;base64,${result.image}`}
                    alt="Tuning Results"
                    style={{ maxWidth: '100%', maxHeight: '300px', objectFit: 'contain' }}
                  />
                </Box>
              )}
            </CardContent>
          </Card>
        )}
      </AccordionDetails>
    </Accordion>
  );
}

export default HyperparameterTuning;