import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import { ExpandMore, Compare, EmojiEvents } from '@mui/icons-material';

function ModelComparison({ task, dataset, models, onComparisonComplete }) {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);

  const runComparison = async () => {
    setLoading(true);
    setError(null);
    setResults([]);

    try {
      const promises = models.map(async (model) => {
        const response = await fetch('http://localhost:8000/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            task,
            dataset,
            model,
            params: {} // Use default parameters
          })
        });
        const data = await response.json();
        return { model, ...data };
      });

      const allResults = await Promise.all(promises);
      setResults(allResults.filter(r => !r.error));
      onComparisonComplete(allResults);
    } catch (err) {
      setError(`Comparison failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getMetricValue = (result) => {
    if (result.score !== undefined) return result.score;
    if (result.r2 !== undefined) return result.r2;
    return 0;
  };

  const getMetricName = () => {
    if (task === 'classification') return 'Accuracy';
    if (task === 'regression') return 'R²';
    return 'Score';
  };

  const sortedResults = [...results].sort((a, b) => getMetricValue(b) - getMetricValue(a));
  const bestModel = sortedResults[0];

  return (
    <Accordion>
      <AccordionSummary expandIcon={<ExpandMore />}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Compare sx={{ color: 'secondary.main' }} />
          <Typography variant="h6">Model Comparison</Typography>
        </Box>
      </AccordionSummary>
      <AccordionDetails>
        <Button
          variant="contained"
          color="secondary"
          onClick={runComparison}
          disabled={loading || task === 'clustering'}
          startIcon={<Compare />}
          fullWidth
          sx={{ mb: 2 }}
        >
          {loading ? 'Comparing Models...' : 'Compare All Models'}
        </Button>

        {loading && (
          <Box sx={{ mb: 2 }}>
            <LinearProgress />
            <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
              Training {models.length} models...
            </Typography>
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {results.length > 0 && (
          <TableContainer component={Paper} sx={{ mb: 2 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Rank</TableCell>
                  <TableCell>Model</TableCell>
                  <TableCell align="right">{getMetricName()}</TableCell>
                  {task === 'regression' && (
                    <TableCell align="right">MSE</TableCell>
                  )}
                  <TableCell>Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {sortedResults.map((result, index) => (
                  <TableRow key={result.model}>
                    <TableCell>
                      {index === 0 && (
                        <EmojiEvents sx={{ color: 'gold', mr: 1 }} />
                      )}
                      #{index + 1}
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" fontWeight={index === 0 ? 600 : 400}>
                        {result.model}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography
                        variant="body2"
                        color={index === 0 ? 'success.main' : 'text.primary'}
                        fontWeight={index === 0 ? 600 : 400}
                      >
                        {getMetricValue(result).toFixed(4)}
                      </Typography>
                    </TableCell>
                    {task === 'regression' && (
                      <TableCell align="right">
                        {result.mse?.toFixed(4) || 'N/A'}
                      </TableCell>
                    )}
                    <TableCell>
                      <Chip
                        label={index === 0 ? 'Best' : 'Complete'}
                        color={index === 0 ? 'success' : 'default'}
                        size="small"
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        {bestModel && (
          <Alert severity="success">
            <Typography variant="subtitle2">
              🏆 Best Model: {bestModel.model} with {getMetricName()}: {getMetricValue(bestModel).toFixed(4)}
            </Typography>
          </Alert>
        )}
      </AccordionDetails>
    </Accordion>
  );
}

export default ModelComparison;