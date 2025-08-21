import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Stack,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper
} from '@mui/material';
import { ExpandMore, Info, DataObject } from '@mui/icons-material';

const DATASET_INFO = {
  // Classification datasets
  'Iris': {
    description: 'Classic flower classification dataset with 3 species',
    samples: 150,
    features: 4,
    classes: 3,
    type: 'Real-world',
    difficulty: 'Easy',
    use_case: 'Multi-class classification, beginner-friendly'
  },
  'Wine': {
    description: 'Wine quality classification based on chemical analysis',
    samples: 178,
    features: 13,
    classes: 3,
    type: 'Real-world',
    difficulty: 'Medium',
    use_case: 'Quality control, chemical analysis'
  },
  'Breast Cancer': {
    description: 'Breast cancer diagnosis (malignant vs benign)',
    samples: 569,
    features: 30,
    classes: 2,
    type: 'Medical',
    difficulty: 'Medium',
    use_case: 'Medical diagnosis, binary classification'
  },
  'Digits': {
    description: 'Handwritten digit recognition (0-9)',
    samples: 1797,
    features: 64,
    classes: 10,
    type: 'Computer Vision',
    difficulty: 'Medium',
    use_case: 'OCR, digit recognition'
  },
  'Moons': {
    description: 'Two interleaving half circles',
    samples: 'Variable',
    features: 2,
    classes: 2,
    type: 'Synthetic',
    difficulty: 'Medium',
    use_case: 'Non-linear classification'
  },
  'Blobs': {
    description: 'Gaussian blobs for clustering',
    samples: 'Variable',
    features: 2,
    classes: 'Variable',
    type: 'Synthetic',
    difficulty: 'Easy',
    use_case: 'Basic classification/clustering'
  },
  'Circles': {
    description: 'Concentric circles',
    samples: 'Variable',
    features: 2,
    classes: 2,
    type: 'Synthetic',
    difficulty: 'Hard',
    use_case: 'Non-linear classification'
  },
  'Gaussian Quantiles': {
    description: 'Gaussian quantiles dataset',
    samples: 'Variable',
    features: 2,
    classes: 'Variable',
    type: 'Synthetic',
    difficulty: 'Medium',
    use_case: 'Multi-class classification'
  },
  // Regression datasets
  'California Housing': {
    description: 'California housing prices prediction',
    samples: 20640,
    features: 8,
    target: 'Continuous',
    type: 'Real-world',
    difficulty: 'Medium',
    use_case: 'Price prediction, real estate'
  },
  'Synthetic': {
    description: 'Synthetic regression dataset',
    samples: 'Variable',
    features: 2,
    target: 'Continuous',
    type: 'Synthetic',
    difficulty: 'Easy',
    use_case: 'Basic regression testing'
  }
};

function DatasetInfo({ dataset, task }) {
  const info = DATASET_INFO[dataset];
  
  if (!info) return null;

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'Easy': return 'success';
      case 'Medium': return 'warning';
      case 'Hard': return 'error';
      default: return 'default';
    }
  };

  return (
    <Accordion>
      <AccordionSummary expandIcon={<ExpandMore />}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Info sx={{ color: 'info.main' }} />
          <Typography variant="h6">Dataset Information</Typography>
        </Box>
      </AccordionSummary>
      <AccordionDetails>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <DataObject />
              {dataset}
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {info.description}
            </Typography>

            <Stack direction="row" spacing={1} sx={{ mb: 2 }} flexWrap="wrap">
              <Chip label={info.type} color="primary" size="small" />
              <Chip 
                label={info.difficulty} 
                color={getDifficultyColor(info.difficulty)} 
                size="small" 
              />
            </Stack>

            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableBody>
                  <TableRow>
                    <TableCell><strong>Samples</strong></TableCell>
                    <TableCell>{info.samples}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell><strong>Features</strong></TableCell>
                    <TableCell>{info.features}</TableCell>
                  </TableRow>
                  {task === 'classification' && (
                    <TableRow>
                      <TableCell><strong>Classes</strong></TableCell>
                      <TableCell>{info.classes}</TableCell>
                    </TableRow>
                  )}
                  {task === 'regression' && (
                    <TableRow>
                      <TableCell><strong>Target</strong></TableCell>
                      <TableCell>{info.target}</TableCell>
                    </TableRow>
                  )}
                  <TableRow>
                    <TableCell><strong>Use Case</strong></TableCell>
                    <TableCell>{info.use_case}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      </AccordionDetails>
    </Accordion>
  );
}

export default DatasetInfo;