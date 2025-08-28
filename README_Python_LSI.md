# Python LSI Implementation for Single-Cell Data

This repository provides a Python implementation of Latent Semantic Indexing (LSI) adapted from the R ArchR package, specifically designed to work with scanpy and scipy sparse matrices.

## Overview

The LSI (Latent Semantic Indexing) algorithm is commonly used in single-cell genomics for dimensionality reduction, especially for scATAC-seq data. This implementation provides:

- **TF-IDF normalization** with multiple methods
- **SVD-based dimensionality reduction** 
- **LSI projection** for new data
- **Scanpy compatibility** with CSR matrices
- **Scikit-learn style API**

## Files

- `iterative_lsi.py` - Main implementation with `IterativeLSI` class and utility functions
- `test_iterative_lsi.py` - Comprehensive test suite
- `example_usage.py` - Usage examples and demonstrations

## Installation

The implementation requires:

```bash
pip install numpy scipy
```

Optional dependencies for examples:
```bash
pip install scanpy pandas  # For scanpy integration examples
```

## Quick Start

### Basic Usage

```python
import numpy as np
import scipy.sparse as sp
from iterative_lsi import compute_lsi

# Create or load your count matrix (features x cells)
# This should be a scipy.sparse.csr_matrix
X = sp.random(1000, 100, density=0.1, format='csr')

# Compute LSI
result = compute_lsi(
    X,
    lsi_method=2,        # log(TF-IDF) method
    n_dimensions=30,     # Number of LSI dimensions
    binarize=True,       # Binarize counts (good for scATAC-seq)
    verbose=True
)

X_lsi = result['X_lsi']  # LSI coordinates (n_cells, n_dimensions)
model = result['model']  # Fitted model for projecting new data
```

### Scanpy Integration

```python
import scanpy as sc
from iterative_lsi import compute_lsi

# Load your data with scanpy
adata = sc.read_h5ad('your_data.h5ad')

# LSI expects (features x cells), but scanpy uses (cells x features)
X_for_lsi = adata.X.T.tocsr()

# Compute LSI
result = compute_lsi(
    X_for_lsi,
    lsi_method=2,
    n_dimensions=30,
    binarize=True
)

X_lsi = result['X_lsi']

# Add to your AnnData object (handling potential cell filtering)
# Note: Some cells may be filtered during LSI computation
print(f"Original: {adata.n_obs} cells, LSI: {X_lsi.shape[0]} cells")
```

### Advanced Usage with Class Interface

```python
from iterative_lsi import IterativeLSI

# Create LSI model
lsi_model = IterativeLSI(
    lsi_method=2,
    n_dimensions=30,
    binarize=True,
    outlier_quantiles=(0.02, 0.98),  # Filter extreme outliers
    random_state=42
)

# Fit and transform
X_lsi = lsi_model.fit_transform(X_train)

# Project new data
X_new_lsi = lsi_model.project(X_new)
```

## LSI Methods

Three TF-IDF normalization methods are available:

### Method 1: `"tf-logidf"` 
- Term Frequency with log(1 + N/document_frequency) IDF
- Good for binary data
- Adapted from Cusanovich et al.

### Method 2: `"log(tf-idf)"` (Recommended)
- log(TF-IDF * scale_factor + 1) 
- Good for most count data
- Adapted from Stuart et al.

### Method 3: `"logtf-logidf"`
- log(TF + 1) with log(1 + N/document_frequency) IDF
- Alternative for count data

## Parameters

### Core Parameters

- **`lsi_method`**: Method for TF-IDF normalization (1, 2, 3 or string names)
- **`n_dimensions`**: Number of LSI dimensions to compute (default: 50)
- **`binarize`**: Whether to binarize input matrix (default: True)
- **`scale_to`**: Scaling factor for normalization (default: 10000)

### Data Filtering Parameters

- **`outlier_quantiles`**: Tuple of quantiles for filtering outlier cells (default: (0.02, 0.98))
- **`keep_zero_lsi`**: Whether to keep cells with zero counts (default: False)

### Technical Parameters

- **`random_state`**: Random seed for reproducibility (default: 1)
- **`verbose`**: Whether to print progress messages (default: True)

## Input Format

The implementation expects:
- **Sparse CSR matrix** (`scipy.sparse.csr_matrix`)
- **Shape**: (n_features, n_cells) - features as rows, cells as columns
- **Data type**: Typically integer counts, but float is supported
- **Non-negative values**: LSI is designed for count-like data

For scanpy users: transpose your AnnData.X matrix since scanpy uses (cells, features).

## Output Format

### LSI Coordinates
- **Shape**: (n_cells, n_dimensions)
- **Type**: numpy.ndarray (dense matrix)
- **Values**: Real-valued LSI coordinates

Note: The number of output cells may be less than input due to filtering of:
- Cells with zero counts in selected features
- Outlier cells (if `outlier_quantiles` is set)

## Differences from R ArchR Implementation

### Similarities
- Same TF-IDF methods and mathematical operations
- Same SVD-based approach for dimensionality reduction
- Similar outlier filtering and data preprocessing

### Differences
- Uses `scipy.sparse.linalg.svds` instead of R's `irlba`
- Python/numpy arrays instead of R matrices
- Simplified logging (Python logging vs ArchR's custom system)
- No dependency on Arrow files (works directly with matrices)

## Performance Notes

- **Memory efficient**: Uses sparse matrices throughout computation
- **Scalable**: SVD computation is the main bottleneck
- **Speed**: Comparable to R implementation for similar matrix sizes
- **Recommended**: Use `binarize=True` for large sparse matrices

## Testing

Run the comprehensive test suite:

```bash
python test_iterative_lsi.py
```

Tests cover:
- Basic LSI functionality with all methods
- LSI projection capabilities  
- Scanpy compatibility
- Edge cases and error handling

## Examples

See `example_usage.py` for detailed examples including:
- Basic LSI computation
- Scanpy integration workflow
- Comparison of different LSI methods
- Training and projection workflow

```bash
python example_usage.py
```

## Troubleshooting

### Common Issues

1. **"Input X must be a scipy.sparse.csr_matrix"**
   - Convert your matrix: `X = X.tocsr()`
   - For dense matrices: `X = sp.csr_matrix(X)`

2. **"Wrong matrix orientation"**
   - Ensure features are rows, cells are columns
   - For scanpy: use `X.T.tocsr()` to transpose

3. **"Too many cells filtered"**
   - Adjust `outlier_quantiles` or set to `None`
   - Check your data for excessive sparsity
   - Use `keep_zero_lsi=True` if needed

4. **SVD computation fails**
   - Try reducing `n_dimensions`
   - Check for invalid values (NaN, inf)
   - Ensure matrix has sufficient non-zero elements

### Memory Issues

For very large matrices:
- Use `binarize=True` to reduce memory usage
- Consider subsampling cells for initial analysis
- Monitor memory usage during SVD computation

## Citation

If you use this implementation, please cite the original ArchR paper:

Granja, Jeffrey M., et al. "ArchR is a scalable software package for integrative single-cell chromatin accessibility analysis." Nature genetics 53.3 (2021): 403-411.

## License

This implementation maintains compatibility with the original ArchR license terms.