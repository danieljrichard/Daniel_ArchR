"""
Python implementation of iterative LSI for single-cell data analysis.

This module provides LSI (Latent Semantic Indexing) functionality compatible with
scanpy and scipy sparse matrices, adapted from the R ArchR package.

The core functionality includes:
- TF-IDF normalization with multiple methods
- SVD-based dimensionality reduction  
- LSI projection for new data

Compatible with scanpy CSR matrices and follows scikit-learn conventions.
"""

import numpy as np
import scipy.sparse
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import svds
import logging
from typing import Union, Optional, Tuple, Dict, Any
import warnings

# Set up logging
logger = logging.getLogger(__name__)


class IterativeLSI:
    """
    Latent Semantic Indexing (LSI) for single-cell data analysis.
    
    This class implements TF-IDF normalization followed by SVD-based dimensionality
    reduction, adapted from the ArchR R package for use with Python/scanpy workflows.
    
    Parameters
    ----------
    lsi_method : int or str, default=2
        TF-IDF normalization method:
        - 1 or "tf-logidf": TF with log(1 + n_samples/row_sums) IDF
        - 2 or "log(tf-idf)": log(TF-IDF * scale_to + 1) 
        - 3 or "logtf-logidf": log(TF + 1) with log(1 + n_samples/row_sums) IDF
    n_dimensions : int, default=50
        Number of LSI dimensions to compute
    scale_to : float, default=10000
        Scaling factor for TF-IDF normalization
    binarize : bool, default=True
        Whether to binarize the input matrix (set all non-zero values to 1)
    outlier_quantiles : tuple of float, default=(0.02, 0.98)
        Quantile range for filtering outlier cells based on total counts
    keep_zero_lsi : bool, default=False
        Whether to keep cells with zero counts in selected features
    random_state : int, default=1
        Random seed for reproducible results
    verbose : bool, default=True
        Whether to print progress messages
        
    Attributes
    ----------
    svd_ : dict
        SVD decomposition results (u, s, vt)
    row_sums_ : ndarray
        Row sums from the training matrix
    n_cols_ : int
        Number of columns in training matrix
    feature_indices_ : ndarray
        Indices of non-zero features used for LSI
    is_fitted_ : bool
        Whether the model has been fitted
    """
    
    def __init__(
        self,
        lsi_method: Union[int, str] = 2,
        n_dimensions: int = 50,
        scale_to: float = 10000,
        binarize: bool = True,
        outlier_quantiles: Optional[Tuple[float, float]] = (0.02, 0.98),
        keep_zero_lsi: bool = False,
        random_state: int = 1,
        verbose: bool = True
    ):
        self.lsi_method = lsi_method
        self.n_dimensions = n_dimensions
        self.scale_to = scale_to
        self.binarize = binarize
        self.outlier_quantiles = outlier_quantiles
        self.keep_zero_lsi = keep_zero_lsi
        self.random_state = random_state
        self.verbose = verbose
        
        # Fitted attributes
        self.svd_ = None
        self.row_sums_ = None
        self.n_cols_ = None
        self.feature_indices_ = None
        self.is_fitted_ = False
        
    def _validate_lsi_method(self) -> None:
        """Validate LSI method parameter."""
        valid_methods = {1, 2, 3, "tf-logidf", "log(tf-idf)", "logtf-logidf"}
        if self.lsi_method not in valid_methods:
            raise ValueError(f"lsi_method must be one of {valid_methods}")
    
    def _log(self, message: str) -> None:
        """Log message if verbose is True."""
        if self.verbose:
            logger.info(message)
            print(f"LSI: {message}")
    
    def _prepare_matrix(self, X: scipy.sparse.csr_matrix) -> Tuple[scipy.sparse.csr_matrix, np.ndarray]:
        """
        Prepare input matrix for LSI computation.
        
        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            Input count matrix (features x cells)
            
        Returns
        -------
        X_processed : scipy.sparse.csr_matrix
            Processed matrix ready for TF-IDF
        excluded_cells : ndarray
            Indices of excluded cells
        """
        X = X.copy()
        
        # Binarize if requested
        if self.binarize:
            self._log("Binarizing matrix")
            X.data = np.ones_like(X.data)
            
        # Compute column sums (cell totals)
        col_sums = np.array(X.sum(axis=0)).flatten()
        
        # Handle zero-sum cells
        excluded_cells = np.array([])
        if self.keep_zero_lsi:
            col_sums[col_sums == 0] = 1
        else:
            zero_cells = np.where(col_sums == 0)[0]
            if len(zero_cells) > 0:
                warning_msg = (
                    f"Filtering {len(zero_cells)} of {X.shape[1]} cells "
                    f"with zero counts in selected features"
                )
                self._log(warning_msg)
                warnings.warn(warning_msg)
                excluded_cells = zero_cells
                # Keep only non-zero cells
                keep_cells = col_sums > 0
                X = X[:, keep_cells]
                col_sums = col_sums[keep_cells]
        
        # Filter outlier cells based on total counts
        outlier_cells = np.array([])
        if self.outlier_quantiles is not None:
            q_low, q_high = np.percentile(col_sums, 
                                        [self.outlier_quantiles[0] * 100, 
                                         self.outlier_quantiles[1] * 100])
            outlier_mask = (col_sums <= q_low) | (col_sums >= q_high)
            outlier_cells = np.where(outlier_mask)[0]
            
            if len(outlier_cells) > 0:
                self._log(f"Filtering {len(outlier_cells)} outlier cells")
                # Keep non-outlier cells for training
                keep_mask = ~outlier_mask
                X_main = X[:, keep_mask]
                col_sums = col_sums[keep_mask]
                
                # Store outlier matrix for later projection
                X_outliers = X[:, outlier_mask]
                self.outlier_matrix_ = X_outliers
                X = X_main
        
        # Remove zero-sum features (rows)
        self._log("Removing zero-sum features")
        row_sums = np.array(X.sum(axis=1)).flatten()
        nonzero_features = row_sums > 0
        X = X[nonzero_features, :]
        self.feature_indices_ = np.where(nonzero_features)[0]
        self.row_sums_ = row_sums[nonzero_features]
        
        return X, excluded_cells
    
    def _compute_tfidf(self, X: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        """
        Apply TF-IDF normalization to the matrix.
        
        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            Input matrix (features x cells)
            
        Returns  
        -------
        X_tfidf : scipy.sparse.csr_matrix
            TF-IDF normalized matrix
        """
        # Compute term frequencies (TF) - normalize by column sums
        self._log("Computing term frequencies")
        col_sums = np.array(X.sum(axis=0)).flatten()
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        
        # TF normalization: divide each column by its sum
        X_tf = X.multiply(1.0 / col_sums[np.newaxis, :])
        
        # Compute inverse document frequencies (IDF) and apply
        n_cells = X.shape[1]
        
        if self.lsi_method == 1 or str(self.lsi_method).lower() == "tf-logidf":
            # Method 1: TF with log(1 + n_cells/row_sums) IDF
            self._log("Computing TF-LogIDF (method 1)")
            idf = np.log(1 + n_cells / self.row_sums_)
            X_tfidf = diags(idf, format='csr') @ X_tf
            
        elif self.lsi_method == 2 or str(self.lsi_method).lower() == "log(tf-idf)":
            # Method 2: log(TF-IDF * scale_to + 1)  
            self._log("Computing log(TF-IDF) (method 2)")
            idf = n_cells / self.row_sums_
            X_tfidf = diags(idf, format='csr') @ X_tf
            # Log transform the result
            X_tfidf.data = np.log(X_tfidf.data * self.scale_to + 1)
            
        elif self.lsi_method == 3 or str(self.lsi_method).lower() == "logtf-logidf":
            # Method 3: log(TF + 1) with log(1 + n_cells/row_sums) IDF
            self._log("Computing LogTF-LogIDF (method 3)")  
            # Log transform TF first
            X_tf.data = np.log(X_tf.data + 1)
            # Apply LogIDF
            idf = np.log(1 + n_cells / self.row_sums_)
            X_tfidf = diags(idf, format='csr') @ X_tf
            
        else:
            raise ValueError(f"Unknown LSI method: {self.lsi_method}")
            
        return X_tfidf
    
    def fit(self, X: scipy.sparse.csr_matrix) -> 'IterativeLSI':
        """
        Fit LSI model to data.
        
        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            Input count matrix (features x cells)
            
        Returns
        -------
        self : IterativeLSI
            Fitted estimator
        """
        self._validate_lsi_method()
        
        if not scipy.sparse.issparse(X) or not isinstance(X, scipy.sparse.csr_matrix):
            raise ValueError("Input X must be a scipy.sparse.csr_matrix")
            
        self._log(f"Running LSI on matrix of shape {X.shape}")
        np.random.seed(self.random_state)
        
        # Prepare matrix
        X_prep, excluded = self._prepare_matrix(X)
        self.n_cols_ = X_prep.shape[1] 
        
        # Apply TF-IDF normalization
        X_tfidf = self._compute_tfidf(X_prep)
        
        # Handle NaN values
        if np.any(np.isnan(X_tfidf.data)):
            self._log("Zeroing NaN elements")
            X_tfidf.data[np.isnan(X_tfidf.data)] = 0.0
            X_tfidf.eliminate_zeros()
        
        # Compute SVD
        self._log(f"Computing SVD with {self.n_dimensions} dimensions")
        try:
            # Use scipy's SVD (similar to irlba in R)
            u, s, vt = svds(X_tfidf, k=self.n_dimensions, random_state=self.random_state)
            
            # Sort by singular values (descending)
            idx = np.argsort(s)[::-1] 
            u, s, vt = u[:, idx], s[idx], vt[idx, :]
            
            self.svd_ = {'u': u, 's': s, 'vt': vt}
            
        except Exception as e:
            self._log(f"SVD computation failed: {e}")
            raise RuntimeError(f"SVD computation failed: {e}")
            
        self.is_fitted_ = True
        self._log("LSI fitting completed")
        
        return self
    
    def transform(self, X: Optional[scipy.sparse.csr_matrix] = None) -> np.ndarray:
        """
        Transform data to LSI space.
        
        Parameters
        ----------
        X : scipy.sparse.csr_matrix, optional
            Data to transform. If None, transforms the training data.
            
        Returns
        -------
        X_lsi : ndarray of shape (n_cells, n_dimensions)
            LSI coordinates
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before transform")
            
        if X is None:
            # Transform training data
            svd_diag = np.diag(self.svd_['s'])
            X_lsi = (svd_diag @ self.svd_['vt']).T
            return X_lsi
        else:
            # Project new data
            return self.fit_transform(X)
    
    def fit_transform(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        """
        Fit LSI model and transform data.
        
        Parameters
        ---------- 
        X : scipy.sparse.csr_matrix
            Input count matrix (features x cells)
            
        Returns
        -------
        X_lsi : ndarray of shape (n_cells, n_dimensions)  
            LSI coordinates
        """
        return self.fit(X).transform()
    
    def project(self, X: scipy.sparse.csr_matrix) -> np.ndarray:
        """
        Project new data into existing LSI space.
        
        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            New data matrix to project (features x cells)
            
        Returns
        -------
        X_proj : ndarray of shape (n_cells, n_dimensions)
            Projected LSI coordinates
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before projection")
            
        self._log(f"Projecting data of shape {X.shape}")
        
        # Use same feature subset as training
        X_proj = X[self.feature_indices_, :]
        
        # Binarize if needed
        if self.binarize:
            X_proj.data = np.ones_like(X_proj.data)
            
        # Compute TF
        col_sums = np.array(X_proj.sum(axis=0)).flatten()
        if self.keep_zero_lsi:
            col_sums[col_sums == 0] = 1
        else:
            # Filter zero-sum cells
            if np.any(col_sums == 0):
                warning_msg = f"Filtering {np.sum(col_sums == 0)} cells with zero counts"
                warnings.warn(warning_msg)
                keep_cells = col_sums > 0
                X_proj = X_proj[:, keep_cells]
                col_sums = col_sums[keep_cells]
        
        # TF normalization
        X_proj = X_proj.multiply(1.0 / col_sums[np.newaxis, :])
        
        # Apply same IDF transformation as training
        n_cells_train = self.n_cols_
        
        if self.lsi_method == 1 or str(self.lsi_method).lower() == "tf-logidf":
            idf = np.log(1 + n_cells_train / self.row_sums_)
            X_proj = diags(idf, format='csr') @ X_proj
            
        elif self.lsi_method == 2 or str(self.lsi_method).lower() == "log(tf-idf)":
            idf = n_cells_train / self.row_sums_
            X_proj = diags(idf, format='csr') @ X_proj
            X_proj.data = np.log(X_proj.data * self.scale_to + 1)
            
        elif self.lsi_method == 3 or str(self.lsi_method).lower() == "logtf-logidf":
            X_proj.data = np.log(X_proj.data + 1)
            idf = np.log(1 + n_cells_train / self.row_sums_)
            X_proj = diags(idf, format='csr') @ X_proj
        
        # Handle NaN values
        if np.any(np.isnan(X_proj.data)):
            X_proj.data[np.isnan(X_proj.data)] = 0.0
            X_proj.eliminate_zeros()
        
        # Project using V matrix
        self._log("Computing projection coordinates")
        V = X_proj.T @ self.svd_['u'] @ np.diag(1.0 / self.svd_['s'])
        
        # Compute final coordinates
        svd_diag = np.diag(self.svd_['s'])
        X_lsi_proj = (svd_diag @ V.T).T
        
        return X_lsi_proj


def compute_lsi(
    X: scipy.sparse.csr_matrix,
    lsi_method: Union[int, str] = 2,
    n_dimensions: int = 50,
    scale_to: float = 10000,
    binarize: bool = True,
    outlier_quantiles: Optional[Tuple[float, float]] = (0.02, 0.98),
    keep_zero_lsi: bool = False,
    random_state: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute LSI (Latent Semantic Indexing) for single-cell data.
    
    This function provides a scikit-learn style interface to LSI computation,
    compatible with scanpy workflows and scipy sparse matrices.
    
    Parameters
    ----------
    X : scipy.sparse.csr_matrix of shape (n_features, n_cells)
        Input count matrix 
    lsi_method : int or str, default=2
        TF-IDF normalization method (see IterativeLSI class)
    n_dimensions : int, default=50
        Number of LSI dimensions
    scale_to : float, default=10000
        Scaling factor for normalization
    binarize : bool, default=True
        Whether to binarize input matrix
    outlier_quantiles : tuple or None, default=(0.02, 0.98)
        Quantile range for outlier filtering
    keep_zero_lsi : bool, default=False
        Whether to keep zero-count cells
    random_state : int, default=1
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print progress
        
    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'X_lsi': LSI coordinates (n_cells, n_dimensions)
        - 'model': fitted IterativeLSI model
        - 'feature_names': names of features used (if available)
    
    Examples
    --------
    >>> import scipy.sparse as sp
    >>> import numpy as np
    >>> # Create example data (features x cells)
    >>> X = sp.random(1000, 100, density=0.1, format='csr', random_state=42)
    >>> result = compute_lsi(X, n_dimensions=10, verbose=True)
    >>> X_lsi = result['X_lsi']
    >>> print(f"LSI shape: {X_lsi.shape}")
    """
    
    # Initialize and fit LSI model
    lsi_model = IterativeLSI(
        lsi_method=lsi_method,
        n_dimensions=n_dimensions,
        scale_to=scale_to,
        binarize=binarize,
        outlier_quantiles=outlier_quantiles,
        keep_zero_lsi=keep_zero_lsi,
        random_state=random_state,
        verbose=verbose
    )
    
    # Fit and transform
    X_lsi = lsi_model.fit_transform(X)
    
    # Prepare result
    result = {
        'X_lsi': X_lsi,
        'model': lsi_model,
        'feature_names': None  # Could be populated if feature names available
    }
    
    return result


def project_lsi(
    X: scipy.sparse.csr_matrix,
    lsi_model: IterativeLSI,
    verbose: bool = True
) -> np.ndarray:
    """
    Project new data into existing LSI space.
    
    Parameters
    ----------
    X : scipy.sparse.csr_matrix of shape (n_features, n_cells)
        New data to project
    lsi_model : IterativeLSI 
        Fitted LSI model
    verbose : bool, default=True
        Whether to print progress
        
    Returns
    -------
    X_proj : ndarray of shape (n_cells, n_dimensions)
        Projected LSI coordinates
    """
    
    if verbose:
        print(f"LSI Projection: Projecting {X.shape[1]} cells")
        
    X_proj = lsi_model.project(X)
    
    if verbose:
        print(f"LSI Projection: Completed, output shape {X_proj.shape}")
        
    return X_proj