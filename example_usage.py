"""
Example usage of iterative_lsi.py with scanpy

This example demonstrates how to use the Python LSI implementation
with scanpy AnnData objects and CSR matrices.
"""

import numpy as np
import scipy.sparse as sp

# Try to import scanpy if available
try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except ImportError:
    print("Note: scanpy not installed, using pure scipy.sparse examples")
    HAS_SCANPY = False

from iterative_lsi import IterativeLSI, compute_lsi, project_lsi


def create_example_data():
    """Create example single-cell count data."""
    np.random.seed(42)
    
    # Simulate scATAC-seq or similar count data
    n_cells = 500
    n_features = 2000
    
    # Create sparse count matrix (features x cells for LSI)
    X = sp.random(n_features, n_cells, density=0.1, format='csr', random_state=42)
    
    # Make it look more like real count data
    X.data = np.random.poisson(X.data * 3) + 1
    X.data = X.data.astype(np.float32)
    
    # Create feature names
    feature_names = [f"feature_{i:04d}" for i in range(n_features)]
    
    # Create cell names  
    cell_names = [f"cell_{i:04d}" for i in range(n_cells)]
    
    return X, feature_names, cell_names


def example_basic_lsi():
    """Basic LSI computation example."""
    print("=== Basic LSI Example ===")
    
    # Create example data
    X, feature_names, cell_names = create_example_data()
    print(f"Created example data: {X.shape} (features x cells)")
    
    # Compute LSI using the simple function interface
    result = compute_lsi(
        X,
        lsi_method=2,  # log(TF-IDF) method (good for most data)
        n_dimensions=30,
        binarize=True,  # Good for scATAC-seq data
        verbose=True
    )
    
    X_lsi = result['X_lsi']
    lsi_model = result['model']
    
    print(f"LSI result shape: {X_lsi.shape} (cells x dimensions)")
    print(f"LSI coordinates range: [{X_lsi.min():.3f}, {X_lsi.max():.3f}]")
    
    return X_lsi, lsi_model, feature_names, cell_names


def example_scanpy_integration():
    """Example of integrating with scanpy workflow."""
    if not HAS_SCANPY:
        print("Scanpy not available, skipping this example")
        return
        
    print("\n=== Scanpy Integration Example ===")
    
    # Create example data in scanpy format
    X, feature_names, cell_names = create_example_data()
    
    # Create AnnData object (scanpy's standard format)
    # Note: AnnData expects (cells x features), but our LSI expects (features x cells)
    X_scanpy = X.T.tocsr()  # Transpose to cells x features
    
    adata = ad.AnnData(
        X=X_scanpy,
        obs=pd.DataFrame(index=cell_names) if 'pd' in globals() else None,
        var=pd.DataFrame(index=feature_names) if 'pd' in globals() else None
    )
    
    print(f"AnnData object shape: {adata.shape} (cells x features)")
    
    # For LSI, we need to transpose back to (features x cells)
    X_for_lsi = adata.X.T.tocsr()
    
    # Compute LSI
    result = compute_lsi(
        X_for_lsi,
        lsi_method=2,
        n_dimensions=30,
        binarize=True,
        outlier_quantiles=(0.02, 0.98),  # Filter extreme outliers
        verbose=True
    )
    
    X_lsi = result['X_lsi']
    
    # Add LSI coordinates to AnnData object
    # Note: We may have fewer cells due to filtering
    if X_lsi.shape[0] < adata.n_obs:
        print(f"Note: {adata.n_obs - X_lsi.shape[0]} cells were filtered during LSI")
        # In practice, you'd want to track which cells were kept
        
    # For demonstration, just show the result
    print(f"LSI coordinates: {X_lsi.shape}")
    print("✓ Successfully integrated with scanpy workflow")


def example_lsi_methods_comparison():
    """Compare different LSI methods."""
    print("\n=== LSI Methods Comparison ===")
    
    X, _, _ = create_example_data()
    
    methods = [
        (1, "tf-logidf", "TF with log(1+N/df) IDF"),
        (2, "log(tf-idf)", "log(TF-IDF * scale + 1)"), 
        (3, "logtf-logidf", "log(TF+1) with log(1+N/df) IDF")
    ]
    
    results = {}
    
    for method_id, method_name, description in methods:
        print(f"\nMethod {method_id}: {description}")
        
        result = compute_lsi(
            X,
            lsi_method=method_id,
            n_dimensions=10,
            binarize=True,
            verbose=False
        )
        
        X_lsi = result['X_lsi']
        results[method_id] = X_lsi
        
        print(f"  Output shape: {X_lsi.shape}")
        print(f"  Value range: [{X_lsi.min():.3f}, {X_lsi.max():.3f}]")
        print(f"  Mean absolute value: {np.mean(np.abs(X_lsi)):.3f}")
    
    return results


def example_projection_workflow():
    """Example of training LSI and projecting new data."""
    print("\n=== LSI Projection Workflow ===")
    
    # Create training data
    X_train, _, _ = create_example_data()
    print(f"Training data: {X_train.shape}")
    
    # Create "new" data (same features, different cells)
    np.random.seed(123)
    n_new_cells = 200
    X_new = sp.random(X_train.shape[0], n_new_cells, density=0.1, format='csr')
    X_new.data = np.random.poisson(X_new.data * 3) + 1
    X_new.data = X_new.data.astype(np.float32)
    print(f"New data: {X_new.shape}")
    
    # Fit LSI on training data
    print("\n1. Training LSI model...")
    lsi_model = IterativeLSI(
        n_dimensions=20,
        lsi_method=2,
        binarize=True,
        verbose=True
    )
    
    X_train_lsi = lsi_model.fit_transform(X_train)
    print(f"Training LSI: {X_train_lsi.shape}")
    
    # Project new data
    print("\n2. Projecting new data...")
    X_new_lsi = lsi_model.project(X_new)
    print(f"New data LSI: {X_new_lsi.shape}")
    
    print("\n3. Comparison:")
    print(f"Training LSI range: [{X_train_lsi.min():.3f}, {X_train_lsi.max():.3f}]")
    print(f"New data LSI range: [{X_new_lsi.min():.3f}, {X_new_lsi.max():.3f}]")
    
    return X_train_lsi, X_new_lsi


def main():
    """Run all examples."""
    print("Python LSI Implementation Examples")
    print("=" * 50)
    
    # Basic LSI example
    X_lsi, model, features, cells = example_basic_lsi()
    
    # Scanpy integration
    example_scanpy_integration()
    
    # Methods comparison
    method_results = example_lsi_methods_comparison()
    
    # Projection workflow
    train_lsi, new_lsi = example_projection_workflow()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\nKey takeaways:")
    print("- Input should be sparse CSR matrix (features x cells)")
    print("- Multiple LSI methods available (1, 2, 3)")
    print("- Method 2 ('log(tf-idf)') is recommended for most data")
    print("- Binarization is often beneficial for count data")
    print("- Outlier filtering helps remove problematic cells")
    print("- Trained models can project new data into same space")


if __name__ == "__main__":
    main()