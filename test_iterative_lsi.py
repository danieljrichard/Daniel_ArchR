#!/usr/bin/env python3
"""
Test script for iterative_lsi.py

This script creates synthetic data similar to single-cell count matrices
and tests the LSI functionality to ensure it works correctly.
"""

import numpy as np
import scipy.sparse as sp
import sys
import os

# Add the current directory to path so we can import iterative_lsi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from iterative_lsi import IterativeLSI, compute_lsi, project_lsi
    print("✓ Successfully imported iterative_lsi module")
except ImportError as e:
    print(f"✗ Failed to import iterative_lsi: {e}")
    sys.exit(1)


def create_test_data(n_features=1000, n_cells=200, density=0.1, random_state=42):
    """Create synthetic single-cell count data."""
    np.random.seed(random_state)
    
    # Create sparse count matrix (features x cells)
    X = sp.random(n_features, n_cells, density=density, format='csr', 
                  random_state=random_state)
    
    # Make it look more like count data (integers, some higher counts)
    X.data = np.random.poisson(X.data * 5) + 1
    X.data = X.data.astype(np.float32)
    
    print(f"Created test data: {X.shape} matrix with {X.nnz} non-zero elements")
    return X


def test_basic_functionality():
    """Test basic LSI functionality."""
    print("\n=== Testing Basic LSI Functionality ===")
    
    # Create test data
    X = create_test_data(n_features=500, n_cells=100, density=0.2)
    
    # Test different LSI methods
    for method in [1, 2, 3]:
        print(f"\nTesting LSI method {method}:")
        try:
            result = compute_lsi(
                X, 
                lsi_method=method,
                n_dimensions=10,
                binarize=True,
                verbose=False
            )
            
            X_lsi = result['X_lsi']
            model = result['model']
            
            print(f"  ✓ Method {method}: Output shape {X_lsi.shape}")
            print(f"  ✓ Method {method}: Model fitted successfully")
            
            # Check that we get the expected number of dimensions
            assert X_lsi.shape[1] == 10, f"Expected 10 dimensions, got {X_lsi.shape[1]}"
            
            # Check that we have reasonable values (not all zeros or NaN)
            assert not np.all(X_lsi == 0), "LSI output should not be all zeros"
            assert not np.any(np.isnan(X_lsi)), "LSI output should not contain NaN"
            
        except Exception as e:
            print(f"  ✗ Method {method} failed: {e}")
            return False
            
    return True


def test_projection():
    """Test LSI projection functionality."""
    print("\n=== Testing LSI Projection ===")
    
    # Create training data
    X_train = create_test_data(n_features=500, n_cells=100, density=0.2, random_state=42)
    
    # Create test data (same features, different cells)
    X_test = create_test_data(n_features=500, n_cells=50, density=0.2, random_state=123)
    
    try:
        # Fit LSI on training data
        print("Fitting LSI model on training data...")
        lsi_model = IterativeLSI(n_dimensions=10, verbose=False)
        lsi_model.fit(X_train)
        X_train_lsi = lsi_model.transform()
        
        print(f"  ✓ Training LSI shape: {X_train_lsi.shape}")
        
        # Project test data
        print("Projecting test data...")
        X_test_lsi = lsi_model.project(X_test)
        
        print(f"  ✓ Test LSI shape: {X_test_lsi.shape}")
        print(f"  ✓ Projection completed successfully")
        
        # Check dimensions match
        assert X_train_lsi.shape[1] == X_test_lsi.shape[1], "Dimension mismatch"
        assert not np.any(np.isnan(X_test_lsi)), "Projected data contains NaN"
        
        return True
        
    except Exception as e:
        print(f"  ✗ Projection failed: {e}")
        return False


def test_scanpy_compatibility():
    """Test compatibility with typical scanpy workflows."""
    print("\n=== Testing Scanpy Compatibility ===")
    
    try:
        # Simulate typical scanpy matrix format
        n_cells, n_features = 200, 1000
        
        # Create data in the format scanpy typically uses (cells x features)
        # But our LSI expects (features x cells), so we'll transpose
        X_scanpy_format = sp.random(n_cells, n_features, density=0.1, 
                                   format='csr', random_state=42)
        X_scanpy_format.data = np.random.poisson(X_scanpy_format.data * 3) + 1
        
        # Transpose for LSI (features x cells)  
        X_for_lsi = X_scanpy_format.T.tocsr()
        
        print(f"Scanpy-style data: {X_scanpy_format.shape} -> LSI format: {X_for_lsi.shape}")
        
        # Run LSI
        result = compute_lsi(
            X_for_lsi,
            n_dimensions=15,
            lsi_method=2,
            binarize=True,
            verbose=False
        )
        
        X_lsi = result['X_lsi']
        print(f"  ✓ LSI result shape: {X_lsi.shape}")
        print(f"  ✓ Compatible with scanpy data format")
        
        # The LSI coordinates should be (n_cells, n_dimensions)
        # Note: some cells may be filtered out due to outlier quantiles
        assert X_lsi.shape[1] == 15, f"Expected 15 dimensions, got {X_lsi.shape[1]}"
        assert X_lsi.shape[0] <= n_cells, f"Got more cells than expected: {X_lsi.shape[0]} > {n_cells}"
        print(f"  ✓ Expected up to {n_cells} cells, got {X_lsi.shape[0]} (some filtered as outliers)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Scanpy compatibility test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    # Test with very sparse data
    try:
        X_sparse = sp.random(100, 50, density=0.01, format='csr', random_state=42)
        X_sparse.data = np.ones_like(X_sparse.data)
        
        result = compute_lsi(X_sparse, n_dimensions=5, verbose=False)
        print("  ✓ Handled very sparse data")
        
    except Exception as e:
        print(f"  ✗ Very sparse data failed: {e}")
        return False
    
    # Test with different matrix formats
    try:
        X_coo = sp.random(100, 50, density=0.1, format='coo', random_state=42)
        X_csr = X_coo.tocsr()
        
        result = compute_lsi(X_csr, n_dimensions=5, verbose=False)
        print("  ✓ Handled CSR matrix format")
        
    except Exception as e:
        print(f"  ✗ CSR format test failed: {e}")
        return False
    
    # Test invalid LSI method
    try:
        X = create_test_data(100, 50, 0.1)
        lsi = IterativeLSI(lsi_method=999)
        lsi.fit(X)
        print("  ✗ Should have failed with invalid LSI method")
        return False
    except ValueError:
        print("  ✓ Properly handled invalid LSI method")
    except Exception as e:
        print(f"  ✗ Wrong exception for invalid LSI method: {e}")
        return False
        
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("Starting iterative_lsi.py test suite...")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("LSI Projection", test_projection), 
        ("Scanpy Compatibility", test_scanpy_compatibility),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*50}")
    print("TEST RESULTS:")
    print(f"{'='*50}")
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {test_name}")
        if not success:
            all_passed = False
    
    print(f"{'='*50}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)