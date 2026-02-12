import numpy as np
import pytest
from scipy import sparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from embetter.tabular import IsolationForestEncoder


def test_isolation_forest_encoder_basic():
    """Test basic functionality of IsolationForestEncoder."""
    # Create synthetic tabular data
    np.random.seed(42)
    X = np.random.randn(50, 5)
    
    # Create encoder
    encoder = IsolationForestEncoder(n_estimators=10, random_state=42)
    
    # Fit and transform
    encoder.fit(X)
    X_transformed = encoder.transform(X)
    
    # Check output type and shape
    assert sparse.issparse(X_transformed)
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] > 0
    
    # Check that values are binary (0 or 1)
    unique_values = np.unique(X_transformed.toarray())
    assert set(unique_values).issubset({0, 1})
    
    # Check repr works
    assert repr(encoder)


def test_isolation_forest_encoder_unfitted():
    """Test that transform raises error when not fitted."""
    encoder = IsolationForestEncoder()
    X = np.random.randn(10, 3)
    
    with pytest.raises(ValueError, match="must be fitted"):
        encoder.transform(X)


def test_isolation_forest_encoder_pipeline():
    """Test IsolationForestEncoder in a scikit-learn pipeline."""
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    # Create a simple binary target based on first two features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create pipeline
    pipe = make_pipeline(
        StandardScaler(),
        IsolationForestEncoder(n_estimators=20, random_state=42),
        LogisticRegression(random_state=42)
    )
    
    # Fit pipeline
    pipe.fit(X, y)
    
    # Make predictions
    predictions = pipe.predict(X)
    
    # Check predictions are binary
    assert set(predictions).issubset({0, 1})
    
    # Check that we get reasonable accuracy on this simple problem
    accuracy = (predictions == y).mean()
    assert accuracy > 0.5  # Better than random


def test_isolation_forest_encoder_reproducibility():
    """Test that the encoder gives consistent results with fixed random_state."""
    X = np.random.randn(30, 4)
    
    encoder1 = IsolationForestEncoder(n_estimators=10, random_state=42)
    encoder2 = IsolationForestEncoder(n_estimators=10, random_state=42)
    
    X_transformed1 = encoder1.fit_transform(X)
    X_transformed2 = encoder2.fit_transform(X)
    
    # Check that the outputs are identical
    assert np.array_equal(X_transformed1.toarray(), X_transformed2.toarray())


def test_isolation_forest_encoder_different_params():
    """Test encoder with different parameter settings."""
    X = np.random.randn(50, 3)
    
    # Test with different n_estimators
    encoder_few = IsolationForestEncoder(n_estimators=5, random_state=42)
    encoder_many = IsolationForestEncoder(n_estimators=50, random_state=42)
    
    X_few = encoder_few.fit_transform(X)
    X_many = encoder_many.fit_transform(X)
    
    # More trees should give more features
    assert X_many.shape[1] > X_few.shape[1]
    
    # Test with max_samples parameter
    encoder_samples = IsolationForestEncoder(
        n_estimators=10, 
        max_samples=0.5,
        random_state=42
    )
    X_samples = encoder_samples.fit_transform(X)
    assert X_samples.shape[0] == X.shape[0]


def test_isolation_forest_encoder_partial_fit():
    """Test that partial_fit works (even though it's a no-op for stateful encoder)."""
    X = np.random.randn(20, 3)
    encoder = IsolationForestEncoder(n_estimators=5, random_state=42)
    
    # partial_fit should still require fit before transform
    encoder.partial_fit(X)
    
    # This encoder is stateful, so it needs explicit fit
    with pytest.raises(ValueError, match="must be fitted"):
        encoder.transform(X)
    
    # After fit, transform should work
    encoder.fit(X)
    X_transformed = encoder.transform(X)
    assert X_transformed.shape[0] == X.shape[0]