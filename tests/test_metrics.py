"""Tests for metric calculation and fairness."""

import pytest
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from eglp.metrics import compute_classification_metrics, EpochMetrics

def test_compute_classification_metrics_perfect():
    """Test metrics on perfect prediction."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    
    metrics = compute_classification_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0

def test_compute_classification_metrics_random():
    """Test metrics on random prediction."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    
    metrics = compute_classification_metrics(y_true, y_pred)
    
    # Acc = 0.5 (2/4)
    assert metrics["accuracy"] == 0.5
    
    # Precision: Class 0: TP=1, FP=1 -> 0.5; Class 1: TP=1, FP=1 -> 0.5. Macro = 0.5
    # Recall: Class 0: TP=1, FN=1 -> 0.5; Class 1: TP=1, FN=1 -> 0.5. Macro = 0.5
    # F1: 0.5
    
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5

def test_compute_classification_metrics_zero_division():
    """Test zero division handling (no predictions for a class)."""
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    
    # Sklearn zero_division=0 should handle this
    metrics = compute_classification_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == 0.0
    # Precision Class 0: TP=0, FP=0 -> 0. 
    # Precision Class 1: TP=0, FP=3 -> 0.
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0

def test_epoch_metrics_fields():
    """Ensure EpochMetrics has new fields."""
    m = EpochMetrics(
        epoch=1,
        train_loss=0.5,
        train_accuracy=0.8,
        val_precision=0.7,
        val_recall=0.6,
        val_f1=0.65
    )
    
    assert m.val_precision == 0.7
    assert m.val_recall == 0.6
    assert m.val_f1 == 0.65

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
