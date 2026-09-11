import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculates the general regression evaluation metrics.
    - RMSE
    - R²
    - Maximum absolute error
    - Tail/high-percentile error (95th percentile)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    errors = np.abs(y_true - y_pred)
    max_abs_error = np.max(errors)
    tail_error_95 = np.percentile(errors, 95)
    
    return {
        "rmse": rmse,
        "r2": r2,
        "max_abs_error": max_abs_error,
        "tail_error_95": tail_error_95
    }
