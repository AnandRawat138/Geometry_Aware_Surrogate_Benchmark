import numpy as np
import pandas as pd

def evaluate_extreme_value(df_true, df_pred, group_cols):
    """
    Evaluates worst-case/extreme value predictions.
    
    1. Determine reference maximum response per group (e.g., per geometry/frequency).
    2. Determine predicted maximum response per group.
    3. Compare predicted and reference maximum.
    4. Report absolute error and relative error.
    """
    
    # Calculate max for true responses
    true_max = df_true.groupby(group_cols)['benchmark_response'].max().reset_index()
    true_max.rename(columns={'benchmark_response': 'true_max'}, inplace=True)
    
    # Calculate max for predicted responses
    pred_max = df_pred.groupby(group_cols)['benchmark_response'].max().reset_index()
    pred_max.rename(columns={'benchmark_response': 'pred_max'}, inplace=True)
    
    # Merge
    merged = pd.merge(true_max, pred_max, on=group_cols)
    
    # Calculate errors
    merged['abs_error'] = np.abs(merged['true_max'] - merged['pred_max'])
    merged['rel_error'] = merged['abs_error'] / np.abs(merged['true_max'] + 1e-9)
    
    # Aggregate to single metrics
    worst_case_abs_error = merged['abs_error'].max()
    worst_case_rel_error = merged['rel_error'].max()
    mean_extreme_abs_error = merged['abs_error'].mean()
    
    return {
        "worst_case_abs_error": worst_case_abs_error,
        "worst_case_rel_error": worst_case_rel_error,
        "mean_extreme_abs_error": mean_extreme_abs_error
    }, merged
