import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def calculate_continuous_gini(y_true, y_pred, debug=False):
    """
    Calculate Gini coefficient for continuous target values.
    
    Parameters:
    y_true: array-like, true continuous values
    y_pred: array-like, predicted continuous values
    debug: bool, if True prints debugging information
    
    Returns:
    float: Gini coefficient
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Validation checks
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be equal")
        
    # Create DataFrame
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    # Sort by predictions
    df = df.sort_values('y_pred').reset_index(drop=True)
    
    # Calculate cumulative sum of true values
    n = len(df)
    cumsum_true = df['y_true'].cumsum()
    total_true = cumsum_true.iloc[-1]
    
    # Calculate Lorenz curve points
    # x-axis: cumulative share of population
    # y-axis: cumulative share of true values
    lorenz_x = np.arange(1, n + 1) / n
    lorenz_y = cumsum_true / total_true
    
    # Calculate area under Lorenz curve using trapezoidal rule
    area_under_lorenz = np.trapz(lorenz_y, lorenz_x)
    
    # Calculate Gini coefficient
    # Gini = 1 - 2 * area_under_lorenz
    gini = 1 - 2 * area_under_lorenz
    
    if debug:
        print(f"Number of samples: {n}")
        print(f"Mean true value: {df['y_true'].mean():.4f}")
        print(f"Mean predicted value: {df['y_pred'].mean():.4f}")
        print(f"Area under Lorenz curve: {area_under_lorenz:.4f}")
        print(f"Gini coefficient: {gini:.4f}")
        
        # Calculate correlation to help with interpretation
        correlation = np.corrcoef(y_true, y_pred)[0,1]
        print(f"Correlation between true and predicted values: {correlation:.4f}")
        
        # Print quartile information
        print("\nQuartile Analysis:")
        df['quartile'] = pd.qcut(df['y_pred'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        print(df.groupby('quartile')['y_true'].agg(['mean', 'count', 'std']))
    
    return gini

def calculate_prediction_quality(y_true, y_pred):
    """
    Calculate various metrics to assess prediction quality for continuous values
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'gini': calculate_continuous_gini(y_true, y_pred)
    }

import numpy as np
import pandas as pd
from scipy import stats

def calculate_ranking_metrics(y_true, y_pred, debug=False):
    """
    Calculate various ranking correlation metrics between true values and predictions.
    
    Parameters:
    y_true: array-like, true continuous values
    y_pred: array-like, predicted continuous values
    debug: bool, if True prints debugging information
    
    Returns:
    dict: Dictionary containing various ranking metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate ranks
    true_ranks = stats.rankdata(y_true)
    pred_ranks = stats.rankdata(y_pred)
    
    # Calculate Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(y_true, y_pred)
    
    # Calculate Kendall's Tau
    kendall_tau, kendall_p = stats.kendalltau(y_true, y_pred)
    
    # Calculate normalized ranking error
    # (sum of absolute rank differences divided by worst possible sum)
    rank_diff = np.abs(true_ranks - pred_ranks)
    n = len(y_true)
    worst_case_diff = (n * n) / 2  # Maximum possible sum of rank differences
    ranking_error = rank_diff.sum() / worst_case_diff
    
    # Calculate percentage of correctly ordered pairs
    correct_order = 0
    total_pairs = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            total_pairs += 1
            if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
               (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or \
               (y_true[i] == y_true[j] and y_pred[i] == y_pred[j]):
                correct_order += 1
    
    ordering_accuracy = correct_order / total_pairs if total_pairs > 0 else 1.0
    
    metrics = {
        'spearman_correlation': spearman_corr,
        'kendall_tau': kendall_tau,
        'ranking_error': ranking_error,
        'ordering_accuracy': ordering_accuracy
    }
    
    if debug:
        print("Ranking Metrics:")
        print(f"Spearman Correlation: {spearman_corr:.4f}")
        print(f"Kendall's Tau: {kendall_tau:.4f}")
        print(f"Normalized Ranking Error: {ranking_error:.4f}")
        print(f"Pairwise Ordering Accuracy: {ordering_accuracy:.4f}")
        
        # Show example of rank differences
        df = pd.DataFrame({
            'true_values': y_true,
            'pred_values': y_pred,
            'true_ranks': true_ranks,
            'pred_ranks': pred_ranks,
            'rank_diff': rank_diff
        }).sort_values('true_values', ascending=False).head(10)
        
        print("\nTop 10 examples (sorted by true values):")
        print(df)
        
    return metrics

def get_feature_importance(model, feature_names=None):

    imp = model.feature_importance()
    imp_dict = {}
    for i in range(len(feature_names)):
        imp_dict[feature_names[i]] = imp[i]

    df = pd.DataFrame([imp_dict]).transpose().sort_values(by=0, ascending=False)
    df.index.name="Feature"
    df.columns=["gain"]
    
    return df

def create_shap_density_plot(feature_values, shap_values, feature_name):
    """
    Create a density plot showing feature SHAP values with a trend line.
    
    Parameters:
        feature_values: array-like, feature values (numeric or categorical)
        shap_values: array-like, corresponding SHAP values
        feature_name: str, name of the feature
    """
    # Convert inputs to numpy arrays and handle categorical features
    shap_values = np.array(shap_values, dtype=float)
    is_categorical = feature_values.dtype.name in ('object', 'category')
    
    if is_categorical:
        le = LabelEncoder()
        feature_values = le.fit_transform(feature_values)
        categories = le.classes_
    else:
        feature_values = np.array(feature_values, dtype=float)
    
    # Remove NaN values
    mask = ~(np.isnan(feature_values) | np.isnan(shap_values))
    feature_values = feature_values[mask]
    shap_values = shap_values[mask]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Calculate point density
    try:
        xy = np.vstack([feature_values, shap_values])
        density = stats.gaussian_kde(xy)(xy)
        density_norm = (density - density.min()) / (density.max() - density.min())
        
        # Sort by density
        idx = density_norm.argsort()
        plt.scatter(feature_values[idx], shap_values[idx],
                   c='#2196F3', alpha=density_norm[idx], s=50,
                   label='SHAP values')
    except:
        plt.scatter(feature_values, shap_values, 
                   c='#2196F3', alpha=0.5, s=50,
                   label='SHAP values')
    
    # Add trend line
    if is_categorical:
        # Mean SHAP per category
        unique_vals = np.unique(feature_values)
        mean_shap = [np.mean(shap_values[feature_values == val]) for val in unique_vals]
        plt.plot(unique_vals, mean_shap, 'r-', linewidth=2, label='Mean SHAP')
        plt.xticks(unique_vals, categories, rotation=45, ha='right')
    else:
        # Moving average trend line
        sort_idx = np.argsort(feature_values)
        window = max(len(feature_values) // 30, 1)
        trend = np.convolve(shap_values[sort_idx], 
                           np.ones(window)/window, 'valid')
        trend_x = feature_values[sort_idx][window-1:]
        plt.plot(trend_x, trend, 'r-', linewidth=2, label='Trend')
    
    # Styling
    plt.title(f'SHAP Values Distribution for {feature_name}', fontsize=12, pad=20)
    plt.xlabel(feature_name)
    plt.ylabel('SHAP Value')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return plt

# Example usage:
"""
feature_name = 'example_feature'
plot = create_shap_density_plot(
    feature_values=df[feature_name],
    shap_values=shap_values[feature_name],
    feature_name=feature_name
)
plt.show()
"""