from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

def custom_sampler_ratio(X,y,undersampler_ratio = 1.0):
    UnderSampler = RandomUnderSampler(sampling_strategy = undersampler_ratio, random_state = 42)
    X_res, y_res = UnderSampler.fit_resample(X,y)
    
    OverSampler = SMOTE(sampling_strategy = 'not majority', random_state=42)
    X_resampled, y_resampled = OverSampler.fit_resample(X_res, y_res)
    return X_resampled, y_resampled

def business_cost(y_true, y_pred, fn_cost=10, fp_cost=1):
    """
    Business cost
    
    Args:
    y_true : pd.Series
      labels réels 
    y_pred : pd.Series
      labels prédits

    Return:
      metric to minimise

    """
    _, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fn_cost * fn + fp_cost * fp