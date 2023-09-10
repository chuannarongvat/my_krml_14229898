def print_regressor_scores(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae

    print(f"RMSE {set_name}: {mse(y_actuals, y_preds, squared=False)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")

def assess_regressor_set(model, features, target, set_name=''):
    """Save the predictions from a trained model on a given set and print its RMSE and MAE scores

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Trained Sklearn model with set hyperparameters
    features : Numpy Array
        Features
    target : Numpy Array
        Target variable
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    preds = model.predict(features)
    print_regressor_scores(y_preds=preds, y_actuals=target, set_name=set_name)

def fit_assess_regressor(model, X_train, y_train, X_val, y_val):
    """Train a regressor model, print its RMSE and MAE scores on the training and validation set and return the trained model

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Instantiated Sklearn model with set hyperparameters
    X_train : Numpy Array
        Features for the training set
    y_train : Numpy Array
        Target for the training set
    X_train : Numpy Array
        Features for the validation set
    y_train : Numpy Array
        Target for the validation set

    Returns
    sklearn.base.BaseEstimator
        Trained model
    -------
    """
    model.fit(X_train, y_train)
    assess_regressor_set(model, X_train, y_train, set_name='Training')
    assess_regressor_set(model, X_val, y_val, set_name='Validation')
    return model

def print_classifier_scores(y_preds, y_actuals, set_name=None, metrics=None):
    """Print the Metrics Score for the provided data.
    The value of the 'average' parameter for F1 score will be determined according to the number of distinct values of the target variable: 'binary' for binary classification or 'weighted' for multi-class classification

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed
    metrics : list of str
        List of metrics to be printed ('accuracy', 'f1', 'precision', 'recall', 'roc_auc')

    Returns
    -------
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay
    import pandas as pd

    average = 'weighted' if pd.Series(y_actuals).nunique() > 2 else 'binary'
    
    if metrics is None:
        metrics = []
    
    results = {}
    
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y_actuals, y_preds)
    
    if 'f1' in metrics:
        results['f1'] = f1_score(y_actuals, y_preds, average=average)
    
    if 'precision' in metrics:
        results['precision'] = precision_score(y_actuals, y_preds, average=average)
        
    if 'recall' in metrics:
        results['recall'] = recall_score(y_actuals, y_preds, average=average)
        
    if 'roc_auc' in metrics:
        try:
            results['roc_auc'] = roc_auc_score(y_actuals, y_preds, average=average)
        except ValueError:
            pass
    
    if set_name is not None:
        print(pd.DataFrame(results, index=[set_name]))
    else:
        for metric, value in results.items():
            print(f"{metric}: {value}")

    
def assess_classifier_set(model, features, target, set_name='', metrics=None):
    """Save the predictions from a trained model on a given set and print its accuracy and F1 scores

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Trained Sklearn model with set hyperparameters
    features : Numpy Array
        Features
    target : Numpy Array
        Target variable
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    preds = model.predict(features)
    print_classifier_scores(y_preds=preds, y_actuals=target, set_name=set_name, metrics=metrics)
    
def fit_assess_classifier(model, X_train, y_train, X_val, y_val, metrics=None):
    """Train a classifier model, print its accuracy and F1 scores on the training and validation set and return the trained model

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Instantiated Sklearn model with set hyperparameters
    X_train : Numpy Array
        Features for the training set
    y_train : Numpy Array
        Target for the training set
    X_train : Numpy Array
        Features for the validation set
    y_train : Numpy Array
        Target for the validation set

    Returns
    sklearn.base.BaseEstimator
        Trained model
    -------
    """    
    model.fit(X_train, y_train)
    
    assess_classifier_set(model, X_train, y_train, set_name='Training', metrics=metrics)
    assess_classifier_set(model, X_val, y_val, set_name='Validation', metrics=metrics)
    
    print_confusion_matrix(model, X_train, y_train, set_name='Training')
    print_confusion_matrix(model, X_val, y_val, set_name='Validation')
    
    return model

def print_confusion_matrix(model, X, y, set_name=None, normalize=True):
    """Print the confusion matrix for the provided data

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
    X : Numpy Array
    y : Numpy Array
    set_name : str
    normalize : bool
    -------
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    preds = model.predict(X)
    cm = confusion_matrix(y, preds)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    display.plot(cmap='viridis', values_format='.2f' if normalize else 'd')
    plt.title(f'{set_name} Confusion Matrix')
    plt.show()