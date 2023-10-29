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
            results['roc_auc'] = roc_auc_score(y_actuals, y_preds)
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
    metrics : list of str
        List of metrics to be used for training and validation

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
    metrics : list of str
        List of metrics to be used for training and validation

    Returns
    sklearn.base.BaseEstimator
        Trained model
    -------
    """    
    model.fit(X_train, y_train)
    
    assess_classifier_set(model, X_train, y_train, set_name='Training', metrics=metrics)
    assess_classifier_set(model, X_val, y_val, set_name='Validation', metrics=metrics)
    
    return model

def plot_confusion_matrix(model, X, y, set_name=None, normalize=True):
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
    import numpy as np
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    y_preds = model.predict(X)
    cm = confusion_matrix(y, y_preds)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    display.plot(cmap='viridis', values_format='.2f' if normalize else 'd')
    plt.title(f'Confusion Matrix for {set_name} Set')
    plt.show()


def plot_roc_auc_curve(model, X, y, set_name=None, model_name=None):
    """Plot the Receiver Operating Characteristic Curve for the provided data
    
    Parameters
    ----------
    model: sklearn.base.BaseEstimator
    X : Numpy Array
    y : Numpy Array
    set_name : str
    model_name : str
    -------
    
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score, auc
    
    y_prob = model.predict_proba(X)[:, 1]
    
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    label = f'{model_name} (AUC = {roc_auc:.2f})' if model_name else f'(AUC = {roc_auc:.2f})'
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.plot(fpr, tpr, lw=2, marker='.', label=label)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic Curve for {set_name} Set')
    plt.legend(loc='lower right')
    plt.show()
    
def plot_importances(df, title_name=None):
    """Plot the Feature Importance for the provided data
    
    Parameters
    ----------
    df : Pandas Dataframe
    title_name : str
    -------
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.barplot(data=df, x='feature', y='importance', palette='ch:.25')
    
    if title_name is not None:
        plt.title(title_name)
    else:
        plt.title('Feature Importance')
    
    sns.barplot(data=df, x='feature', y='importance', palette='ch:.25')
    plt.xticks(rotation=90)
    plt.show()
    
def permutation_importance(df, target_feature, model, X, y, set_name=None, model_name=None):
    """
    Plot the Permutation Importance for the provided data
    
    Parameters
    ----------
    df : Pandas Dataframe
    target_feature : str
    model : sklearn.base.BaseEstimator
    X : Numpy Array
    y : Numpy Array
    set_name : str
    model_name : str
    ----------
    Returns
    permutation_importance : Pandas Dataframe
    """

    from sklearn.inspection import permutation_importance
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    r = permutation_importance(model, X, y, n_repeats=30,random_state=42)
    
    permu_imp = []
    
    for i in r.importances_mean.argsort()[::-1]:
        feature_name = df.columns[i]
        feature_imp = r.importances_mean[i]
        
        if feature_name == target_feature:
            pass
        else:
            permu_imp.append({'feature': feature_name, 'importance': feature_imp})
    
    permu_imp_df =  pd.DataFrame(permu_imp).sort_values(by='importance', ascending=False)
    
    plot_importances(permu_imp_df, title_name=f'{model_name} Permutation Importance on {set_name} Set')
    return permu_imp_df
    
def misclassified_samples_df(model, X, y):
    import pandas as pd
    """Return a dataframe containing samples that the model misclassified.

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
    X : Numpy Array
    y : Numpy Array

    Returns
    -------
    DataFrame: A dataframe containing misclassified samples with their feature values.
    """
    y_preds = model.predict(X)

    df = pd.DataFrame(X)
    df['y_true'] = y
    df['y_pred'] = y_preds

    TP_samples = df[(df['y_true'] == 1) & (df['y_pred'] == 1)].copy()
    TN_samples = df[(df['y_true'] == 0) & (df['y_pred'] == 0)].copy()
    FP_samples = df[(df['y_true'] == 0) & (df['y_pred'] == 1)].copy()
    FN_samples = df[(df['y_true'] == 1) & (df['y_pred'] == 0)].copy()

    TP_samples['classification'] = 'TP'
    TN_samples['classification'] = 'TN'
    FP_samples['classification'] = 'FP'
    FN_samples['classification'] = 'FN'

    return TP_samples, TN_samples, FP_samples, FN_samples

def plot_adaboost_feature_importances(model, feature_names):
    """Plot feature importances from an AdaBoost model.

    Parameters
    ----------
    model : sklearn's AdaBoost model instance
        The trained AdaBoost model.
    feature_names : list
        List of feature names in the order they were fed to the model.

    Returns
    -------
    None
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    importances = model.feature_importances_

    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_features = np.array(feature_names)[sorted_indices]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="ch:.25")
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances from AdaBoost')
    plt.tight_layout()
    plt.show()
    
def generate_neighbors(df, X_val_df, n_neighbors):
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute').fit(X_val_df)
    
    distances, indices = knn.kneighbors(df.iloc[:, :-3].values)
    neighbors_val = knn._fit_X[indices]
    neighbors_val = neighbors_val.reshape(-1, neighbors_val.shape[-1])
    neighbors_df = pd.DataFrame(neighbors_val, columns=df.columns[:-3])
    neighbors_df.drop_duplicates(inplace=True)
    
    return neighbors_df

def explain_instances_with_lime(df, model, training_data, num_features=20):
    """
    Explain instances in a dataframe using LIME.

    Parameters:
    - df: DataFrame containing instances to be explained.
    - classifier: The trained classifier.
    - training_data: The data used to train the classifier.
    - feature_names: List of feature names.
    - class_names: List of class names.
    - num_features: Number of top features to be shown in the explanation. Default is 20.

    Returns:
    None. Displays the LIME explanations
    """
    from lime.lime_tabular import LimeTabularExplainer
    import pandas as pd
    lime_explainer = LimeTabularExplainer(training_data=training_data.values,
                                        mode='classification',
                                        feature_names=training_data.columns,
                                        discretize_continuous=False,
                                        verbose=True)
    
    for idx, instance in df.iterrows():
        display(pd.DataFrame([instance]))
        
        exp = lime_explainer.explain_instance(
            instance.values,
            model.predict_proba,
            top_labels=1,
            num_features=num_features
        )
        exp.show_in_notebook()
        
def sample_obs(df, n_sample):
    sampled_obs = df.sample(n=n_sample, random_state=42)
    return sampled_obs