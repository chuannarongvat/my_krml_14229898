import pandas as pd

def pop_target(df, target_col):
    """Extract target variable from dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    target_col : str
        Name of the target variable

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    pd.Series
        Subsetted Pandas dataframe containing the target
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)

    return df_copy, target

def split_sets_random(features, target, test_ratio=0.2):
    """Split sets randomly

    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe
    target : pd.Series
        Target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    from sklearn.model_selection import train_test_split

    val_ratio = test_ratio / (1 - test_ratio)
    X_data, X_test, y_data, y_test = train_test_split(features, target, test_size=test_ratio, random_state=8)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=val_ratio, random_state=8)

    return X_train, y_train, X_val, y_val, X_test, y_test

def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    """
    import numpy as np

    if X_train is not None:
        np.save(f'{path}X_train', X_train)
    if X_val is not None:
        np.save(f'{path}X_val',   X_val)
    if X_test is not None:
        np.save(f'{path}X_test',  X_test)
    if y_train is not None:
        np.save(f'{path}y_train', y_train)
    if y_val is not None:
        np.save(f'{path}y_val',   y_val)
    if y_test is not None:
        np.save(f'{path}y_test',  y_test)

def load_sets(path='../data/processed/'):
    """Load the different locally save sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    import numpy as np
    import os.path

    X_train = np.load(f'{path}X_train.npy', allow_pickle=True) if os.path.isfile(f'{path}X_train.npy') else None
    X_val   = np.load(f'{path}X_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}X_val.npy')   else None
    X_test  = np.load(f'{path}X_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}X_test.npy')  else None
    y_train = np.load(f'{path}y_train.npy', allow_pickle=True) if os.path.isfile(f'{path}y_train.npy') else None
    y_val   = np.load(f'{path}y_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}y_val.npy')   else None
    y_test  = np.load(f'{path}y_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}y_test.npy')  else None

    return X_train, y_train, X_val, y_val, X_test, y_test

def missing_values(features):
    """Count the number of missing values for each feature

    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe

    Returns
    -------
    dict
        Dictionary containing the number of missing values for each feature
    """
    missing_values = features.isnull().sum()
    
    if missing_values.sum() == 0:
        print('No missing values')
    else:
        missing_percentage = (features.isna().mean() * 100 ).round(2)
        
        missing_data = pd.DataFrame({
            'features': missing_values.index,
            'missing values': missing_values.values,
            'missing percentage': missing_percentage.values
        })
        
        missing_data = missing_data[missing_data['missing values'] > 0]
        missing_data = missing_data.sort_values(by='missing values', ascending=False)
        
        print('Features with missing values:')
        print(missing_data)
        
def cat_num_split(features):
    """Split categorical and numerical columns

    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe

    Returns
    -------
    list
        List of categorical columns
    list
        List of numerical columns
    """
    cat_features = [feature for feature in features.columns if features[feature].dtypes=='object']
    num_features = [feature for feature in features.columns if features[feature].dtypes!='object']
    
    return cat_features, num_features

def count_unique_values(features, cat_features):
    """Count the number of unique values for each categorical column

    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe
    cat_cols : list
        List of categorical columns

    Returns
    -------
    dict
        Dictionary containing the number of unique values for each categorical column
    """
    unique_values = {}
    for feature in cat_features:
        unique_values[feature] = len(features[feature].unique())
    
    return unique_values

def plot_cat_feature(feautres, cat_feature, target):
    """Plot the Categorical Features with the target feature
    
    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe
    cat_feature : str
    target : str
    
    Returns
    -------
    Distribution Graph
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    palette='ch:.25'
    
    ax = sns.countplot(x=cat_feature, data=feautres, hue=target, palette=palette)
    
    total = len(feautres)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center')
        
    plt.xticks(rotation=90)
    plt.title(cat_feature)
    plt.show()
    
def plot_num_feature(feautres, num_feature, target):
    """Plot the Numerical Features with the target feature
    
    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe
    num_feature : str
    target : str
    
    Returns
    -------
    Distribution Graph
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    palette = 'ch:.25'
    
    num_bins = int(np.sqrt(len(feautres[num_feature])))
        
    sns.histplot(data=feautres, x=num_feature, kde=True, hue=target, palette=palette, bins=num_bins)
        
    mean = feautres[num_feature].mean()
    median = feautres[num_feature].median()
        
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')                   
    plt.legend()
    plt.title(num_feature)
    plt.show()
    
def plot_target(features, target):
    """Plot the target feature
    
    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe
    target : str
    
    Returns
    -------
    Distribution Graph
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    palette = 'ch:.25'
    
    ax = sns.countplot(x=target, data=features, palette=palette)
    
    total = len(features)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center')
    
    plt.title('target')
    plt.show()
    
def plot_correlation(features, target):
    """Plot the correlation between the features and the target feature
    
    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe containing the numerical features only.
    target : str
    
    Returns
    -------
    Horizontal Bar Graph
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    palette = 'ch:.25'
    
    df_corr = features.corr()[target].sort_values(ascending=False).reset_index()
    df_corr.columns = ['Feature', 'Correlation']
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Feature', y='Correlation', data=df_corr, palette=palette)
    
    plt.title('Correlation between the features and target')
    plt.xticks(rotation=90)
    plt.show