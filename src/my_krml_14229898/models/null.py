import numpy as np
class NullRegressor:
    """
    Class used as baseline model for regression problem
    ...

    Attributes
    ----------
    y : Numpy Array-like
        Target variable
    pred_value : Float
        Value to be used for prediction
    preds : Numpy Array
        Predicted array

    Methods
    -------
    fit(y)
        Store the input target variable and calculate the predicted value to be used
    predict(y)
        Generate the predictions
    fit_predict(y)
        Perform a fit followed by predict
    """


    def __init__(self):
        self.y = None
        self.pred_value = None
        self.preds = None

    def fit(self, y):
        self.y = y
        self.pred_value = y.mean()

    def predict(self, y):
        self.preds = np.full((len(y), 1), self.pred_value)
        return self.preds

    def fit_predict(self, y):
        self.fit(y)
        return self.predict(self.y)
    
from scipy.stats import mode

class NullClassifier:
    """
    Class used as a baseline model for classification problems.
    ...

    Attributes
    ----------
    y : Numpy Array-like
        Target variable
    pred_value : Float
        Value to be used for prediction
    preds : Numpy Array
        Predicted array

    Methods
    -------
    fit(y)
        Store the input target variable and calculate the predicted value to be used
    predict(y)
        Generate the predictions
    fit_predict(y)
        Perform a fit followed by predict
    """

    def __init__(self):
        self.y = None
        self.pred_value = None
        self.preds = None

    def fit(self, y):
        self.y = y
        zero_count = np.count_nonzero(y == 0)
        one_count = np.count_nonzero(y == 1)

        # Check if there are equal numbers of 0s and 1s in y (SMOTE only) 
        if zero_count == one_count:
            self.pred_value = 0
            print("SMOTE: Number of 0s and 1s in y are equal. Defaulting to 0.")
        else:
            try:
                mode_result = mode(y)
                self.pred_value = mode_result.mode.item()
                print("Mode calculation successful. Mode:", self.pred_value)
            except Exception as e:
                print("Mode calculation error:", e)

    def predict(self, y):
        self.preds = np.full((len(y), 1), self.pred_value)
        return self.preds

    def fit_predict(self, y):
        self.fit(y)
        return self.predict(self.y)