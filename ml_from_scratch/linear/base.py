import numpy as np
import pandas as pd


class baseLinearRegression:
    def __init__(self, fit_intercept=True):
        # inisialisasi setup
        self.coef_ = None
        self.intercept = None
        self.fit_intercept = fit_intercept

    # *************************************
        
    def fit(self, X, y):

        # handling different data type input
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.array(X)
        if isinstance(y, pd.Series):
            y = y.values
        else:
            y = np.array(y)
        
        # buat design matrix A
        if self.fit_intercept:
            A = np.c_[np.ones((X.shape[0], 1)), X]
        else:
            A = X

        # hitung optimal coefficients 
        theta = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y)

        # extract parameter
        if self.fit_intercept:
            self.intercept = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept = 0.0
            self.coef_ = theta

    # *************************************
    
    def predict(self, X):

        # handling different data type input
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.array(X)

        y_pred = np.dot(X, self.coef_) + self.intercept

        return y_pred