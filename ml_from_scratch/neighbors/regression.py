from .base import baseKNN
import numpy as np
import pandas as pd


class regressionKNN(baseKNN):
    def __init__(self, k=3, metric="euclidean", p=2):
        super().__init__(k, metric, p)

    def predict(self, X):
        """
        Predict the target value for each sample in X
        Parameters:
            - X (array or dataframe): Input samples; [n_samples, n_features]
        Returns:
            - predictions (array): Predicted values.   
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = [] # init empty list to store prediction
        for i in X:
            # for each sample in X, find its nearest neighbors in the training set
            neighbors_labels, _ = self._get_nearest_neighbors(i.reshape(1, -1))
            nearest_neighbors, nearest_neighbors_distances = self._get_nearest_neighbors(i)

            # calculate the average from the neighbors
            predictions.append(np.mean(neighbors_labels))

        return np.array(predictions), np.array(nearest_neighbors), np.array(nearest_neighbors_distances)