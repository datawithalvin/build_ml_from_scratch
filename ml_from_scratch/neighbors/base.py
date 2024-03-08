import numpy as np
import pandas as pd

# ----- base class knn -----
# ----- handle distance calculation and storage of training data ----


class baseKNN:
    def __init__(self, k=3, metric="euclidean", p=2):
        """
        Initialized KNN base, default k value it's 3.
        Parameters:
            - k (int): The number of nearest neighbors
        """
        self.k = k # jumlah neighbors yg akan digunakan untuk prediksi
        self.X_train = None # placeholder aja untuk training features
        self.y_train = None # placeholder juga untuk training target
        self.metric = metric
        self.p = p

    # *************************************

    def fit(self, X, y):
        """
        Store the training set, convert to arrays if the are not already 
        Parameters:
            - X (array): The training data features
            - y (array): The training data target
        """
        # cek input pandas dataframe atau pandas series
        if isinstance(X, pd.DataFrame):
            X = X.values # convert ke array
        if isinstance(y, pd.Series):
            y = y.values

        self.X_train = np.array(X) if not isinstance(X, np.ndarray) else X # cek dan transform to array
        self.y_train = np.array(y) if not isinstance(y, np.ndarray) else y # cek dan transform to array

    # *************************************

    def _calc_distance(self, x1, x2):
        """
        Calculate distance antara 2 data point dengan beberapa pilihan distance metrics.
        Default euclidean.
        Parameters:
            - x1 (array): First data point. 
            - x2 (array): Second data point.
        Returns:
            - distance (ndarray): Computed distance.
        """
        if self.metric == "euclidean":
            return np.sqrt(np.sum((x1-x2) ** 2, axis=1))
        elif self.metric == "manhattan":
            return np.sum(np.abs(x1-x2), axis=1)
        elif self.metric == "minkowski":
            return np.sum(np.abs(x1-x2) ** self.p, axis=1) ** (1/self.p)
        else:
            raise ValueError("Unsupported metric. Choose: euclidean, manhattan, or minkowski only.")

    # *************************************

    def _get_nearest_neighbors(self, x):
        """
        Identify the nearest neighbors of a given data point.
        This function will calculates the distance from the given point to all data points in the training set,
        and then sort the distance, select the index of the k smallest distance.
        Parameters:
            - x (array): Data point for which to findthe nearest neighbors.
        Returns:
            - (array): The labels of the k-nearest neighbors. 
        """

        distances = self._calc_distance(x, self.X_train) # calculate distance x ke semua X training points
        nn_idx = np.argsort(distances)[:self.k]

        return self.y_train[nn_idx], distances[nn_idx]