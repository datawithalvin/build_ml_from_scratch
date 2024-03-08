from .base import baseKNN
import numpy as np
import pandas as pd
from collections import Counter

class classificationKNN(baseKNN):
    def __init__(self, k=3, metric="euclidean", p=2):
        super().__init__(k, metric, p)

    # *************************************
        
    def predict(self, X):
        """
        Predict the class label and probability score for each sample in X
        Parameters:
            - X (array or dataframe): Input samples; [n_samples, n_features]
        Returns:
            - predictions (list of dict): Predicted class and their proba score.   
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = [] # init empty list to store predictions
        for i in X:
            neighbors_labels, _ = self._get_nearest_neighbors(i.reshape(1, -1))

            # count
            votes = Counter(neighbors_labels)

            majority_vote_class, majority_vote_count = votes.most_common(1)[0]
            prob_score = majority_vote_count / self.k
            predictions.append(({"class":majority_vote_class, "probability":prob_score}))

        return predictions