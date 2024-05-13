import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV


class my_model:
    def __init__(self):
        self.clf = None
        self.textual_cols = ["title", "location", "description"]
        self.numerical_cols = ["telecommuting", "has_company_logo", "has_questions"]

    def fit(self, X, y):
        # do not exceed 29 mins
        text_arrays = []

        # Fit and transform the textual data columns and converting to dense array
        for col in self.textual_cols:
            preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
            text_array = preprocessor.fit_transform(X[col])
            text_array = text_array.toarray()
            text_arrays.append(text_array)
            setattr(self, f"preprocessor_{col}", preprocessor)

        # getting the numerical data columns and concatenating them with textual data columns
        num_array = X[self.numerical_cols].to_numpy()
        XX = np.hstack(text_arrays + [num_array])
        XX = pd.DataFrame(XX)

        # Using SGDClassifier
        model = SGDClassifier()
        decision_keys = {"loss": ("hinge", "log_loss", "perceptron"), "penalty": ("l2", "l1"), "alpha": [0.0001, 0.01]}
        self.clf = RandomizedSearchCV(model, decision_keys, cv=5)
        self.clf.fit(XX, y)

        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        text_arrays = []

        # transforming the textual data columns based on stored preprocessor attributes
        for col in self.textual_cols:
            preprocessor = getattr(self, f"preprocessor_{col}")
            text_array = preprocessor.transform(X[col])
            text_array = text_array.toarray()
            text_arrays.append(text_array)

        # getting the numerical data columns and concatenating them with textual data columns
        num_array = X[self.numerical_cols].to_numpy()
        XX = np.hstack(text_arrays + [num_array])
        XX = pd.DataFrame(XX)

        # predicting the labels
        predictions = self.clf.predict(XX)
        return predictions
