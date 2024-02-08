import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np 
class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []

    def fit(self, X, y):
        # Convert labels to 0 and 1
        
        y_binary = (y + 1) // 2

        # Initialize XGBoost model
        model = xgb.XGBClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate)

        # Fit the model
        model.fit(X, y_binary)

        # Save the model
        self.estimators.append(model)

    def predict(self, X):
        # Use the last trained model for prediction
        model = self.estimators[-1]

        # Convert to binary predictions
        return model.predict(X)

    def accuracy(self, X, y):
        # Make predictions
        y_pred = self.predict(X)

        # Calculate accuracy
        accuracy = np.mean(y_pred == y)
        return accuracy

    def get_params(self, deep=True):
        return {'n_estimators': self.n_estimators, 'learning_rate': self.learning_rate}

    def set_params(self, **params):
        self.n_estimators = params['n_estimators']
        self.learning_rate = params['learning_rate']
        return self

