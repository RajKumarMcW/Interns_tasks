import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self, target_column, test_size=0.2, random_state=None):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def preprocess_data(self, data):
        X = data.drop([self.target_column], axis=1)
        y = data[self.target_column]

        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=self.test_size, random_state=self.random_state
        )

        # Standardize numerical features
        numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
        X_train[numerical_columns] = self.scaler.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = self.scaler.transform(X_test[numerical_columns])

        return X_train, X_test, y_train, y_test

avocado_target_column = 'selling_price'
avocado_preprocessor = DataPreprocessor(target_column=avocado_target_column)
X_train_avocado, X_test_avocado, y_train_avocado, y_test_avocado = avocado_preprocessor.preprocess_data(df_avocado)

spam_target_column = 'target'  # Replace with your actual target column name
spam_preprocessor = DataPreprocessor(target_column=spam_target_column)
X_train_spam, X_test_spam, y_train_spam, y_test_spam = spam_preprocessor.preprocess_data(df_spam)

car_price_target_column = 'selling_price'  # Replace with your actual target column name
car_price_preprocessor = DataPreprocessor(target_column=car_price_target_column)
X_train_car_price, X_test_car_price, y_train_car_price, y_test_car_price = car_price_preprocessor.preprocess_data(df_car_price)

rain_target_column = 'RainTomorrow'  # Replace with your actual target column name
rain_preprocessor = DataPreprocessor(target_column=rain_target_column)
X_train_rain, X_test_rain, y_train_rain, y_test_rain = rain_preprocessor.preprocess_data(df_rain)
