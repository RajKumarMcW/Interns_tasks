import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessing import preprocess_data
from sklearn.metrics import r2_score
from dd.Taxi_price_linreg import LinearRegression
from dd.rain_in_australia import LogisticRegressionCustom
from dd.email_spam_classification import MultinomialNaiveBayes
from dd.credit_car_spam import KNNClassifier
from dd.car_price_predict import RandomForestRegressor
from dd.avacado import DecisionTree
from dd.salary import AdaBoostClassifierCustom
from dd.movies_hit_classification import XGBoostClassifier
import dd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
def load_dataset(dataset_choice):
    datasets = {
        1: 'G:\\New folder\\Downloads\\pack_8\\dd\\dd\\taxitrip.csv',
        2: 'https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/avocado.csv',
        3: 'G:\\New folder\\Downloads\\pack_8\\dd\\dd\\car_price_prediction.csv',
        4: 'G:\\New folder\\Downloads\\pack_8\\dd\\dd\\creditcard.csv',
        5: 'G:\\New folder\\Downloads\\pack_8\\dd\\dd\\spam.csv',
        6: 'G:\\New folder\\Downloads\\pack_8\\dd\\dd\\salary.csv',
        7: 'G:\\New folder\\Downloads\\pack_8\\dd\\dd\\weatherAUS.csv',
        8: 'G:\\New folder\\Downloads\\pack_8\\dd\\dd\\movies_metadata.csv',
    }

    if dataset_choice == 9:
        csv_path = input("Enter the path to your CSV file: ")
        dataset = pd.read_csv(csv_path)
        target_column = input("Enter the target column: ")
    elif dataset_choice in datasets:
        dataset = pd.read_csv(datasets[dataset_choice], encoding='latin1') if dataset_choice == 5 else pd.read_csv(datasets[dataset_choice])
        target_column = None
    else:
        print("Invalid dataset choice.")
        return None, None

    return dataset, target_column

def get_model_instance(model_name, task):
    regression_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTree(max_depth=None),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1),
    }

    classification_models = {
        "KNN": KNNClassifier(k=3),
        "Naive Bayes": MultinomialNaiveBayes(alpha=1.0),
        "AdaBoost": AdaBoostClassifierCustom(n_estimators=100, random_state=1),
        "Logistic Regression": LogisticRegressionCustom(learning_rate=0.01, num_iterations=1000, verbose=True),
        "XGBoost": XGBoostClassifier(),
    }

    return regression_models[model_name] if task == "regression" else classification_models[model_name]

def train_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f"Build started for {model_name}...")
    model.fit(X_train, y_train)
    print(f"{model_name} - Build completed...")

    if model_name == "Linear Regression":
        y_pred = model.predict(X_test)
        mae = sklearn.metrics.mean_absolute_error(y_pred,y_test)
        print("Mean squared error :",mae)
        mse = sklearn.metrics.mean_squared_error(y_pred,y_test)
        print("Mean squared error :",mse)
        r2 = r2_score(y_pred, y_test)
        print("R2 Score(custom):", r2)
        lr = LinearRegression()
        lr.fit(X_train,y_train)
        lrpred = lr.predict(X_test)
        mae = sklearn.metrics.mean_absolute_error(lrpred,y_test)
        print("Mean squared error(inbuilt) :",mae)
        mse = sklearn.metrics.mean_squared_error(lrpred,y_test)
        print("Mean squared error(inbuilt) :",mse)
        l_r2 = sklearn.metrics.r2_score(lrpred, y_test)
        print("R2 Score(inbuilt):", l_r2)

    elif model_name == "Decision Tree":
        y_pred = model.predict(X_test)
        mae = sklearn.metrics.mean_absolute_error(y_pred,y_test)
        print("Mean squared error :",mae)
        mse = sklearn.metrics.mean_squared_error(y_pred,y_test)
        print("Mean squared error :",mse)
        r2 = r2_score(y_pred, y_test)
        print("R2 Score(custom):", r2)
        dr = DecisionTreeRegressor()
        dr.fit(X_train,y_train)
        drpred = dr.predict(X_test)
        mae = sklearn.metrics.mean_absolute_error(drpred,y_test)
        print("Mean squared error(inbuilt) :",mae)
        mse = sklearn.metrics.mean_squared_error(drpred,y_test)
        print("Mean squared error(inbuilt) :",mse)
        d_r2 = sklearn.metrics.r2_score(drpred, y_test)
        print("R2 Score(inbuilt):", d_r2)

    elif model_name == "Random Forest":
        
        predictions = model.predict(X_test)
        mae = sklearn.metrics.mean_absolute_error(predictions,y_test)
        print("Mean squared error :",mae)
        mse = sklearn.metrics.mean_squared_error(predictions,y_test)
        print("Mean squared error :",mse)
        r2 = r2_score(predictions, y_test)
        print("R2 Score(custom):", r2)
        rdr = RandomForestRegressor()
        rdr.fit(X_train,y_train)
        rp = rdr.predict(X_test)
        mae = sklearn.metrics.mean_absolute_error(rp,y_test)
        print("Mean squared error(inbuilt) :",mae)
        mse = sklearn.metrics.mean_squared_error(rp,y_test)
        print("Mean squared error(inbuilt) :",mse)
        r2 = r2_score(rp, y_test)
        print("R2 Score(inbuilt):", r2)

    elif model_name == "KNN":
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        kn = KNeighborsClassifier()
        kn.fit(X_train,y_train)
        kn_pred = kn.predict(X_test)
        accuracy = accuracy_score(y_test, kn_pred)
        conf_matrix = confusion_matrix(y_test, kn_pred)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(classification_report(y_test, kn_pred))

    elif model_name == "Naive Bayes":
        y_pred = model.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        nb = MultinomialNaiveBayes()
        nb.fit(X_train,y_train)
        nb_pred = nb.predict(X_test)
        accuracy = accuracy_score(y_test, nb_pred)
        conf_matrix = confusion_matrix(y_test, nb_pred)
        print("Accuracy(inbuilt):", accuracy)
        print("Confusion Matrix(inbuilt):")
        print(conf_matrix)
        print("Classification Report(inbuilt):")
        print(classification_report(y_test, nb_pred))

    elif model_name == "AdaBoost":
        y_pred = model.predict(X_test)
        accuracy_custom = accuracy_score(y_test, y_pred)
        print(f'Accuracy (Custom AdaBoost): {accuracy_custom}')
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        ab = AdaBoostClassifier()
        ab.fit(X_train,y_train)
        ab_pred = ab.predict(X_test)
        accuracy = accuracy_score(y_test, ab_pred)
        conf_matrix = confusion_matrix(y_test, ab_pred)
        print("Accuracy(inbuilt):", accuracy)
        print("Confusion Matrix(inbuilt):")
        print(conf_matrix)
        print("Classification Report(inbuilt):")
        print(classification_report(y_test, ab_pred))

    elif model_name == "Logistic Regression":
        # X_train = X_train.to_numpy()
        # y_train = y_train.to_numpy()
        # model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_logr = accuracy_score(y_test, y_pred)
        print(accuracy_logr)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        lr = LogisticRegression()
        lr.fit(X_train,y_train)
        lr_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, lr_pred)
        conf_matrix = confusion_matrix(y_test, lr_pred)
        print("Accuracy(inbuilt):", accuracy)
        print("Confusion Matrix(inbuilt):")
        print(conf_matrix)
        print("Classification Report(inbuilt):")
        print(classification_report(y_test, lr_pred))

    elif model_name == "XGBoost":
        # preprocessor = preprocess_data(dataset, dataset_choice)
        # X_train, X_test, y_train, y_test = preprocessor
        print("Build started for XGBoost...")
        clf = XGBoostClassifier()
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        grid_search = GridSearchCV(clf, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
        grid_search.fit(X_train, y_train)
        print("Build completed for XGBoost...")
        best_model = grid_search.best_estimator_
        # train_accuracy = best_model.accuracy(X_train, y_train)
        # print(f"Training Accuracy: {train_accuracy}")
        test_accuracy = best_model.accuracy(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy}")
        y_pred = best_model.predict(X_test)
        conf_matrix_xgboost = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix (XGBoost):")
        print(conf_matrix_xgboost)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        xg = XGBClassifier()
        xg.fit(X_train,y_train)
        xg_pred = xg.predict(X_test)
        accuracy = accuracy_score(y_test, xg_pred)
        conf_matrix = confusion_matrix(y_test, xg_pred)
        print("Accuracy(inbuilt):", accuracy)
        print("Confusion Matrix(inbuilt):")
        print(conf_matrix)
        print("Classification Report(inbuilt):")
        print(classification_report(y_test, xg_pred))
        

    else:
        print(f"Invalid model name: {model_name}")

def main():
    print("""Choose a dataset
            (1 - Taxi_price_prediction,
             2 - Avocado,
             3 - car_price_predict, 
             4 - credit_card_spam,
             5 - email spam classification, 
             6 - Salary classification,
             7 - rain in australia, 
             8 - Movie prediction,
             9 - Custom CSV file)""")
    dataset_choice = int(input("Enter a number(1 - 9) : "))
    dataset, target_column = load_dataset(dataset_choice)

    if dataset is not None:
        if dataset_choice in range(1, 4):
            task = "regression"
        elif dataset_choice in range(4, 9):
            task = "classification"
        elif dataset_choice == 9:
            task = input("Enter the task (regression/classification): ")
        else:
            print("Invalid dataset choice.")
            return
        
        if dataset_choice == 9:
            preprocessor = preprocess_data(dataset,dataset_choice, target_column)
        else:
            preprocessor = preprocess_data(dataset,dataset_choice)

        X_train, X_test, y_train, y_test = preprocessor

        model_names = {
            "regression": ["Linear Regression", "Decision Tree", "Random Forest"],
            "classification": ["KNN", "Naive Bayes", "AdaBoost", "Logistic Regression", "XGBoost"],
        }

        print(f"Choose a {task} model:")
        for i, model in enumerate(model_names[task], start=1):
            print(f"{i}. {model}")

        model_choice = int(input("Choose a model: "))
        selected_model_name = model_names[task][model_choice - 1]

        model = get_model_instance(selected_model_name, task)
        train_model(model, X_train, y_train, X_test, y_test, selected_model_name)
        print("Prediction completed...")

if __name__ == "__main__":
    main()


############################################################################# two models

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from preprocessing import preprocess_data
# from Taxi_price_linreg import LinearRegression
# from rain_in_australia import LogisticRegressionCustom
# from sklearn.metrics import r2_score
# from email_spam_classification import MultinomialNaiveBayes
# from credit_car_spam import KNNClassifier
# from car_price_predict import RandomForestRegressor
# from avacado import DecisionTree
# from salary import AdaBoostClassifierCustom
# from movies_hit_classification import XGBoostClassifier
# from sklearn.linear_model import LinearRegression as SklearnLinearRegression
# from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier

# def load_dataset(dataset_choice):
#     datasets = {
#         1: 'C:\\Users\\dines\\OneDrive\\Documents\\pack_8\\dd\\dd\\taxitrip.csv',
#         2: 'https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/avocado.csv',
#         3: 'C:\\Users\\dines\\OneDrive\\Documents\\pack_8\\dd\\dd\\car_price_prediction.csv',
#         4: 'C:\\Users\\dines\\OneDrive\\Documents\\pack_8\\dd\\dd\\creditcard.csv',
#         5: 'C:\\Users\\dines\\OneDrive\\Documents\\pack_8\\dd\\dd\\spam.csv',
#         6: 'C:\\Users\\dines\\OneDrive\\Documents\\pack_8\\dd\\dd\\salary.csv',
#         7: 'C:\\Users\\dines\\OneDrive\\Documents\\pack_8\\dd\\dd\\weatherAUS.csv',
#         8: 'C:\\Users\\dines\\OneDrive\\Documents\\pack_8\\dd\\dd\\movies_metadata.csv',
#     }

#     if dataset_choice == 9:
#         csv_path = input("Enter the path to your CSV file: ")
#         dataset = pd.read_csv(csv_path)
#         target_column = input("Enter the target column: ")
#     elif dataset_choice in datasets:
#         dataset = pd.read_csv(datasets[dataset_choice], encoding='latin1') if dataset_choice == 5 else pd.read_csv(datasets[dataset_choice])
#         target_column = None
#     else:
#         print("Invalid dataset choice.")
#         return None, None

#     return dataset, target_column

# def get_model_instance(model_name, task):
#     regression_models = {
#         "Linear Regression": LinearRegression(),
#         "Decision Tree": DecisionTree(max_depth=None),
#         "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1),
#     }

#     classification_models = {
#         "KNN": KNNClassifier(k=3),
#         "Naive Bayes": MultinomialNaiveBayes(alpha=1.0),
#         "AdaBoost": AdaBoostClassifierCustom(n_estimators=100, random_state=1),
#         "Logistic Regression": LogisticRegressionCustom(learning_rate=0.01, num_iterations=1000, verbose=True),
#         "XGBoost": XGBoostClassifier(),
#     }

#     return regression_models[model_name] if task == "regression" else classification_models[model_name]

# def get_inbuilt_model_instance(model_name, task):
#     regression_models = {
#         "Linear Regression": SklearnLinearRegression(),
#         "Decision Tree": DecisionTreeRegressor(),
#         "Random Forest": RandomForestRegressor(),
#     }

#     classification_models = {
#         "KNN": KNeighborsClassifier(),
#         "Naive Bayes": MultinomialNB(),
#         "AdaBoost": AdaBoostClassifier(),
#         "Logistic Regression": LogisticRegression(),
#         "XGBoost": XGBClassifier(),
#     }

#     return regression_models[model_name] if task == "regression" else classification_models[model_name]

# def train_model(model, X_train, y_train, X_test, y_test, model_name):
#     print(f"Build started for {model_name} (custom model)...")
#     model.fit(X_train, y_train)
#     print(f"{model_name} (custom model) - Build completed...")

#     if model_name == "Linear Regression":
#         y_pred = model.predict(X_test)
#         r2 = r2_score(y_pred, y_test)
#         print("R2 Score (custom model):", r2)

#     elif model_name == "Decision Tree":
#         y_pred_tree = model.predict(X_test)
#         r2_tree = r2_score(y_test, y_pred_tree)
#         print("R-squared error (custom model): ", r2_tree)

#     elif model_name == "Random Forest":
#         predictions = model.predict(X_test)
#         accuracy = np.mean(np.abs(predictions - y_test))
#         print(f"Mean Absolute Error on Test Set (custom model): {accuracy}")
#         r2 = r2_score(y_test, predictions)
#         print("R-squared (custom model):", r2)

#     elif model_name == "KNN":
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         conf_matrix = confusion_matrix(y_test, y_pred)
#         print("Accuracy (custom model):", accuracy)
#         print("Confusion Matrix (custom model):")
#         print(conf_matrix)
#         print("Classification Report (custom model):")
#         print(classification_report(y_test, y_pred))

#     elif model_name == "Naive Bayes":
#         y_pred = model.predict(X_test)
#         accuracy = np.sum(y_pred == y_test) / len(y_test)
#         conf_matrix = confusion_matrix(y_test, y_pred)
#         print("Accuracy (custom model):", accuracy)
#         print("Confusion Matrix (custom model):")
#         print(conf_matrix)
#         print("Classification Report (custom model):")
#         print(classification_report(y_test, y_pred))

#     elif model_name == "AdaBoost":
#         y_pred = model.predict(X_test)
#         accuracy_custom = accuracy_score(y_test, y_pred)
#         print(f'Accuracy (Custom AdaBoost): {accuracy_custom}')
#         print("Classification Report (custom model):")
#         print(classification_report(y_test, y_pred))

#     elif model_name == "Logistic Regression":
#         X_train = X_train.to_numpy()
#         y_train = y_train.to_numpy()
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         accuracy_logr = accuracy_score(y_test, y_pred)
#         print(f"Accuracy (custom model): {accuracy_logr}")
#         print("Classification Report (custom model):")
#         print(classification_report(y_test, y_pred))

#     elif model_name == "XGBoost":
#         print("Build started for XGBoost (custom model)...")
#         clf = XGBoostClassifier()
#         param_grid = {
#             'n_estimators': [50, 100, 150],
#             'learning_rate': [0.01, 0.1, 0.2]
#         }
#         grid_search = GridSearchCV(clf, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
#         grid_search.fit(X_train, y_train)
#         print("Build completed for XGBoost (custom model)...")
#         best_model = grid_search.best_estimator_
#         test_accuracy = best_model.accuracy(X_test, y_test)
#         print(f"Test Accuracy (custom model): {test_accuracy}")
#         y_pred = best_model.predict(X_test)
#         conf_matrix_xgboost = confusion_matrix(y_test, y_pred)
#         print("Confusion Matrix (XGBoost - custom model):")
#         print(conf_matrix_xgboost)
#         print("Classification Report (custom model):")
#         print(classification_report(y_test, y_pred))

#     else:
#         print(f"Invalid model name: {model_name}")

# def train_inbuilt_model(model, X_train, y_train, X_test, y_test, model_name):
#     print(f"Build started for {model_name} (inbuilt model)...")
#     model.fit(X_train, y_train)
#     print(f"{model_name} (inbuilt model) - Build completed...")

#     if model_name in ["Linear Regression", "Decision Tree", "Random Forest"]:
#         predictions = model.predict(X_test)
#         r2 = r2_score(y_test, predictions)
#         print(f"R-squared (inbuilt model): {r2}")

#     elif model_name in ["KNN", "Naive Bayes", "AdaBoost", "Logistic Regression", "XGBoost"]:
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         conf_matrix = confusion_matrix(y_test, y_pred)
#         print(f"Accuracy (inbuilt model): {accuracy}")
#         print("Confusion Matrix (inbuilt model):")
#         print(conf_matrix)
#         print("Classification Report (inbuilt model):")
#         print(classification_report(y_test, y_pred))

#     else:
#         print(f"Invalid inbuilt model name: {model_name}")

# def main():
#     print("""Choose a dataset
#             (1 - Taxi_price_prediction,
#              2 - Avocado,
#              3 - car_price_predict, 
#              4 - credit_card_spam,
#              5 - email spam classification, 
#              6 - Salary classification,
#              7 - car_price_predict, 
#              8 - Movie prediction,
#              9 - Custom CSV file)""")
#     dataset_choice = int(input("Enter a number(1 - 9) : "))
#     dataset, target_column = load_dataset(dataset_choice)

#     if dataset is not None:
#         if dataset_choice in range(1, 4):
#             task = "regression"
#         elif dataset_choice in range(4, 9):
#             task = "classification"
#         elif dataset_choice == 9:
#             task = input("Enter the task (regression/classification): ")
#         else:
#             print("Invalid dataset choice.")
#             return
        
#         if dataset_choice == 9:
#             preprocessor = preprocess_data(dataset, dataset_choice, target_column)
#         else:
#             preprocessor = preprocess_data(dataset, dataset_choice)

#         X_train, X_test, y_train, y_test = preprocessor

#         model_names = {
#             "regression": ["Linear Regression", "Decision Tree", "Random Forest"],
#             "classification": ["KNN", "Naive Bayes", "AdaBoost", "Logistic Regression", "XGBoost"],
#         }

#         print(f"Choose a {task} model:")
#         for i, model in enumerate(model_names[task], start=1):
#             print(f"{i}. {model}")

#         model_choice = int(input("Choose a model: "))
#         selected_model_name = model_names[task][model_choice - 1]

#         # Train and predict using custom model
#         custom_model = get_model_instance(selected_model_name, task)
#         train_model(custom_model, X_train, y_train, X_test, y_test, selected_model_name)
#         print("Custom Model - Prediction completed...")

#         # Train and predict using inbuilt model
#         inbuilt_model = get_inbuilt_model_instance(selected_model_name, task)
#         train_inbuilt_model(inbuilt_model, X_train, y_train, X_test, y_test, selected_model_name)
#         print("Inbuilt Model - Prediction completed...")

# if __name__ == "__main__":
#     main()


# C:\\Users\\dines\\OneDrive\\Documents\\pack_8\\dd\\dd\\creditcard.csv