
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dj import logistic,decisiontree,ridgereg,svm,nb,xgb,knnreg,knncla
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix,precision_score,recall_score,f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import math
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.svm import SVC

def evalution_matrices(y_pred,y_test):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred,  average='weighted')
        recall = recall_score(y_test, y_pred,  average='weighted')
        f1 = f1_score(y_test, y_pred ,  average='weighted')
        print('Accuracy score of the test data : ', accuracy)
        print('precision score of the test data : ', precision)
        print('Recall score of the test data : ', recall)
        print('F1 score of the test data : ', f1)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test,y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

def evalution_matrices_reg(y_pred,y_test):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test,y_pred)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean absolute percentage Error (MSPE): {mape}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R-squared: {r2}")


def predict_logistic(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn):

    if sklearn == True:
        print("\n\nAccuracy using sklearn model \n\n")
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
            classifier = logistic.Logistic_Regression(learning_rate=learning_rate, no_of_iterations=num_iterations)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

    evalution_matrices(y_pred,y_test)

def predict_knn_cla(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn):
    if sklearn == True:
        knn = KNeighborsClassifier(n_neighbors=1)
        # Train the model on the training data
        knn.fit(X_train, y_train)
        # Make predictions on the test set
        X_test = np.array(X_test)
        y_pred = knn.predict(X_test)
    else:
        knn = knncla.KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
    
    evalution_matrices(y_pred,y_test)
        


def predict_knn_reg(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    y_train = sc.fit_transform(y_train.values.reshape(-1, 1))
    y_test = sc.fit_transform(y_test.values.reshape(-1, 1))

    if sklearn == True:
        knn_regressor = KNeighborsRegressor(n_neighbors=3)

        # Train the model
        knn_regressor.fit(X_train, y_train)
        y_pred = knn_regressor.predict(X_test)

    else:
        model = knnreg.KNNRegressor(K=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    evalution_matrices_reg(y_pred,y_test)



def predict_decisiontree(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn):
    print("Decision Tree")
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    y_train = sc.fit_transform(y_train.values.reshape(-1, 1))
    y_test = sc.fit_transform(y_test.values.reshape(-1, 1))

    print("After normalization ... ")
    print(f"X_train : {X_train}")
    print(f"Y_train : {y_train}")
    print(f"X_test : {X_train}")
    print(f"Y_test : {y_train}")

    if sklearn == True:
        print("\n\nAccuracy using sklearn model \n\n")
        model = DecisionTreeRegressor(max_depth=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
            regressor = decisiontree.DecisionTreeRegressor(min_samples_split=3, max_depth=5)
            regressor.fit(X_train,y_train)
            y_pred = regressor.predict(X_test)
    
    evalution_matrices_reg(y_pred,y_test)



def predict_ridge_Reg(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn):

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    y_train = sc.fit_transform(y_train.values.reshape(-1, 1))
    y_test = sc.fit_transform(y_test.values.reshape(-1, 1))
    print("After normalization ... ")
    print(f"X_train : {X_train}")
    print(f"Y_train : {y_train}")
    print(f"X_test : {X_train}")
    print(f"Y_test : {y_train}")
    
    if sklearn == True:
        print("\n\nAccuracy using sklearn model \n\n")
        reg = Ridge(alpha=1)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
    else:
        ridge_model = ridgereg.RidgeRegression(learning_rate=learning_rate, iterations=num_iterations, lambda_param=1)
        ridge_model.fit(X_train, y_train)
        # Make predictions
        y_pred = ridge_model.predict(X_test)
    evalution_matrices_reg(y_pred,y_test)




def predict_svm(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn):

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    y_train = pd.to_numeric(y_train, errors='coerce')
    y_test = pd.to_numeric(y_test, errors='coerce')

    print("After normalization ... ")
    print(f"X_train : {X_train}")
    print(f"Y_train : {y_train}")
    print(f"X_test : {X_train}")
    print(f"Y_test : {y_train}")

    if sklearn == True:
        clf = SVC(kernel='linear') # Linear Kernel
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    else:
        classifier = svm.SVM_classifier(learning_rate=learning_rate, no_of_iterations=num_iterations, lambda_parameter = 0.01)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    
    evalution_matrices(y_pred,y_test)


def predict_naive(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn):

    if sklearn == True:
        nb_classifier = MultinomialNB()
        nb_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = nb_classifier.predict(X_test)

        # Evaluate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        
        F1_score = f1_score(y_test, predictions, average='weighted')

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {F1_score}")
    else:
            nb_classifier = nb.NaiveBayesClassifier()
            nb_classifier.fit(X_train, y_train)
            predictions = nb_classifier.predict(X_test)

            # Evaluate metrics
            accuracy, avg_precision, avg_recall, avg_f1 = nb_classifier.evaluate_metrics(y_test, predictions)

            print("Accuracy:", accuracy)
            print("Precision:", avg_precision)
            print("Recall:", avg_recall)
            print("F1 Score:", avg_f1)




def predict_XGB(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn):

    if sklearn == True:
        # Calculate the number of classes
            num_classes = len(np.unique(y_train))

            # Calculate class weights
            class_weights = len(y_train) / (num_classes * np.bincount(y_train))
            # print(class_weights)
            # Create and train the XGBoost classifier with class weights
            xgb_classifier = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, objective='multi:softprob', num_class=num_classes)
            xgb_classifier.fit(X_train, y_train)
            # Make predictions
            predictions = xgb_classifier.predict(X_test)
            print(len(predictions.shape))
            if len(predictions.shape) > 1:
                predictions = np.argmax(predictions, axis=1)
            #Evaluate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
                
            F1_score = f1_score(y_test, predictions, average='weighted')
                
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {F1_score}")
    else:
            num_classes = len(np.unique(y_train))
            class_weights = len(y_train) / (num_classes * np.bincount(y_train))

            xgb_classifier = xgb.XGBoostClassifier(learning_rate=learning_rate, num_iterations=num_iterations, max_depth=3, num_classes=4)
            xgb_classifier.fit(X_train, y_train, sample_weights=class_weights)
            predictions = xgb_classifier.predict(X_test)
            # Evaluate metrics
            print(predictions)
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
                
            F1_score = f1_score(y_test, predictions, average='weighted')

            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {F1_score}")


def predict(model_name,X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn):
    if model_name == "Decision Tree":
        return predict_decisiontree(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn)
    elif model_name == "NB":
        return predict_naive(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn)
    elif model_name == "Logistic Regression":
        return predict_logistic(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn)
    elif model_name == "Ridge Regression":
        return predict_ridge_Reg(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn)
    elif model_name == "SVM":
        return predict_svm(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn)
    elif model_name == "XGB":
        return predict_XGB(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn)
    elif model_name == "knnreg":
        return predict_knn_reg(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn)
    elif model_name == "knncla":
        return predict_knn_cla(X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn)