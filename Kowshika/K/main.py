from k import Knearestneighbors,Decisiontree,Logisticregression,Adaboost,Linearregression,SupportVectorMachineclassification,SupportVectorMachineregression,Ridgeregression,preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,SVR
    
import pandas as pd
import numpy as np

def get_dataset():
    dst_no=input("""Datasets:
                     Classification:
                     1.apple_quality.csv
                     2.Employee.csv
                     3.bank-additional-full.csv
                     4.Telco-Customer-Churn.csv
                     5.mushrooms.csv

                     Regression:
                     6.OLX_cars.csv
                     7.USA_Housing.csv
                     8.Video_Games_Sales.csv
                     Enter a number:
                     """)
    if dst_no=='1':
        file_path='datasets\\apple_quality.csv'
    elif dst_no=='2':
        file_path='datasets\\Employee.csv'
    elif dst_no=='3':
        file_path='datasets\\bank-additional-full.csv'
    elif dst_no=='4':
        file_path='datasets\\Telco-Customer-Churn.csv'
    elif dst_no=='5':
        file_path='datasets\\mushrooms.csv'
    elif dst_no=='6':
        file_path='datasets\\OLX_cars_dataset00.csv'
    elif dst_no=='7':
        file_path='datasets\\USA_Housing.csv'
    elif dst_no=='8':
        file_path='datasets\\Video_Games_Sales_as_at_22_Dec_2016.csv'
    else:
        print("Invalid Input")
        return 0,0,0

    df=pd.read_csv(file_path)
    print("\n",df.head(),"\n")
    print(df.info(),"\n")

    
    X,y=preprocessing.preprocess(df,dst_no)
    return X,y,dst_no


def get_classification_model():
    model_no=input("""Classification:
                     1.K-Nearest Neighbour
                     2.Logistic Regression 
                     3.Support Vector Machine 
                     4.Decision Tree 
                     5.AdaBoost 
                     6.Exit
                     Enter a number:
                     """)
    if model_no=='1':
        model=Knearestneighbors.K_Nearest_Neighbors_Classifier()
        Bmodel=KNeighborsClassifier(n_neighbors = 5)
    elif model_no=='2':
        model=Logisticregression.Logistic_Regression()
        Bmodel=LogisticRegression()
    elif model_no=='3':
        model=SupportVectorMachineclassification.Support_Vector_Classification()
        Bmodel=SVC(kernel='linear')
    elif model_no=='4':
        model=Decisiontree.decision_Tree()
        Bmodel=DecisionTreeClassifier()
    elif model_no=='5':
        model=Adaboost.Adaptive_Boost()
        Bmodel=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1))
    elif model_no=='6':
        exit()
    else:
        print("\nINVALID INPUT!!\n")
        model,Bmodel=get_classification_model()
    return model,Bmodel

def get_regression_model():
    model_no=input("""Regression: 
                     1.Linear Regression           
                     2.Support Vector Regression
                     3.Ridge Regression
                     4.Exit
                     Enter a number:
                     """)
    if model_no=='1':
        model=Linearregression.Linear_Regression()
        Bmodel=linear_model.LinearRegression()
    elif model_no=='2':
        model=SupportVectorMachineregression.Support_Vector_Regressor()
        Bmodel=SVR(kernel='linear')
    elif model_no=='3':
        model=Ridgeregression.Ridge_Regression()
        Bmodel=linear_model.Ridge(alpha=0.1)
    elif model_no=='4':  
        exit()
    else:
        print("\nINVALID INPUT!!\n")
        model,Bmodel=get_regression_model()
    return model,Bmodel

def custom_train_test_split(X, Y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = len(X)

    num_test_samples = int(test_size * num_samples)

    test_indices = np.random.choice(num_samples, size=num_test_samples, replace=False)

    mask_train = np.ones(num_samples, dtype=bool)
    mask_train[test_indices] = False
    X_train, X_test = X[mask_train], X[~mask_train]
    Y_train, Y_test = Y[mask_train], Y[~mask_train]

    return X_train, X_test, Y_train, Y_test



def cmetric(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def rmetric(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2) Score: {r2:.2f}")

def main():
    X,y,dst_no=get_dataset()
    if dst_no==0:
        return  

    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.3, random_state=42)
    print("\nTraining data size:",np.shape(X_train),"\nTesting data size:",np.shape( X_test))
    if(int(dst_no)<=5):
        model,Bmodel=get_classification_model()
    else:
        model,Bmodel=get_regression_model()

    # print(X_train,X_test )
    Bmodel.fit(X_train,y_train)
    B_y_pred=Bmodel.predict(X_test)

    model.cfit(X_train,y_train)
    y_pred=model.cpredict(X_test)
    

    if(int(dst_no)<=5):
        print(f"\nCustom Model for {model}:")
        cmetric(y_test, y_pred)
        print(f"\nSklearn Model for {Bmodel}:")
        cmetric(y_test, B_y_pred)
    else:
        print(f"\nCustom Model for {model}:")
        rmetric(y_test, y_pred)
        print(f"\nSklearn Model for {Bmodel}:")
        rmetric(y_test, B_y_pred)
    
    return 


if __name__=='__main__':
    main()