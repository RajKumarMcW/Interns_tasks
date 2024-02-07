from rk import knn,dt,logR,AB,LR,svc,svr,rr,preprocess,metric
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
    assert dst_no in {'1', '2', '3', '4', '5', '6', '7', '8'}, "Invalid Input"

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

    df=pd.read_csv(file_path)
    print("\n",df.head(),"\n")
    print(df.info(),"\n")

    
    X,y=preprocess.preprocess(df,dst_no)
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
    assert model_no in {'1', '2', '3', '4', '5', '6'}, "Invalid Input"
    if model_no=='1':
        custom_model=knn.K_Nearest_Neighbors_Classifier()
        sklearn_model=KNeighborsClassifier(n_neighbors = 3)
    elif model_no=='2':
        custom_model=logR.Logistic_Regression()
        sklearn_model=LogisticRegression()
    elif model_no=='3':
        custom_model=svc.Support_Vector_Classification()
        sklearn_model=SVC(kernel='linear')
    elif model_no=='4':
        custom_model=dt.decision_Tree()
        sklearn_model=DecisionTreeClassifier(min_samples_split = 100,max_depth = 100)
    elif model_no=='5':
        custom_model=AB.Adaptive_Boost()
        sklearn_model=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1))
    elif model_no=='6':
        exit()

    return custom_model,sklearn_model

def get_regression_model():
    model_no=input("""Regression: 
                     1.Linear Regression           
                     2.Support Vector Regression
                     3.Ridge Regression
                     4.Exit
                     Enter a number:
                     """)
    assert model_no in {'1', '2', '3', '4'}, "Invalid Input"

    if model_no=='1':
        custom_model=LR.Linear_Regression()
        sklearn_model=linear_model.LinearRegression()
    elif model_no=='2':
        custom_model=svr.Support_Vector_Regressor()
        sklearn_model=SVR(kernel='linear')
    elif model_no=='3':
        custom_model=rr.Ridge_Regression()
        sklearn_model=linear_model.Ridge(alpha=0.1)
    elif model_no=='4':  
        exit()
    
    return custom_model,sklearn_model

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





def main():
    X,y,dst_no=get_dataset()
    if dst_no==0:
        return  
    while(1):
        X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.3, random_state=42)
        print("\nTraining data size:",np.shape(X_train),"\nTesting data size:",np.shape( X_test))
        if(int(dst_no)<=5):
            custom_model,sklearn_model=get_classification_model()
        else:
            custom_model,sklearn_model=get_regression_model()

        # print(X_train,X_test )
        sklearn_model.fit(X_train,y_train)
        y_pred=sklearn_model.predict(X_test)

        custom_model.cfit(X_train,y_train)
        custom_y_pred=custom_model.cpredict(X_test)
        

        if(int(dst_no)<=5):
            print(f"\nCustom Model for {custom_model}:")
            metric.classification_metrics(y_test, custom_y_pred)
            print(f"\nSklearn Model for {sklearn_model}:")
            metric.classification_metrics(y_test, y_pred)
        else:
            print(f"\nCustom Model for {custom_model}:")
            metric.regression_metric(y_test, custom_y_pred)
            print(f"\nSklearn Model for {sklearn_model}:")
            metric.regression_metric(y_test, y_pred)
    
    return 


if __name__=='__main__':
    main()