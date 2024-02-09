# Machine Learning

## Table of Contents

1) [Decision Tree](https://github.com/RajKumarMcW/Interns_tasks/blob/master/Dhamu%20Pradeep/Readme.md#decision-tree-using-wine-quality-dataset)
2) KNN Regression
3) Logistic Regression
4) Ridge Regression
5) Heart Failure

## Decision Tree using Wine Quality Dataset

1.DATASET AND CLEANING

    i)Shape - (6497, 13)

    ii)Droping Duplicates  

    iii)Checking Null Values

    iv)Filling null values using 
        LinearRegression

2.VISUALIZATION

    Quality is dependent on,
        alcohol
        sulphates
        pH
        free sulpher dioxide
        critric acis
        Model Building
  
3.MODEL

    Decision Tree

4.MODEL BENCHMARKING

    Our model:
        Mean Absolute Error (MAE): 0.6858085656701763
        Mean Squared Error (MSE): 0.8425800088062526
        Mean absolute percentage Error (MSPE): 1.1277804331456676
        Root Mean Squared Error (RMSE): 0.917921570073529
        R-squared: 0.1574199911937474

    For Sklearn model:
        Mean Absolute Error (MAE): 0.6540451749116256
        Mean Squared Error (MSE): 0.7417753605889827
        Mean absolute percentage Error (MSPE): 1.0575792618229556
        Root Mean Squared Error (RMSE): 0.8612638159060108
        R-squared: 0.25822463941101725

## KNN Regression using LONDON BIKE SHARING Dataset

1.DATASET AND VISUALIZATION

    i)shape - (17414, 10)

    ii)NO duplicate and No missing values

2.VISUALIZATION

    cnt is maximum on the 4th day of the week
    The target variable cnt depends on
        hour
        month
        t1, t2
    cnt is maximum on the 4th day of the week

3.MODEL

    KNN Regressor

4.MODEL BENCHMARKING

    Our model:
        Mean Absolute Error (MAE): 0.3922950506911678
        Mean Squared Error (MSE): 0.4248061861316331
        Mean absolute percentage Error (MSPE): 1.4419354770252029
        Root Mean Squared Error (RMSE): 0.651771575117873
        R-squared: 0.575193813868367

    For Sklearn model:
        Mean Absolute Error (MAE): 0.40987694252447576
        Mean Squared Error (MSE): 0.4264285673125478
        Mean absolute percentage Error (MSPE): 1.406702229296142
        Root Mean Squared Error (RMSE): 0.6530149824564118
        R-squared: 0.5735714326874524

## Logistc Regression using FLIGHT SERVICE Dataset

1.DATASET AND CLEANING

    i)shape - (103904, 25)
    
    ii)Filling null values using 
        RandomForestRegressor

2)VISUALIZATION

        When the distance traveled by the flight is high, more people like the travel
        Most of the people traveled in business class
        Eco plus and Eco class are not liked by many peole
        Clealiness is liked by most of the people
        Online booking is not liked by many people
        Dissatisfied or neutral count more than satisfied count
        Male and Female passengers counts are more are less equal
        Satisfied people count is more than neutral or dissatisfied

3.MODEL

    Logistic Regression

4.MODEL BENCHMARK

    Without Sklearn
        Accuracy score of the test data :  0.8267832121171682
        precision score of the test data :  0.7919372294372294
        Recall score of the test data :  0.8469328703703703
        F1 score of the test data :  0.8185123042505593

    For Sklearn model:
        Accuracy score of the test data :  0.8759591646093281
        precision score of the test data :  0.8820505065779525
        Recall score of the test data :  0.8438946759259259
        F1 score of the test data :  0.8625508317929759

## Ridge Regression using HOUSE PRICE PREDICTION

1.DATASET AND CLEANING

    i)Shape - (29050, 9)

    ii)Droping Duplicates  

    iii)NO Missing values

2.VISUALIZATION

    Price of the house depends on ,
        RERA
        BHK_NO.
        SQUARE_FT
        Under Construction
    The map represents most of the houses are from india


 3.MODEL
 
    Ridge Regression

 4.MODEL BENCHMARKING
 
    Our model:
        Mean Absolute Error (MAE): 1.2905405546932795e-05
        Mean Squared Error (MSE): 2.384726993512421e-09
        Mean absolute percentage Error (MSPE): 0.00015286394293989608
        Root Mean Squared Error (RMSE): 4.883366659910375e-05
        R-squared: 0.999999997615273

    For Sklearn model:
        Mean Absolute Error (MAE): 1.0926295998656892e-05
        Mean Squared Error (MSE): 1.9564817196168587e-09
        Mean absolute percentage Error (MSPE): 0.00012039974108123874
        Root Mean Squared Error (RMSE): 4.423213446824445e-05
        R-squared: 0.9999999980435182

