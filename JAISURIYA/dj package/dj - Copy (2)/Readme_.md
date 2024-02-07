
----------------------------------------------------------CLASSIFICATION------------------------------------------------------ 


1.INCOME DATASET
----------------
1.DATASET and CLEANING

    i)shape 43957, 15

    ii)columns 
        'age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income_>50K'

    iii)Missing values
        workclass
        occupation
        native-country

    iV)remove missing values using mode

    v)dataset is imbalanced 
        using oversampling balance the dataset

    vi)For normalize
        Min-Max Normalize

2.VISUALIZATION 
    People maximunm working in private sector
    From self emp-Income most of the people earn more than 50k
    Most of the people educated Hs-grad or some college
    If people done Doctorate or Masters maximum of people earn more than 50k


3.MODEL
    KNN

4)MODEL BENCHMARKING
    Using sklearn
        Accuracy: 0.9030
        Precision: 0.8663
        Recall: 0.9525
        AUC-ROC Score: 0.9031
        Confusion Matrix:
          [[8587 1470]
           [ 475 9531]]
        AUC-ROC Score: 0.9031
    Without using sklearn
        Accuracy: 0.9031
        Precision: 0.907
        Recall: 0.90315
        F1-score: 0.90285


--------------------------------------------------------------------------------------------------------------------------------------

2.TRAFFIC DATASET
 ------------------
1.DATASET AND CLEANING

    i)shape 
        5952, 9

    ii)columns 
        Time,Date, Day of the week,CarCount,BikeCount,BusCount,TruckCount,Total,Traffic Situation

    iii)Missing values
        CarCount,BikeCount,BusCount,TruckCount

    iv)Remove missing values using,
        Linear Regression for 'CarCount' and 'BikeCount'
        Random Forest Regression for 'BusCount' and 'TruckCount'

    v)Dataset is imbalanced
        using class weight balance the dataset

    vi)For normalize
        Min-Max Normalize

2.VISUALIZATION

    Number of Truck(TruckCount) has the most contribution to Traffic
    Thursday and Wednesday are the most busy days for traffic
    Peak hours of traffic are between 6:00am-8:00am and 4:00pm-6:00pm
    Normal traffic situation counts the most
    Friday sees the minimum Traffic
Feature selection
    using correlation heatmap drop the Total vehecles coloumn

3.MODEL
    XGBOOST


4.MODEL BENCHMARKING
    Using sklearn
        Accuracy: 0.9756302521008403
        AUC-ROC Score: 0.9974870036284005
        Precision: 0.9755307486138463
        Recall: 0.9756302521008403
        F1 Score: 0.9754317192164059
    Without using sklearn
        Accuracy: 0.9798488664987406
        Precision: 0.9786790780141844
        Recall: 0.9616924233832641
        F1 Score: 0.9694994734994734


---------------------------------------------------------------------------------------------------------------------------

3.COMMENT EMOTION
-----------------
1.DATASET AND CLEANING
    i)shape 
        5937, 2

    ii)columns 
        Comment,Emotion

    iii)Missing values
        No Missing values

    iv)dataset is balanced


2.PREPROCESSING
    Remove Puntuation
    Remove stopwords,REmove white spaces
    Tokenization
    Normalizing text(stemming,lemmatization)
    Vectorization
    lable Encoder


3.MODEL
    Multinomial NaiveBayes

4.MODEL BENCHMARKING
    Using sklearn
        Accuracy: 0.9031986531986532
        AUC-ROC Score: 0.9767583679110395
        Precision: 0.9036858149024937
        Recall: 0.9031986531986532
        F1 Score: 0.9033014240181332
    Without using sklearn
        Accuracy: 0.9015151515151515
        Precision: 0.9013421363942585
        Recall: 0.9014360474510551
        F1 Score: 0.901248254974282

------------------------------------------------------------------------------------------------------------------------------------------------

4.WATER QUALITY PREDICTION
-----------------------------

1.DATASET AND CLEANING

    i)Shape - (7999, 21)

    ii)Droping Duplicates  
    
    iii)Checking Null Values

    iv) Filling null values using mode, mean, median, std

2.VISUALIZATION
    If silver value is greater than 0.2 then mostly water qulity is unsafe.
    Virus values mostly ranges above 0.4

3.MODEL
    SVM

4.MODEL BENCHMARKING
    Our Implementaion 
        Accuracy:  0.7897707231040564
        precision :  0.8014760147601476
        Recall :  0.7685774946921444
        F1 score  :  0.7846820809248556

    Sklearn model:
        Accuracy :  0.7922398589065256
        precision :  0.7998544395924309
        Recall  :  0.7777777777777778
        F1 score:  0.7886616433440976

------------------------------------------------------------------------------------------------------------------------------------

5.FLIGHT SERVICE

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



---------------------------------------------------------REGRESSION----------------------------------------------------


1.HOUSE PRICE PREDICTION
-----------------------
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

-----------------------------------------------------------------------------------------------------------------------------------------

2.WINE QUALITY PREDICTION

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


-----------------------------------------------------------------------------------------------------------------------------------------

3.LONDON BIKE SHARING

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

---------------------------------------------------------------------------------------------------------------


5.Building PYPI Package


---------------------------------------REGRESSION------------------------------

REGRESSION MODELS

1.RIDGE REGRESSION   
2.KNN REGRESSION
3.DECISION TREE

REGRESSION DATASETS

1.WINE QUALITY
2.HOUSE PRICE
3.BIKE SHARE

--------------------------------------CLASSIFICATION-----------------------------

CLASSIFICATION MODELS

1.LOGISTIC REGRESSION
2.XGBOOST
3.KNN CLASSIFICATION
4.SVM
5.NAIBE BAYES

CLASSIFICATION DATASETS

1.FLIGHT
2.WATER QUALITY
3.TRAFFIC
4.INCOME
5.EMOTION


DATASET NAME AND RESPECTIVE FILE PATHS

    "flight": "./datasets/flight.csv"
    "wine_quality": "./datasets/winequality.csv"
    "house_price": "./datasets/houseprice.csv"
    "bike_share": "./datasets/london_merged.csv"
    "water_quality": "./datasets/waterquality.csv"
    "emotion_classify": "./datasets/Emotion_classify_Data.csv"
    "traffic": "./datasets/traffic.csv"
    "income":"./datasets/income.csv" 


-------------
Main.py file |
-------------


dataset = {
    "flight": "./datasets/flight.csv",  #Logistic Regression
    "wine_quality": "./datasets/winequality.csv", #Decision Tree
    "house_price": "./datasets/houseprice.csv", #Ridge Regression
    "bike_share": "./datasets/london_merged.csv", #knnreg
    "water_quality": "./datasets/waterquality.csv", #SVM
    "emotion_classify": "./datasets/Emotion_classify_Data.csv", #NB
    "traffic": "./datasets/traffic.csv", #XGB
    "income":"./datasets/income.csv"#knncla
}

model_name = "XGB"
learning_rate = 0.002
num_iterations = 1000
sklearn=True
target_column = ""

def main():
    X_train,X_test,y_train,y_test = preprocess.prepreocess(dataset["income"],target_column)
    predict.predict(model_name,X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn)

if __name__ == "__main__":
    main()



this is the main file 
explanation below

In this if you want XGB give model name above as it is ,you want another model means change model_name ,
    1.LOGISTIC REGRESSION  ->Logistic Regression
    2.XGBOOST              ->XGB
    3.KNN CLASSIFICATION   ->knncla
    4.SVM                  ->SVM
    5.NAIBE BAYES          ->NB


    1.RIDGE REGRESSION     ->Ridge REgression
    2.KNN REGRESSION       ->knnreg
    3.DECISION TREE        ->Decision Tree

If you want change learing rate num_iteration change main.py itself

If you want to see sklearn accuracy also give sklearn=True(above mentioned)

preprocessing step

    X_train,X_test,y_train,y_test = preprocess.prepreocess(dataset["income"],target_column)
        DATASET NAME AND RESPECTIVE FILE PATHS in this heading(above mentioned) you can give dataset name

    After preprocesing that function will return X_train,X_test,y_train,y_test

Prediction
    
    predict.predict(model_name,X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn)

