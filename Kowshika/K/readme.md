CLASSIFICATION:

1.KNN:

Dataset: apple_quality.csv

column description:

    A_id: Unique identifier for each fruit

    Size: Size of the fruit

    Weight: Weight of the fruit

    Sweetness: Degree of sweetness of the fruit

    Crunchiness: Texture indicating the crunchiness of the fruit

    Juiciness: Level of juiciness of the fruit

    Ripeness: Stage of ripeness of the fruit

    Acidity: Acidity level of the fruit

    Quality: Overall quality of the fruit

Data Cleaning:

    ->It has no missing values
    ->It has no duplicate rows
    ->MinMaxScaler is used to scale the values in range of 0-1.

Data Visualization:
    Distribution:
        ->It has a balanced class(Quality) 

    Correlation heatmap:
        ->education is correlated with job column
        ->job,default,housing are less correlated with y so we can remove those columns for training

    Pair Plot:
        ->Age is uniformly distributed
        ->Default column is right skewed
        ->If they have loan and married, then they definitly have long term subscription
        ->If they not have loan and divorsed, then they not take subscription

Model Build:
    Decision Tree using sklearn and custom implementation

Model Benchmark:
Sklearn:
Accuracy: 63.24%
Confusion Matrix:
[[1757  803]
 [1075 1474]]
Classification Report:
              precision    recall  f1-score   support

         0.0       0.62      0.69      0.65      2560
         1.0       0.65      0.58      0.61      2549

    accuracy                           0.63      5109
   macro avg       0.63      0.63      0.63      5109
weighted avg       0.63      0.63      0.63      5109

Custom:
Accuracy: 64.18%
Confusion Matrix:
[[1370 1190]
 [640 1909]]
Classification Report:
              precision    recall  f1-score   support

         0.0       0.68      0.54      0.60      2560
         1.0       0.62      0.75      0.68      2549

    accuracy                           0.64      5109
   macro avg       0.65      0.64      0.64      5109
weighted avg       0.65      0.64      0.64      5109 

2.SVMClassifier:
Dataset: mushrooms.csv

column description:

    classes: edible=e, poisonous=p

    cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s

    cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s

    cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y

    bruises: bruises=t,no=f

    odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

    gill-attachment: attached=a,descending=d,free=f,notched=n

    gill-spacing: close=c,crowded=w,distant=d

    gill-size: broad=b,narrow=n

    gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

    stalk-shape: enlarging=e,tapering=t

    stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

    stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s

    stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s

    stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

    stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

    veil-type: partial=p,universal=u

    veil-color: brown=n,orange=o,white=w,yellow=y

    ring-number: none=n,one=o,two=t

    ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z

    spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y

    population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y

    habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

Data Cleaning:

    ->Replacing the '?' in the columns with null.
    ->stalk-root has missing values
    ->Performing imputation on categorical columns in the dataset using mode,RandomForestClassifier, and droping null values.
    ->It has no duplicate rows
    ->MinMaxScaler is used to scale the values in range of 0-1.

Data Visualization:
    Distribution:
        1.The class columns is balanced
        2.Most of the edible mushrooms does not have any odor or have almond and anise odor
        3.Most of the gill spacing is closed

    Correlation heatmap:
        The columns 'cap-shape','cap-surface','cap-color','gill-attachment','stalk-shape','stalk-color-below-ring','veil-color','ring-number','population' and 'habitat' has very less correlation with the target variable 'class' so we can remove those columns for training

    Pair Plot:
        1.When the gill-spacing is more,the population of edible mushrooms are more.
        2.As the gill-size increases,the population of edible mushrooms are also high.
        3.When the odor is almond,alise and none,the mushrooms are more edible

Model Build:
    SVC using sklearn and custom implementation

Model Benchmark:
Sklearn:
Accuracy: 97.09%
Confusion Matrix:
[[1169   12]
 [  59 1198]]
Classification Report:
              precision    recall  f1-score   support

         0.0       0.95      0.99      0.97      1181
         1.0       0.99      0.95      0.97      1257

    accuracy                           0.97      2438
   macro avg       0.97      0.97      0.97      2438
weighted avg       0.97      0.97      0.97      2438

Custom:
Accuracy: 92.33%
Confusion Matrix:
[[1051 130]
 [57 1200]]
Classification Report:
              precision    recall  f1-score   support

         0.0       0.95      0.89      0.92      1181
         1.0       0.90      0.95      0.93      1257

    accuracy                           0.92      2438
   macro avg       0.93      0.92      0.92      2438
weighted avg       0.92      0.92      0.92      2438



3.Decision Tree:

Dataset: bank-additional-full.csv

column description:

    ->Age (numeric)

    ->Job : type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')

    ->Marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown' ; note: 'divorced' means divorced or widowed)

    ->Education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')

    ->Default: has credit in default? (categorical: 'no', 'yes', 'unknown')

    ->Housing: has housing loan? (categorical: 'no', 'yes', 'unknown')

    ->Loan: has personal loan? (categorical: 'no', 'yes', 'unknown')

    ->y - has the client subscribed a term deposit? (binary: 'yes', 'no')

Data Cleaning:

    ->Replacing the unknown values in the columns with null.
    ->job, education, default, housing and loan has missing values
    ->Performing imputation on categorical columns in the dataset using mode,KNN imputation, and droping null values.
    ->It has 30020 duplicate rows and removed it.
    ->MinMaxScaler is used to scale the values in range of 0-1.

Data Visualization:
    Distribution:
        ->Bank has more customer who has job in administrative and blue-collar
        ->illiterate customers are less in the bank
        ->It has a imbalanced class(y) 35000 as no and 5000 as yes, i used smote oversampling technique to balance the class

    Correlation heatmap:
        ->education is correlated with job column
        ->job,default,housing are less correlated with y so we can remove those columns for training

    Pair Plot:
        ->Age is uniformly distributed
        ->Default column is right skewed
        ->If they have loan and married, then they definitly have long term subscription
        ->If they not have loan and divorsed, then they not take subscription

Model Build:
    Decision Tree using sklearn and custom implementation

Model Benchmark:
Sklearn:
Accuracy: 63.24%
Confusion Matrix:
[[1757  803]
 [1075 1474]]
Classification Report:
              precision    recall  f1-score   support

         0.0       0.62      0.69      0.65      2560
         1.0       0.65      0.58      0.61      2549

    accuracy                           0.63      5109
   macro avg       0.63      0.63      0.63      5109
weighted avg       0.63      0.63      0.63      5109

Custom:
Accuracy: 64.18%
Confusion Matrix:
[[1370 1190]
 [640 1909]]
Classification Report:
              precision    recall  f1-score   support

         0.0       0.68      0.54      0.60      2560
         1.0       0.62      0.75      0.68      2549

    accuracy                           0.64      5109
   macro avg       0.65      0.64      0.64      5109
weighted avg       0.65      0.64      0.64      5109



4.logistic regression:

Dataset: Employee.csv

column description:

    Education-->Educational qualification of employees

    JoiningYear-->Year the employee joined in the company

    City-->City the employee lives

    PaymentTier-->Category of payment tier the employee belongs to

    Age-->Age of the employee

    Gender-->Gender identity of the employee

    EverBenched-->Whether an employee has ever been temporarily without assigned work

    ExperienceInCurrentDomain-->Number of years of experience employees have in their current field

    LeaveOrNot-->Whether the employee will leave the company or not

Data Cleaning:

    ->It has no missing values
    ->It has 1889 duplicate rows and removed it.
    ->MinMaxScaler is used to scale the values in range of 0-1.

Data Visualization:
    Distribution:
        ->Mostly Employees stays upto 5 years of experience
        ->If employee has highest payment tier, then they mostly not prefer to leave
        ->It has a imbalanced class(LeaveOrNot) 1700 as no and 1100 as yes, I used smote oversampling technique to balance the class

    Correlation heatmap:
        ->education is correlated with city column
        ->'Education','City','EverBenched','ExperienceInCurrentDomain' are less correlated to leaveOrNot, so we can remove those columns for training

    Pair Plot:
        ->EverBenced column is left skewed
        ->Male under 2 year experience not leave company even they has low payment 
        ->High aged people has high experience in current domain

Model Build:
    LogisticRegression using sklearn and custom implementation

Model Benchmark:
Sklearn:
Accuracy: 62.29%
Confusion Matrix:
[[307 187]
 [192 319]]
Classification Report:
              precision    recall  f1-score   support

         0.0       0.62      0.62      0.62       494
         1.0       0.63      0.62      0.63       511

    accuracy                           0.62      1005
   macro avg       0.62      0.62      0.62      1005
weighted avg       0.62      0.62      0.62      1005

Custom:
Accuracy: 60.00%
Confusion Matrix:
[[309 185]
 [217 294]]
Classification Report:
              precision    recall  f1-score   support

         0.0       0.59      0.63      0.61       494
         1.0       0.61      0.58      0.59       511

    accuracy                           0.60      1005
   macro avg       0.60      0.60      0.60      1005
weighted avg       0.60      0.60      0.60      1005



5.AdaBoost:
Dataset: Telco-Customer-Churn.csv

column description:

    Customers who left within the last month – the column is called Churn

    Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies

    Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges

    Demographic info about customers – gender, age range, and if they have partners and dependents

Data Cleaning:

    ->Converting the 'TotalCharges' column to numeric and drop rows with missing values
    ->TotalCharges has missing values
    ->Converting no phone service and no internet service as no
    ->One-hot encoding specified columns in the dataset
    ->Performing imputation on numerical columns in the dataset using mean,median,RandomForestRegressor, and droping null values.
    ->It has 22 duplicate rows and removed it.
    ->MinMaxScaler is used to scale the values in range of 0-1.

Data Visualization:
    Distribution:
        1.churn is imbalanced has 5000 as no and 2000 as yes
        2.Majority of the senior citizens has less number of churns
        3.The customers with partners are majority in the churn distribution

    Correlation heatmap:
        ->The columns 'gender', 'PhoneService', 'MultipleLines', 'OnlineBackup', 'DeviceProtection','StreamingTV', 'StreamingMovies', and 'PaymentMethod_Mailed check'has very less correlation with churn so we can remove those columns for training

    Pair Plot:
        1.When the TotalCharges increases,the tenure also increases
        2.When the MonthlyCharges is high and the tenure is low,the churn is high
        3.When the MonthlyCharges is high,the TotalCharges is also high

Model Build:
    AdaBoostClassifier using sklearn and custom implementation

Model Benchmark:
Sklearn:
Accuracy: 77.15%
Confusion Matrix:
[[1227  471]
 [ 308 1403]]
Classification Report:
              precision    recall  f1-score   support

         0.0       0.80      0.72      0.76      1698
         1.0       0.75      0.82      0.78      1711

    accuracy                           0.77      3409
   macro avg       0.77      0.77      0.77      3409
weighted avg       0.77      0.77      0.77      3409

Custom:
Accuracy: 75.33%
Confusion Matrix:
[[1231 467]
 [374 1337]]
Classification Report:
              precision    recall  f1-score   support

         0.0       0.77      0.72      0.75      1698
         1.0       0.74      0.78      0.76      1711

    accuracy                           0.75      3409
   macro avg       0.75      0.75      0.75      3409
weighted avg       0.75      0.75      0.75      3409


REGRESSION:

1.Linear Regression:

Dataset:OLX_cars_dataset00.csv

Column description:

    Ad Id
    Car Name
    Make
    Model
    Year
    KM's driven
    Price
    Fuel
    Registration city
    Car documents
    Assembly
    Transmission
    Condition
    Seller Location
    Description
    Car Features
    Images URL's
    Car Profile

Data Cleaning:

    ->Only the column 'Car Profile' has missing values
    ->The dataset has 201 duplicate rows
    ->Dropped the columns 'Car Profile','Condition','Car Name','Ad ID','Registration city','Car documents','Seller Location','Description','Car Features' and 'Images URL's' since they have very less correlation with the target variables
    ->StandardScaler is used to scale the values keeping the mean as 0 and standard deviation as 1

Data Visualization:

    Correlation heatmap:
        ->The columns 'Make','Model','Year','KM's driven','Fuel','Assembly','Transmission' and 'Car Features' are highly correlated with the target variable
    
    Pair Plot:
        1.As the year moves on,the price also increases
        2.When the KM's driven is low,the price is high

Model Build:

    Linear Regression using sklearn and custom implementation

Model Benchmark:
Sklearn:
Mean Absolute Error (MAE): 546620.27
Mean Squared Error (MSE): 468887687187.75
Root Mean Squared Error (RMSE): 684753.74
R-squared (R2) Score: 0.65

Custom:
Mean Absolute Error (MAE): 546618.74
Mean Squared Error (MSE): 468885524891.34
Root Mean Squared Error (RMSE): 684752.16
R-squared (R2) Score: 0.65

2.SVR:

Dataset: USA_Housing.csv

column description:

    Avg. Area Income
    Avg. Area House Age 
    Avg. Area Number of Rooms
    Avg. Area Number of Bedrooms
    Area Population
    Price 
    Address                       

Data Cleaning:

    ->It has no missing values
    ->It has no duplicate rows
    ->droped Address columns since it is not needed
    ->MinMaxScaler is used to scale the values in range of 0-1.

Data Visualization:

    Correlation heatmap:
        ->All the columns except 'Avg. Area Number of Bedrooms' in the dataset are highly correlated with the target variable

    Pair Plot:
        1.All the columns are uniformly distributed 
        2.Avg area number of bedroom is dependent on Avg area number of room

Model Build:
    SVR using sklearn and custom implementation

Model Benchmark:
Sklearn:
Mean Absolute Error (MAE): 0.03
Mean Squared Error (MSE): 0.00
Root Mean Squared Error (RMSE): 0.04
R-squared (R2) Score: 0.91

Custom:
Mean Absolute Error (MAE): 0.03
Mean Squared Error (MSE): 0.00
Root Mean Squared Error (RMSE): 0.04
R-squared (R2) Score: 0.91

3.Ridge Regression:

Dataset: Video_Games_Sales_as_at_22_Dec_2016.csv

column description:
    
    Name
    Platform
    Year_of_Release
    Genre
    Publisher
    NA_Sales
    EU_Sales              
    JP_Sales
    Other_Sales
    Global_Sales
    Critic_Score
    Critic_Count
    User_Score
    User_Count
    Developer
    Rating

Data Cleaning:

    ->The columns 'Year_of_Release','Critic_Score','Critic_Count' and 'User_Count' has missing values
    ->It has no duplicate rows
    ->Dropped the columns 'Platform','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales','Publisher','Critic_Score' and 'User_Count' since they have very less correlation with ther target variable
    ->StandardScaler is used to scale the values with mean as 0 and standard deviation as 1.

Data Visualization:

    Correlation heatmap:
        ->The columns 'Name','Year_of_Release','Genre','User_Score' and 'Developer' are highly correlated with the target variable

    Pair Plot:
        1.Some of the columns are uniformly distributed 

Model Build:
    Ridge Regression using sklearn and custom implementation

Model Benchmark:
Sklearn:
Mean Absolute Error (MAE): 1.06
Mean Squared Error (MSE): 1.85
Root Mean Squared Error (RMSE): 1.36
R-squared (R2) Score:0.65

Custom:
Mean Absolute Error (MAE): 1.06
Mean Squared Error (MSE): 1.85
Root Mean Squared Error (RMSE): 1.36
R-squared (R2) Score: 0.65


Package API:
To import package:
from rk import knn,dt,logR,AB,LR,svc,svr,rr,preprocess

preprocess.preprocess(df,dst_no)
preprocess the df dataframe based on dst_no


model=knn.K_Nearest_Neighbors_Classifier(K)
Intialize the knn classification model and K is number of neighbours

model.cfit(X_train, Y_train)
Train model with X_train as features and Y_train as target

model.cpredict(X_test)
returns predicted target value for X_test features


model=LogR.Logistic_Regression(learning_rate, n_iterations)
Intialize the logistic regression classification model

model.cfit(X, y)
Train model with X as features and y as target

model.cpredict(X)
returns predicted target value for X features


model=svc.Support_Vector_Classification(C)
Intialize the svc classification model and C is the complexity

model.cfit(X, Y, batch_size, learning_rate, epochs)
Train model with X as features and Y as target

model.cpredict(X)
returns predicted target value for X features


model=dt.decision_Tree()
Intialize the decision tree classification model

model.cfit(X, y)
Train model with X as features and y as target

model.cpredict(X_test)
returns predicted target value for X_test features


model=AB.Adaptive_Boost(n_clf)
Intialize the adaptive boost classification model and n_clf is the number of classification performed

model.cfit(X, y)
Train model with X as features and y as target

model.cpredict(X)
returns predicted target value for X features


model=LR.Linear_Regression(learning_rate, epsilon)
Intialize the linear regression model and epsilon is the error term that can occur

model.cfit(x, y)
Train model with x as features and y as target

model.cpredict(x)
returns predicted target value for x features


model=rr.Ridge_Regression(learning_rate,alpha=0.1)
Intialize the ridge regression model and alpha is the amount of emphasis given to minimizing RSS vs minimizing the sum of squares of coefficients

model.cfit(X_train,y_train)
Train model with X_train as features and y_train as target

model.cpredict(X_test)
returns predicted target value for X_test features


model=svr.Support_Vector_Regressor()
Intialize the SVR regression model 

model.cfit(trainX, trainZ)
Train model with trainX as features and trainZ as target

model.cpredict(testX)
returns predicted target value for testX features