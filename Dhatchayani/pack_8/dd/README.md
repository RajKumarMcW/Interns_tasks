# ML Models
---

## Table of contents 
---
> ### Regression
---
>> #### Avacado price prediction----------------[click here](https://github.com/DhatchayaniL/DD/blob/main/README.md#avocado-price-prediction---decision-tree-regressor)

>> #### Car price prediction----------------------[click here](https://github.com/DhatchayaniL/DD/blob/main/README.md#car-price-prediction---random-forest-regressor)

>> #### Taxi price prediction---------------------[click here](https://github.com/DhatchayaniL/DD/blob/main/README.md#taxi-price-prediction---linear-regression)
---
> ### Classification
---
>> #### Credit card payment classification------[click here](https://github.com/DhatchayaniL/DD/blob/main/README.md#credit-card-payment-classification---knn)

>> #### Rain in australia--------------------------[click here](https://github.com/DhatchayaniL/DD/blob/main/README.md#rain-in-australia---logistic-regression)

>> #### Email spam classification-----------------[click here](https://github.com/DhatchayaniL/DD/blob/main/README.md#email-spam-classification---multinomial-naive-bayes)

>> #### Salary prediction-------------------------[click here](https://github.com/DhatchayaniL/DD/blob/main/README.md#salary-prediction---ada-boost-classifier)

>> #### Movie hits prediction--------------------[click here](https://github.com/DhatchayaniL/DD/blob/main/README.md#movie-hits-prediction---xg-boost-classifier)

> #### API calling-----------------------------------[click here](https://github.com/DhatchayaniL/DD/blob/main/README.md#api-documentation-for-your-package)

## Regression:
1. ### Avocado price prediction - Decision tree regressor
    1. *Data cleaning*
        * No null values
        * No duplicates
    2. *Data visualization*
        * Each and every columns are correlated
          
          ![pairplot](https://github.com/DhatchayaniL/DD/assets/88893048/085e775a-cb75-489e-9522-0fb8162823eb)
          
        * There is a positive relation between 'total volume' and all the other columns.
        * All the other columns are also positively correlated with all the columns except the target column.
        * Total volume can go above 3 if the average price is in between 0.5 to 2
        * Average price of 1.0 to 1.6 are high in frequency
    3. *Model Building*
        * Decision tree regressor
    4. *Model Benckmarking*
        * sklearn accuracy - 72.1%
        * Custom code accuracy - 70.3%
      
        | Linear Regression | Decision tree Regressor | Random forest regressor |
        |-------------------|-------------------------|-------------------------|
        |                   |                         |                         |
        | 70.3 %            | 71.9 %                  | 73.4 %                  | 

2. ### Car price prediction - Random forest regressor
    1. *Data cleaning*
        * Had null values
        * Numerical columns - Imputed using Iterative Imputer
        * Categorical columns - Imputed using Mode
        * Dropped all duplicates
    2. *Data visualization*

          ![pairplot](https://github.com/DhatchayaniL/DD/assets/88893048/ead2e554-d7bb-4777-b06d-aadbf01037f3)
       
        * Selling price is maximum in 50,000 to 3Lakhs
        * Fuel efficiency is maximum at 20 km/L
        * Dropped LPG and CNG from fuel type as it is very less.
    4. *Model Building*
        * Random forest regressor
    5. *Model Benckmarking*
        * sklearn accuracy - 81.4%
        * Custom code accuracy - 80.4%

          | Linear Regression | Decision tree Regressor | Random forest regressor |
          |-------------------|-------------------------|-------------------------|
          |                   |                         |                         |
          | 84.4 %            | 78.9 %                  | 81.5 %                  |

3. ### Taxi price prediction - Linear regression
    1. *Data cleaning*
        * No null values
        * Dropped all duplicates
    2. *Data visualization*
   ![pairplot](https://github.com/DhatchayaniL/DD/assets/88893048/6bca7e3b-4574-4e81-9d9c-e31fd174d461)
        * Dropped distance_traveled and num_of_passengers as they are less correlated with other columns.
        * Total fare and fare are highly correlated
        * miscellaneous fees and surge_applied are highly correlated in correlation matrix.
   ![taxi_cor](https://github.com/DhatchayaniL/DD/assets/88893048/0df62a66-1ebc-4548-8bcd-ff9bf7a27ab6) 
           
    4. *Model Building*
        * Linear regression
    5. *Model Benckmarking*
        * sklearn accuracy - 99%
        * Custom code accuracy - 99%
          

    ![lin](https://github.com/DhatchayaniL/DD/assets/88893048/481afef5-3b10-42f8-abe3-b2eb3b75b9b4)

| Linear Regression | Decision tree Regressor | Random forest regressor |
|-------------------|-------------------------|-------------------------|
|                   |                         |                         |
| 100 %             | 99.6 %                  | 96.4 %                  | 

## Classification

4. ### Credit card payment classification - KNN
    1. *Data cleaning*
        * No null values
        * Dropped all duplicates
    2. *Data visualization*
        * There is no relation between v1 to v28 columns
   ![correlation](https://github.com/DhatchayaniL/DD/assets/88893048/aebe3e2a-9f7c-4e5c-9ef4-9860fb504ea4)
        * Did principal component analysis and reduces its dimentionality which increases correlation between v1 to v28 columns
   ![correlation after PCA](https://github.com/DhatchayaniL/DD/assets/88893048/b4c524d4-28f3-4bdc-a757-48f90689f93c)
        * Transaction amount is higher in 0 to 2000
        * Used robust scalar for normalization
   ![distribution plot](https://github.com/DhatchayaniL/DD/assets/88893048/4c4eeb04-4641-49c5-8724-09c8c1e58df9)
    3. *Model Building*
        * K-Nearest Neightbour
    4. *Model Benckmarking*
        * sklearn accuracy - 94.3%
        * Custom code accuracy - 94.3%
    ![confusion matrix heatmap](https://github.com/DhatchayaniL/DD/assets/88893048/41acd77b-13a7-4823-9291-d21a0949f51a)

    | KNN    | Multinomial Naive Bayes | Adaboost | Logistic Regression | XG boost |
    |--------|-------------------------|----------|---------------------|----------|
    |        |                         |          |                     |          |
    | 91.7 % | 49.7 %                  | 49.7 %   | 92.2 %              | 91.7     | 

5. ### Rain in australia - Logistic regression
    1. *Data cleaning*
        * Had null values
        * Used KNN - imputer to fill null values
        * Dropped all duplicates
    2. *Data visualization*
        * Imbalanced dataset
        
        <p align="center">
          <img src="https://github.com/DhatchayaniL/DD/assets/88893048/9432afb3-3bd3-41e5-ad23-e497e47ac088" alt="Rain Tomorrow vs. Rain Today">
        </p>
  
        * Did over sampling
          
       <p align="center">
          <img src="https://github.com/DhatchayaniL/DD/assets/88893048/9f8a2c30-0459-44b7-b483-15b9a3d03a77" alt="Visualization 2">
       </p>   

        * Min Temp and Max Temp are positively correlated 
        * Temp3pm and Temp9am are positively correlated
      
        ![corr](https://github.com/DhatchayaniL/DD/assets/88893048/611d4cbe-b1ec-4257-9721-fdb67a6ddb68)     



    4. *Model Building*
        * Logistic regression
    5. *Model Benckmarking*
        * sklearn accuracy - 1.0%
        * Custom code accuracy - 1.0%

    ![ROC](https://github.com/DhatchayaniL/DD/assets/88893048/3b3a7359-d58b-416f-bf7f-f4d3c9ee6421)

7. ### Email spam classification - Multinomial naive bayes
    1. *Data cleaning*
        * No null values
        * Dropped all duplicates
    2. *Data visualization*
        * Add 2 necessary columns
        * All columns are positively correlated
          
          ![correlation](https://github.com/DhatchayaniL/DD/assets/88893048/5c98d2eb-8aaa-4950-8dc7-b656f22ee3e0)
  
          
        * Did stemming and lemmetization
        * Call is the most repeated work in spam messages
        * U is the most repeted work in ham messages
      
   <p align="center">
      <img src="https://github.com/DhatchayaniL/DD/assets/88893048/c2965974-7600-4d4b-bf1a-d48425248f68" width="400" alt="Visualization 1">
      <img src="https://github.com/DhatchayaniL/DD/assets/88893048/8e5f1904-ebfa-49cb-a748-e5086ae4a431" width="400" alt="Visualization 2">
   </p>
        
    3. *Model Building*
        * Multinomial naive bayes
    4. *Model Benckmarking*
        * sklearn accuracy - 96.9%
        * Custom code accuracy - 95.9%
    ![ROC](https://github.com/DhatchayaniL/DD/assets/88893048/af8c4aee-89c8-4916-a376-d2e297c9d0fc)

8. ### Salary prediction - Ada boost classifier
    1. *Data cleaning*
        * No null values
        * Dropped all duplicates
    2. *Data visualization*
  
       ![download](https://github.com/DhatchayaniL/DD/assets/88893048/2178b236-d0e0-468d-8fee-5dcd3a926d9f)
  
       
        * Imbalanced dataset
      
          ![download](https://github.com/DhatchayaniL/DD/assets/88893048/89ab916e-9976-4eb8-8897-c65ea1ec6864)    
          
        * Did oversampling using Random sampler
        * Work class of self-emp-inc is higher
        * Doctorate people are higher in the dataset
        * In marital status married-civ-spouse is higher
        * U is the most repeted work in ham messages
    4. *Model Building*
        * Ada boost classifier
    5. *Model Benckmarking*
        * sklearn accuracy - 80%
        * Custom code accuracy - 74%
      
          ![download](https://github.com/DhatchayaniL/DD/assets/88893048/e104a843-7b2b-4dc9-9568-ce425cfd6a9d)

10. ### Movie hits prediction - XG boost classifier
    1. *Data cleaning*
        * Had null values
        * Did iterative imputer
        * Dropped all duplicates
    2. *Data visualization*
        * Movies with drama genre are higher
        * Most number of movies are released in 2007
        * Most number of movies are release on friday
        * May is the highset average gross month
        * French movies are highest in number
    3. *Model Building*
        * XG boost classifier
    4. *Model Benckmarking*
        * sklearn accuracy - 81.7%
        * Custom code accuracy - 81.7%
      
## API Documentation for Your Package
### LinearRegression
Usage:
    
            from dd.dd.Taxi_price_linreg import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            

### LogisticRegressionCustom
Usage:

     
            from dd.dd.rain_in_australia import LogisticRegressionCustom
            model = LogisticRegressionCustom(learning_rate=0.01, num_iterations=1000, verbose=True)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
           

### MultinomialNaiveBayes
Usage:
    
    
            from dd.dd.email_spam_classification import MultinomialNaiveBayes
            model = MultinomialNaiveBayes(alpha=1.0)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
           

### KNNClassifier
Usage:

    
            from dd.dd.credit_car_spam import KNNClassifier
            model = KNNClassifier(k=3)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
         

### RandomForestRegressor
Usage:

    
            from dd.dd.car_price_predict import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
          

### DecisionTree
Usage:
    
     
        from dd.dd.avacado import DecisionTree
        model = DecisionTree(max_depth=None)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

       
### AdaBoostClassifierCustom
Usage:
    
    
        from dd.dd.salary import AdaBoostClassifierCustom
        model = AdaBoostClassifierCustom(n_estimators=100, random_state=1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
      

### XGBoostClassifier
Usage:
    
     
         from dd.dd.movies_hit_classification import XGBoostClassifier
         model = XGBoostClassifier()
         model.fit(X_train, y_train)
         predictions = model.predict(X_test)
      
