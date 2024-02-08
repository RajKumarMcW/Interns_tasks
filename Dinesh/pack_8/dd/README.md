Regression:

1. Avocado price prediction - Decision tree regressor
    1. Data cleaning
        * No null values
        * No duplicates
    2. Data visualization
        * Each and every columns are correlated
        * There is a positive relation between 'total volume' and all the other columns.
        * All the other columns are also positively correlated with all the columns except the target column.
        * Total volume can go above 3 if the average price is in between 0.5 to 2
        * Average price of 1.0 to 1.6 are high in frequency
    3. Model Building
        * Decision tree regressor
    4. Model Benckmarking
        * sklearn accuracy - 72.1%
        * Custom code accuracy - 70.3%

2. Car price prediction - Random forest regressor
    1. Data cleaning
        * Had null values
        * Numerical columns - Imputed using Iterative Imputer
        * Categorical columns - Imputed using Mode
        * Dropped all duplicates
    2. Data visualization
        * Selling price is maximum in 50,000 to 3Lakhs
        * Fuel efficiency is maximum at 20 km/L
        * Dropped LPG and CNG from fuel type as it is very less.
    3. Model Building
        * Random forest regressor
    4. Model Benckmarking
        * sklearn accuracy - 81.4%
        * Custom code accuracy - 80.4%

3. Taxi price prediction - Linear regression
    1. Data cleaning
        * No null values
        * Dropped all duplicates
    2. Data visualization
        * Dropped distance_traveled and num_of_passengers as they are less correlated with other columns.
        * Total fare and fare are highly correlated
        * miscellaneous fees and surge_applied are highly correlated in correlation matrix.
    3. Model Building
        * Linear regression
    4. Model Benckmarking
        * sklearn accuracy - 99%
        * Custom code accuracy - 99%

Classification

4. Credit card payment classification - KNN
    1. Data cleaning
        * No null values
        * Dropped all duplicates
    2. Data visualization
        * There is no relation between v1 to v28 columns
        * Did principal component analysis and reduces its dimentionality which increases correlation between v1 to v28 columns
        * Transaction amount is higher in 0 to 2000
        * Used robust scalar for normalization
    3. Model Building
        * K-Nearest Neightbour
    4. Model Benckmarking
        * sklearn accuracy - 94.3%
        * Custom code accuracy - 94.3%

5. Rain in australia - Logistic regression
    1. Data cleaning
        * Had null values
        * Used KNN - imputer to fill null values
        * Dropped all duplicates
    2. Data visualization
        * Imbalanced dataset
        * Did over sampling
        * Min Temp and Max Temp are positively correlated 
        * Temp3pm and Temp9am are positively correlated
    3. Model Building
        * Logistic regression
    4. Model Benckmarking
        * sklearn accuracy - 1.0%
        * Custom code accuracy - 1.0%

6. Email spam classification - Multinomial naive bayes
    1. Data cleaning
        * No null values
        * Dropped all duplicates
    2. Data visualization
        * Add 2 necessary columns
        * All columns are positively correlated
        * Did stemming and lemmetization
        * Call is the most repeted work in spam messages
        * U is the most repeted work in ham messages
    3. Model Building
        * Multinomial naive bayes
    4. Model Benckmarking
        * sklearn accuracy - 96.9%
        * Custom code accuracy - 95.9%

7. Salary prediction - Ada boost classifier
    1. Data cleaning
        * No null values
        * Dropped all duplicates
    2. Data visualization
        * Imbalanced dataset
        * Did oversampling using Random sampler
        * Work class of self-emp-inc is higher
        * Doctorate people are higher in the dataset
        * In marital status married-civ-spouse is higher
        * U is the most repeted work in ham messages
    3. Model Building
        * Ada boost classifier
    4. Model Benckmarking
        * sklearn accuracy - 80%
        * Custom code accuracy - 74%

8. Salary prediction - Ada boost classifier
    1. Data cleaning
        * Had null values
        * Did iteratie imputer
        * Dropped all duplicates
    2. Data visualization
        * Movies with drama genre are higher
        * Most number of movies are released in 2007
        * Most number of movies are release on friday
        * May is the highset average gross month
        * French movies are highest in number
    3. Model Building
        * XG boost classifier
    4. Model Benckmarking
        * sklearn accuracy - 81.7%
        * Custom code accuracy - 81.7%