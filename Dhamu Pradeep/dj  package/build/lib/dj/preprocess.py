
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import plotly.express as px
from imblearn.over_sampling import SMOTE
import contractions
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import  TfidfVectorizer
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn import metrics

def preprocess_for_logistic(data_path):
    data = pd.read_csv(data_path)
    print(data.head())
    print(f"Shape of the dataset : {data.shape}")
    data.drop(['Unnamed: 0','id'], axis=1, inplace=True)
    print(data.head())
    print(data.info())
    print(data.describe().T)
    print(f"Duplication Check : {data.duplicated().sum()}")
    df1 = data.copy()
    df1["Arrival Delay in Minutes"] = df1["Arrival Delay in Minutes"].fillna(df1["Departure Delay in Minutes"])
    print(f"Checking for null values : {df1.isnull().sum()}")
    print("The numerical features are:")
    num_features = df1.select_dtypes(include=['int64', 'float64'])
    for c in num_features.columns:
        print(c)
    print("The categorical features are:")
    cat_features = df1.select_dtypes(include=['object'])
    for c in cat_features:
        print(c)
    tf =df1[["satisfaction"]].value_counts()
    print(tf)

    lencoders = {}
    for col in df1.select_dtypes(include=['object']).columns:
        lencoders[col] = LabelEncoder()
        df1[col] = lencoders[col].fit_transform(df1[col])
    
    print(df1.head())
    r_scaler = preprocessing.MinMaxScaler()
    r_scaler.fit(df1)

    # df1 = pd.DataFrame(r_scaler.transform(df1), columns=df1.columns)
    print(df1.head())
    x = df1.drop("satisfaction", axis=1)
    y = df1["satisfaction"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    return X_train,X_test,y_train,y_test

def preprocess_for_knn_cla(data_path):
    data = pd.read_csv(data_path)
    print(data.head())
    print(f"Shape of the dataset : {data.shape}")
    print(data.info())
    print(data.describe().T)
    print(data.isnull().sum())
    categorical_features = []
    categorical_features = data.select_dtypes(include=['object']).copy()
    categorical_features.columns
    for col in list(categorical_features.columns):
        print(col +' has '+ str(categorical_features[col].nunique()) +' unique elements, which are :\n ' + str(categorical_features[col].unique()))
        print('\n')
        print(round(data[col].value_counts(normalize=True)*100,2).map(str)+'%')
        print('\n')  
    print(data.columns)
    data['workclass'].fillna(data['workclass'].mode()[0], inplace=True)
    data['occupation'].fillna(data['occupation'].mode()[0], inplace=True)
    data['native-country'].fillna(data['native-country'].mode()[0], inplace=True)
    print(data.isnull().sum())
    column=[]
    column =(pd.concat([categorical_features, data['income_>50K']], axis=1))
    column.loc[column['income_>50K']==0, ['income_>50K']] ='No'
    column.loc[column['income_>50K']==1, ['income_>50K']] ='Yes'
    print(column.head())
    aux = data.copy(deep=True)
    print(aux.head())
    wrk_govt = [ 'State-gov','Federal-gov','Local-gov']
    wrk_unemployed =['nan', 'Never-worked', 'Without-pay']
    wrk_self = ['Self-emp-not-inc','Self-emp-inc']
    for wrk in wrk_govt:
        aux['workclass'] = aux['workclass'].replace({wrk:'govt'})
    for wrk in wrk_unemployed:
        aux['workclass'] = aux['workclass'].replace({wrk:'unemployed'})
    for wrk in wrk_self:
        aux['workclass'] = aux['workclass'].replace({wrk:'self-emp'})
    edu_hs = ['1st-4th', '5th-6th','7th-8th', '9th', '10th', '11th','12th','HS-grad']
    edu_ass= ['Assoc-voc','Assoc-acdm','Some-college','Prof-school']

    for edu in edu_hs :
        aux['education']=aux['education'].replace({edu:'High-school'})
    for edu in edu_ass:
        aux.education = aux.education.replace({edu:'Association'})
    mar_single = ['Separated','Widowed','Married-spouse-absent']
    mar_marriage =['Married-civ-spouse','Married-AF-spouse']

    for mar in mar_single:
        aux['marital-status'] =  aux['marital-status'].replace({mar:'Single'})
    for mar in mar_marriage:
        aux['marital-status'] =  aux['marital-status'].replace({mar:'Married'})

    rel_mar = ['Husband', 'Wife']

    for rel in rel_mar:
        aux.relationship = aux.relationship.replace({rel:'Married'})

    c_nna = ['United-States','Mexico','Dominican-Republic','Puerto-Rico','Cuba','El-Salvador','Canada','Guatemala','Haiti','Nicaragua','Honduras','Jamaica','Outlying-US(Guam-USVI-etc)']
    c_sa= ['Ecuador','Columbia','Peru','Trinadad&Tobago','South']
    c_asia= ['Japan','Philippines','China','Vietnam','Thailand','India','Cambodia','Iran','Taiwan','Laos','Hong']
    c_europe = ['Portugal','Italy','England','Germany','Yugoslavia','Poland','Greece', 'Ireland','Scotland','France','Hungary','Holand-Netherlands']

    for c in c_nna:
        aux['native-country'] = aux['native-country'].replace({c:'Central and North America'})
    for c in c_sa:
        aux['native-country'] = aux['native-country'].replace({c:'South America'})
    for c in c_asia:
        aux['native-country']=aux['native-country'].replace({c:'Asia'})
    for c in c_europe:
        aux['native-country'] = aux['native-country'].replace({c:'Europe'})

    cat_features = [col for col in list(aux.columns) if aux[col].dtypes==object]
    print(cat_features)

    num_features = [col for col in list(aux.columns) if aux[col].dtypes!=object and col!='income_>50K']
    print(num_features)

    d = LabelEncoder()

    for col in aux[cat_features]:
        aux[col]= d.fit_transform(aux[col])
    print(aux.head())

    # Assuming 'df' is your DataFrame
    description = aux.describe()

    # Display the descriptive statistics
    print(description)

    scaler = MinMaxScaler()
    aux[num_features] = scaler.fit_transform(aux[num_features])
    print(aux.head())

    X = aux.drop("income_>50K",axis=1)
    y = aux['income_>50K']

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate random indices for the test set
    test_indices = np.random.choice(X.index, size=int(0.3 * len(X)), replace=False)

    # Create the training and testing sets
    X_train = X.drop(test_indices)
    y_train = y.drop(test_indices)
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]

    # Check the shape of the resulting sets
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return X_train,X_test,y_train,y_test


def preprocess_for_knn_reg(data_path):
    data = pd.read_csv(data_path)
    print(data.head())
    print(f"Shape of the dataset : {data.shape}")
    print(data.info())
    print(data.describe().T)
    print(data.describe().T)
    df = data
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df= df.set_index('timestamp')
    df['year_month']= df.index.strftime('%Y-%m')
    df['year'] = df.index.year
    df['month']= df.index.month
    df['day_of_week']=df.index.dayofweek
    df['hour']=df.index.hour
    print(df.head())
    X = df.drop(columns = ["cnt","year_month"], axis=1)
    Y = df["cnt"]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    rf_model.fit(X, Y)

    # Get importances features
    feature_importances = rf_model.feature_importances_

    # Create a DataFrame to display feature importances
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Display the sorted DataFrame
    print(feature_importance_df)
    X.drop(columns=["month","season","year"])
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state = 42)
    return X_train,X_test,Y_train,Y_test

def preprocess_for_decision_tree(data_path):
    data = pd.read_csv(data_path)
    print(data.head())
    print(f"Shape of the dataset : {data.shape}")
    print(data.info())
    print(f"Duplication Check : {data.duplicated().sum()}")
    print("Dropping Duplicates ... ")
    data = data.drop_duplicates()
    print(f"After Dropping : {data.duplicated().sum()}")
    print(f"Shape of the dataset : {data.shape}")
    print(f"Checking for null values : {data.isnull().sum()}")
    cols_with_missing_values = data.columns[data.isnull().sum() > 0]
    print(cols_with_missing_values)
    data = data.drop('type', axis = 1)
    train_set = data.dropna(subset=cols_with_missing_values)
    test_set = data[data['fixed acidity'].isnull() | data['volatile acidity'].isnull() | data['citric acid'].isnull() | data['residual sugar'].isnull() | data['chlorides'].isnull() | data['pH'].isnull() |data['sulphates'].isnull() ]
    linear = LinearRegression()
    linear.fit(train_set[['quality']],train_set[['fixed acidity','volatile acidity','citric acid','residual sugar']])
    predicted_values = linear.predict(test_set[['quality']])
    data.loc[test_set.index,['fixed acidity','volatile acidity','citric acid','residual sugar']] = predicted_values
    rfr = RandomForestRegressor()
    rfr.fit(train_set[['quality']],train_set[['chlorides','pH','sulphates']])
    predicted_values = rfr.predict(test_set[['quality']])
    data.loc[test_set.index,['chlorides','pH','sulphates']] = predicted_values
    print(f"After filling null values using LinearRegression and  RandomForestRegressor \n: {data.isnull().sum()}")

    x = data.drop("quality", axis=1)
    y = data["quality"] 
    Y = np.array(y).ravel()
    # Create a RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    rf_model.fit(x, Y)
    # Get importances features
    feature_importances = rf_model.feature_importances_

    # Create a DataFrame to display feature importances
    feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances})

    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Display the sorted DataFrame
    print(feature_importance_df)
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.20,random_state = 42)

    # sc = StandardScaler()

    # X_train = sc.fit_transform(X_train)
    # X_test = sc.fit_transform(X_test)

    # y_train = sc.fit_transform(Y_train.reshape(-1, 1))
    # y_test = sc.fit_transform(Y_test.reshape(-1, 1))
    # print("After normalization ... ")
    # print(f"X_train : {X_train}")
    # print(f"Y_train : {y_train}")
    # print(f"X_test : {X_train}")
    # print(f"Y_test : {y_train}")

    return X_train,X_test,Y_train,Y_test
    

def preprocess_for_redge_reg(data_path):
    data = pd.read_csv(data_path)
    print(data.head())
    print(f"Shape of the dataset : {data.shape}")
    print(data.info())
    print(f"Duplication Check : {data.duplicated().sum()}")
    print("Dropping Duplicates ... ")
    data = data.drop_duplicates()
    print(f"Checking for null : {data.isnull().sum()}")
    print(f"number of unique values : {data.nunique()}")
    print(data.describe().T)
    data = data.drop(['POSTED_BY','BHK_OR_RK','ADDRESS'], axis = 1)

    
    x = data.drop("TARGET(PRICE_IN_LACS)", axis=1)
    y = pd.DataFrame(data["TARGET(PRICE_IN_LACS)"])
    Y = np.array(y).ravel()
    # Create a RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    rf_model.fit(x, Y)

    # Get importances features
    feature_importances = rf_model.feature_importances_

    # Create a DataFrame to display feature importances
    feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances})

    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Display the sorted DataFrame
    print(feature_importance_df)

    x = data.drop(['UNDER_CONSTRUCTION','READY_TO_MOVE'], axis=1)
    print("Droping UNDER_CONSTRUCTION and READY_TO_MOVE")
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.20,random_state = 42)
    # sc = StandardScaler()

    # X_train = sc.fit_transform(X_train)
    # X_test = sc.fit_transform(X_test)

    # y_train = sc.fit_transform(Y_train.reshape(-1, 1))
    # y_test = sc.fit_transform(Y_test.reshape(-1, 1))

    # print("After normalization ... ")
    # print(f"X_train : {X_train}")
    # print(f"Y_train : {y_train}")
    # print(f"X_test : {X_train}")
    # print(f"Y_test : {y_train}")

    return X_train,X_test,Y_train,Y_test

def preprocess_for_svm(data_path):
    data = pd.read_csv(data_path)
    print(data.head())
    print(f"Shape of the dataset : {data.shape}")
    print(data.info())
    print(f"Duplication Check : {data.duplicated().sum()}")
    print(f"Checking for null : {data.isnull().sum()}")
    print(f"number of unique values : {data.nunique()}")
    print(data.describe().T)
    data = data.replace(to_replace="#NUM!", value="0")
    print(data[["is_safe"]].value_counts())

    x = data.drop("is_safe", axis=1)
    y = data["is_safe"]
    smote = SMOTE(random_state=2)
    X_resampled, y_resampled = smote.fit_resample(x, y)
    print(f"After applying SMOTE : {y_resampled.value_counts()}")
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    rf_model.fit(X_resampled, y_resampled)

    # Get importances features
    feature_importances = rf_model.feature_importances_

    # Create a DataFrame to display feature importances
    feature_importance_df = pd.DataFrame({'Feature': X_resampled.columns, 'Importance': feature_importances})

    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Display the sorted DataFrame
    print(feature_importance_df)

    X_resampled = X_resampled.drop(['flouride','chromium','mercury'], axis=1)

    print(f"Dropping Flouride, Chromium, Mercury.")

    # data = pd.concat([X_resampled, y_resampled], axis=1)

    # x = data.drop("is_safe", axis=1)
    # y = data["is_safe"]
    # r_scaler = preprocessing.MinMaxScaler()
    # r_scaler.fit(X_resampled)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=42)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    y_train = pd.to_numeric(y_train, errors='coerce')
    y_test = pd.to_numeric(y_test, errors='coerce')

    return X_train,X_test,y_train,y_test


def remove_punctuation(text):
    '''This function is for removing punctuation'''
    # replacing the punctuations with no space, hence punctuation marks will be removed
    translator = text.translate(str.maketrans('', '', string.punctuation))
    # return the text stripped of punctuation marks
    return (translator)
    #remove punctuation using function created

def download():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

def remove_stopwords(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    sw = stopwords.words('english')
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)

def lemmatize_tokens(tokens):
    '''function for lemmatization'''
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def preprocess_for_naive(data_path):
    df = pd.read_csv(data_path)
    print(df.head())
    print(df.describe())
    print(df['Emotion'].unique())
    print(df.isnull().sum())
    print(sns.countplot(x = df['Emotion']))
    df['Comment']=df['Comment'].apply(lambda x:contractions.fix(str(x)))
    df['Comment'] = df['Comment'].apply(remove_punctuation)
    download()
    df['Comment'] = df['Comment'].apply(remove_stopwords)
    # Remove White spaces
    df['Comment'] =df['Comment'].apply(lambda x: " ".join(x.split()))
    # Tokenization
    df['Comment'] = df['Comment'].apply(nltk.word_tokenize)
    df.sample(3)
    # Create a lemmatizer

    # Lemmatize the 'Review' column
    df['Comment'] = df['Comment'].apply(lemmatize_tokens)
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    X = df['Comment']
    X= vectorizer.fit_transform(X)
    dictionary = vectorizer.vocabulary_.items()
    print(dictionary)
    # Get unique labels from the 'Emotion' column
    unique_labels = df['Emotion'].unique()

    # Create a mapping for each unique label
    emotion_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    # Map the 'Emotion' column to numerical values
    df['Emotion_num'] = df['Emotion'].map(emotion_mapping)

    print(df['Emotion_num'].unique())

    y = df['Emotion_num']

    # Convert X to a dense array if it's a sparse matrix
    X_dense = X.toarray() if isinstance(X, coo_matrix) else X

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Define the proportion of data for training (e.g., 80%)
    train_proportion = 0.8

    # Determine the number of samples for training
    num_train_samples = int(train_proportion * X_dense.shape[0])

    # Create an array of indices for shuffling
    indices = np.arange(X_dense.shape[0])
    np.random.shuffle(indices)

    # Use the shuffled indices to split the data
    train_indices = indices[:num_train_samples]
    test_indices = indices[num_train_samples:]

    # Create training and testing sets
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    print(f"y_train \n : {y_train}")
    
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    return X_train,X_test,y_train,y_test

def preprocess_for_XGB(data_path):
    df = pd.read_csv(data_path)
    print(df.head())
    print(f"Shape of the dataset : {df.shape}")
    columns_to_introduce_missing_values = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']
    percentage_to_replace = 0.03  # 3% of values will be replaced

    for column in columns_to_introduce_missing_values:
        mask = np.random.rand(len(df)) < percentage_to_replace
        df.loc[mask, column] = np.nan

    print(df['Day of the week'].unique())
    print(df['Date'].unique())
    print(df['Traffic Situation'].unique())
    print(f"Checking for null : {df.isnull().sum()}")
    train_set = df.dropna(subset=['CarCount', 'BikeCount', 'BusCount', 'TruckCount'])
    test_set = df[df['CarCount'].isnull() | df['BikeCount'].isnull() | df['BusCount'].isnull() | df['TruckCount'].isnull()]
    features = df[['Total']]
    # Linear Regression for 'CarCount' and 'BikeCount'
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(train_set[['Total']], train_set[['CarCount', 'BikeCount']])
    predicted_car_bike = linear_reg_model.predict(test_set[['Total']])
    df.loc[test_set.index, ['CarCount', 'BikeCount']] = predicted_car_bike

    # Random Forest Regression for 'BusCount' and 'TruckCount'
    random_forest_model = RandomForestRegressor()
    random_forest_model.fit(train_set[['Total']], train_set[['BusCount', 'TruckCount']])
    predicted_bus_truck = random_forest_model.predict(test_set[['Total']])
    df.loc[test_set.index, ['BusCount', 'TruckCount']] = predicted_bus_truck

    print(df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']].isnull().sum())

    # Assuming 'df' is your DataFrame
    description = df.describe()

    # Display the descriptive statistics
    print(description)

    print(df['Traffic Situation'].value_counts())

    # Count the occurrences of each class
    class_counts = df['Traffic Situation'].value_counts()

    # # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightblue', 'lightgreen', 'gold'], startangle=90)

    # # Add a title
    plt.title('Distribution of Traffic Situations')

    # # Show the plot
    # plt.show()
    print(df['Day of the week'].value_counts())
    print(df["Date"].value_counts())
    # Assuming 'features_to_normalize' is a list of numerical features to be normalized
    features_to_normalize = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Apply Min-Max Scaling to the specified features
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

    # Display the normalized DataFrame
    print(df.head())
    description = df.describe()

    # Display the descriptive statistics
    print(description)

    df['TotalVehicles'] = df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']].sum(axis=1)

    most_contributing_vehicle = df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']].idxmax(axis=1)
    df['MostContributingVehicle'] = most_contributing_vehicle


    daily_traffic = df.groupby('Day of the week')['TotalVehicles'].sum()
    busiest_days = daily_traffic.idxmax()
    print(f"The busiest days for traffic are: {busiest_days}")

    df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p')
    df['Hour'] = df['Time'].dt.hour

    # Group by hour and calculate the total vehicles for each hour
    hourly_traffic = df.groupby('Hour')['TotalVehicles'].sum()


    # Identify the peak hours with the highest total vehicles
    peak_hours = hourly_traffic.nlargest(2).index
    print(f"The peak hours of traffic are between {peak_hours[0]}:00 and {peak_hours[1]}:00")

    print(df.columns)
    print(df.dtypes)
    df = df.drop(columns=['Time'])
    df=df.drop(columns=['MostContributingVehicle'])
    # Get unique days from the 'Day of the week' column
    unique_days = df['Day of the week'].unique()

    # Create a mapping for each unique day
    day_mapping = {day: idx for idx, day in enumerate(unique_days)}

    # Map the 'Day of the week' column to numerical values
    df['Day_of_week_num'] = df['Day of the week'].map(day_mapping)
    df=df.drop(columns=['Day of the week'])
    df=df.drop(columns=['TotalVehicles'])
    # Get unique labels from the 'Traffic Situation' column
    unique_traffic_situations = df['Traffic Situation'].unique()

    # Create a mapping for each unique label
    traffic_situation_mapping = {situation: idx for idx, situation in enumerate(unique_traffic_situations)}

    # Map the 'Traffic Situation' column to numerical values
    df['Traffic Situation'] = df['Traffic Situation'].map(traffic_situation_mapping)

    print(df.dtypes)
    # Separate features (X) and target variable (y)
    X = df.drop(columns=['Traffic Situation'])  # Assuming 'Traffic Situation' is the target column
    y = df['Traffic Situation']

    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate random indices for the test set
    test_indices = np.random.choice(X.index, size=int(0.2 * len(X)), replace=False)

    # Create the training and testing sets
    X_train = X.drop(test_indices)
    y_train = y.drop(test_indices)
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]

    # Check the shape of the resulting sets
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)


    # Separate features (X) and target variable (y)
    X = df.drop(columns=['Traffic Situation'])
    y = df['Traffic Situation']

    
    # Initialize the RandomForestClassifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Train the model
    rf_classifier.fit(X, y)

    # Get feature importances
    feature_importances = rf_classifier.feature_importances_

    # Create a DataFrame to visualize feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })

    # Sort features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    num_classes = len(np.unique(y_train))
    class_weights = len(y_train) / (num_classes * np.bincount(y_train))

    return X_train,X_test,y_train,y_test

def prepreocess_common(data_path,target):
    data = pd.read_csv(data_path)
    print(data.head())
    print(f"Shape of the dataset : {data.shape}")
    print(data.info())
    print(f"Duplication Check : {data.duplicated().sum()}")
    print("Dropping Duplicates ... ")
    data = data.drop_duplicates()
    print(f"Checking for null : {data.isnull().sum()}")
    data = data.dropna()
    print(f"Checking for null : {data.isnull().sum()}")
    print(f"number of unique values : {data.nunique()}")
    print(data.describe().T)
    lencoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        lencoders[col] = LabelEncoder()
        data[col] = lencoders[col].fit_transform(data[col])
    
    scaler = MinMaxScaler()
    # Apply Min-Max Scaling to the specified features
    scaler.fit(data)
    print(data[["is_safe"]].value_counts())
    # Display the normalized DataFrame
    print(data)
    x = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    return X_train,X_test,y_train,y_test


def prepreocess(data,target):
    if data.find("winequality") != -1:
        return preprocess_for_decision_tree(data)
    elif data.find("Emotion_classify_Data") != -1:
        return preprocess_for_naive(data)
    elif data.find("flight") != -1:
        return preprocess_for_logistic(data)
    elif data.find("houseprice") != -1:
        return preprocess_for_redge_reg(data)
    elif data.find("waterquality") != -1:
        return preprocess_for_svm(data)
    elif data.find("traffic") != -1:
        return preprocess_for_XGB(data)
    elif data.find("london_merged") != -1:
        return preprocess_for_knn_reg(data)
    elif data.find("income") != -1:
        return preprocess_for_knn_cla(data)
    else:
        return prepreocess_common(data,target)