import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import time,re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import string
from string import ascii_uppercase


def preprocess_data(data, dataset_choice,target_column=None):
    if dataset_choice == 1:
        return preprocess_taxi(data)
    elif dataset_choice == 2:
        return preprocess_avocado(data)
    elif dataset_choice == 3:
        return preprocess_car_price(data)
    elif dataset_choice == 4:
        return preprocess_creditcard(data)
    elif dataset_choice == 5:
         return preprocess_emailspam(data)
    elif dataset_choice == 6:
         return preprocess_salary(data)
    elif dataset_choice == 7:
         return preprocess_rain(data)
    elif dataset_choice == 8:
        return preprocess_movie(data)
    elif dataset_choice == 9:
        return preprocess_general(data,target_column)
    else:
        raise ValueError("Invalid model type")

def remove_nulls_with_mode(df, cat_cols):
    df_no_nulls = df.copy()

    # Iterate through each categorical column
    for col in cat_cols:
        # Find the mode for the column
        mode_value = df[col].mode().values[0]

        # Replace missing values with the mode
        df_no_nulls[col].fillna(mode_value, inplace=True)

    return df_no_nulls

def preprocess_taxi(df, target_column='total_fare', test_size=0.2, random_state=None):
    print("Preprocessing started..")
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    X = df.drop(columns=target_column)   
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Preprocessing completed..")
    return X_train, X_test, y_train, y_test

def custom_one_hot_encoder(data, categorical_cols):
    # Create an empty DataFrame to store the encoded data
    encoded_data = pd.DataFrame(index=data.index)

    # List to store the names of encoded columns
    encoded_cols = []

    # Iterate over each categorical column
    for col in categorical_cols:
        # Skip 'Date' column
        if col == 'Date':
            continue

        # Use get_dummies to one-hot encode the current column
        one_hot_encoded = pd.get_dummies(data[col], prefix=col, drop_first=True)

        # Concatenate the one-hot encoded columns to the result DataFrame
        encoded_data = pd.concat([encoded_data, one_hot_encoded], axis=1)

        # Store the names of the encoded columns
        encoded_cols.extend(one_hot_encoded.columns)

    return encoded_data, encoded_cols

def preprocess_rain(df, target_column='RainTomorrow', test_size=0.2, random_state=None):
    # Drop rows with missing values in target_column
    df.dropna(subset=[target_column], inplace=True)

    # Apply KNN imputation for numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    imputer = KNNImputer(n_neighbors=3)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Extract years from the 'Date' column
    df['Year'] = pd.to_datetime(df['Date']).dt.year

    # Split the data into train, validation, and test based on years
    unique_years = df['Year'].unique()
    train_years, rest_years = train_test_split(unique_years, test_size=0.2, random_state=random_state)
    val_years, test_years = train_test_split(rest_years, test_size=0.25, random_state=random_state)

    train_df = df[df['Year'].isin(train_years)].copy()
    val_df = df[df['Year'].isin(val_years)].copy()
    test_df = df[df['Year'].isin(test_years)].copy()

    # Remove nulls with mode for categorical columns
    cat_cols = ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
    train_df = remove_nulls_with_mode(train_df, cat_cols)
    val_df = remove_nulls_with_mode(val_df, cat_cols)
    test_df = remove_nulls_with_mode(test_df, cat_cols)

    # Separate inputs and targets
    train_inputs = train_df.drop(columns=[target_column, 'Year', 'Date'])
    train_targets = train_df[target_column]

    val_inputs = val_df.drop(columns=[target_column, 'Year', 'Date'])
    val_targets = val_df[target_column]

    test_inputs = test_df.drop(columns=[target_column, 'Year', 'Date'])
    test_targets = test_df[target_column]

    # Encode categorical columns using custom_one_hot_encoder
    train_inputs_encoded, _ = custom_one_hot_encoder(train_inputs, cat_cols[1:])
    val_inputs_encoded, _ = custom_one_hot_encoder(val_inputs, cat_cols[1:])
    test_inputs_encoded, _ = custom_one_hot_encoder(test_inputs, cat_cols[1:])

    # Scale numeric columns
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

    # Apply SMOTE to balance the dataset
    y_train_labels = train_targets.replace({'Yes': 1, 'No': 0}).values
    train_inputs_encoded, encoded_cols = custom_one_hot_encoder(train_inputs, cat_cols[1:])
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(train_inputs_encoded[encoded_cols], y_train_labels)
    train_resampled_df = pd.DataFrame(data=X_train_resampled, columns=encoded_cols)
    train_resampled_df[target_column] = y_train_resampled

    return train_resampled_df, val_inputs_encoded, val_targets, test_inputs_encoded, test_targets, encoded_cols,train_targets, train_inputs_encoded




def preprocess_avocado(df):
    # Drop the 'Unnamed: 0' column
    print("preprocess started...")
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract month and day from 'Date'
    df['Month'] = df['Date'].apply(lambda x: x.month)
    df['Day'] = df['Date'].apply(lambda x: x.day)

    # One-hot encode categorical columns and drop the first category to avoid multicollinearity
    df_final = pd.get_dummies(df.drop(['region', 'Date'], axis=1), drop_first=True)

    # Separate features and target variable
    X = df_final.iloc[:, 1:14]
    y=df_final['AveragePrice']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Returning splitted datas")
    return X_train_scaled, X_test_scaled, y_train, y_test

######################################################################################################model 3
def extract_numeric_torque(torque_str):
        if isinstance(torque_str, str):
            # Extract the first numeric value
            numeric_match = re.search(r'\d+', torque_str)
            numeric_value = int(numeric_match.group()) if numeric_match else None
            return numeric_value
        else:
            return torque_str

def KNNImpute(df):
        # Extract numerical columns for kNN imputation
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Initialize KNNImputer
        imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors as needed

        # Perform kNN imputation on numeric columns
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        return df
  
def remove_nulls_with_mode(df, cat_cols):
        df_no_nulls = df.copy()

        # Iterate through each categorical column
        for col in cat_cols:
            # Find the mode for the column
            mode_value = df[col].mode().values[0]

            # Replace missing values with the mode
            df_no_nulls[col].fillna(mode_value, inplace=True)

        return df_no_nulls

def preprocess_car_price(df):
    # Drop duplicates
    print("Pre-processing started...")
    df = df.drop_duplicates()

    # Strip and clean specific columns
    df['mileage'] = df['mileage'].str.strip('kmpl').str.strip('km/kg')
    df['engine'] = df['engine'].str.strip('CC')
    df['max_power'] = df['max_power'].str.strip('bhp').str.strip()


    # Apply the extract_numeric_torque function to the 'torque' column and replace the entire column
    df['torque'] = df['torque'].apply(extract_numeric_torque)


    df = KNNImpute(df.copy())

    cat_cols = ['max_power','mileage','engine']
    df = remove_nulls_with_mode(df, cat_cols)

    df['age'] = 2024 - df['year']
    df.drop(['year'], axis=1, inplace=True)
    df['owner'] = df['owner'].replace({'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3})
    df['mileage'] = pd.to_numeric(df['mileage'])
    df['engine'] = pd.to_numeric(df['engine'])
    df['max_power'] = pd.to_numeric(df['max_power'])
    df_model = df.copy()

    # Create the 'brand' column by splitting the 'name' column
    df_model['brand'] = df_model['name'].str.split(' ').str.get(0)
    df_model.drop(['name'], axis=1, inplace=True)

    # Filter the outlier and log-transform the target variable('selling_price')
    df_model = df_model[df_model['selling_price'] < 2500000]
    df_model['selling_price'] = np.log(df_model['selling_price'])

    # Filter the outlier in 'km_driven' feature
    df_model = df_model[df_model['km_driven'] < 300000]

    # Filter the unwanted rows in 'fuel' feature
    df_model = df_model[~df_model['fuel'].isin(['CNG','LPG'])]

    # Filter the outliers in 'mileage' feature
    df_model = df_model[(df_model['mileage'] > 5) & (df_model['mileage'] < 35)]

    # Filter the outlier in 'max_power' feature and log-transform the data.
    df_model = df_model[df_model['max_power'] < 300]
    df_model['max_power'] = np.log(df_model['max_power'])

    # Log-transform the 'age' feature data.
    df_model['age'] = np.log(df_model['age'])
    df_model = pd.get_dummies(data=df_model, drop_first=True)
    df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_model.fillna(df_model.mean(), inplace=True)
    df_model.fillna(1e10, inplace=True)

    # Split into features and target variable
    X = df_model.drop(['selling_price'], axis=1)
    y = df_model['selling_price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print("Pre-processing completed...")
    X_train = X_train.astype(float)
    y_train = y_train.astype(float)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    return X_train, X_test, y_train, y_test

#################################################################################################model - 4

def preprocess_creditcard(df):
    # Drop duplicates
    print("Preprocess started...")
    df = df.drop_duplicates()

    # Initialize RobustScaler
    rob_scaler = RobustScaler()

    # Scale 'Amount' and 'Time'
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    # Drop original 'Time' and 'Amount'
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Save scaled 'Amount' and 'Time'
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']

    # Drop temporary scaled columns
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

    # Insert scaled 'Amount' and 'Time' at the beginning
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    # Shuffle the DataFrame
    df = df.sample(frac=1)

    # Separate fraud and non-fraud instances
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0][:492]

    # Concatenate and shuffle fraud and non-fraud instances
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    new_df = normal_distributed_df.sample(frac=1, random_state=42)

    # Split into features (X) and target variable (y)
    X = new_df.drop('Class', axis=1)
    y = new_df['Class']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    print("Preprocess completed...")
    return X_train, X_test, y_train, y_test

###################################################################################################model - 5

def transform_text(text,ps):
        text = text.lower()
        text = nltk.word_tokenize(text)

        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)

def preprocess_emailspam(df):
    print("Preprocessing started...")

    # Drop unnecessary columns
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    
    # Rename columns
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
    
    # Replace target labels
    df.replace({'target': {'ham': 0, 'spam': 1}}, inplace=True)
    
    # Remove duplicate rows
    df = df.drop_duplicates(keep='first')
    
    # Add features
    df['num_characters'] = df['text'].apply(len)
    df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
    df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    
    # Text preprocessing
    ps = PorterStemmer()

    

    df['transformed_text'] = df['text'].apply(lambda x: transform_text(x, ps))
    df['transformed_text'] = df['transformed_text'].str.replace(r'\d+', '')
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    X = tfidf.fit_transform(df['transformed_text']).toarray()
    y = df['target'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    print("Preprocessing completed...")
    
    return X_train, X_test, y_train, y_test


##################################################################################################### model - 6


def into_bins(column, bins):
        group_names = list(ascii_uppercase[:len(bins)-1])
        binned = pd.cut(column, bins, labels=group_names)
        return binned

def preprocess_salary(df):
    # Convert 'income' to binary labels
    print("Preprocessing started...")
    df['income'] = np.where(df['income'] == '>50K', 1, 0)

    # Remove rows with '?' in 'occupation' and 'workclass'
    df = df[(df['occupation'] != '?') & (df['workclass'] != '?')]

    # Create dummy variables for categorical columns
    education_dummies = pd.get_dummies(df['education'])
    marital_dummies = pd.get_dummies(df['marital.status'])
    relationship_dummies = pd.get_dummies(df['relationship'])
    sex_dummies = pd.get_dummies(df['sex'])
    occupation_dummies = pd.get_dummies(df['occupation'])
    native_dummies = pd.get_dummies(df['native.country'])
    race_dummies = pd.get_dummies(df['race'])
    workclass_dummies = pd.get_dummies(df['workclass'])

    

    # Create bins and dummy variables for 'capital.loss' and 'capital.gain'
    loss_bins = into_bins(df['capital.loss'], list(range(-1, 4500, 500)))
    loss_dummies = pd.get_dummies(loss_bins)
    gain_bins = into_bins(df['capital.gain'], list(range(-1, 42000, 5000)) + [100000])
    gain_dummies = pd.get_dummies(gain_bins)

    # Concatenate all features into the final X dataframe
    X = pd.concat([df[['age', 'hours.per.week']], gain_dummies, occupation_dummies, workclass_dummies, education_dummies, marital_dummies, race_dummies, sex_dummies], axis=1)

    # Extract the target variable 'income'
    y = df['income']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=1)
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    print("Preprocessing completed...")
    return X_train, X_test, y_train, y_test

############################################################################################################### model - 7


# def preprocess_rain(df):
#     print("Preprocessing started...")
#     # Drop rows with missing values in 'RainToday' and 'RainTomorrow'
#     df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

#     # Fill NaN values in categorical columns with the most common value
#     # Fill NaN values in numeric columns with the average of the column
#     for column in df.columns:
#         if df[column].dtype == 'object':
#             df[column].fillna(df[column].mode()[0], inplace=True)
#         else:
#             df[column].fillna(df[column].mean(), inplace=True)

#     # Drop specified columns
#     columns_to_drop = ['Date', 'Pressure3pm', 'Pressure9am', 'Rainfall', 'WindSpeed9am', 'WindSpeed3pm']
#     df = df.drop(columns=columns_to_drop)

#     # One-hot encode categorical columns
#     df = pd.get_dummies(df, columns=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], drop_first=True)

#     # Standardize numeric columns
#     numeric_columns = df.select_dtypes(include=['float64']).columns
#     scaler = StandardScaler()
#     df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

#     # Encode binary categorical variables using LabelEncoder
#     LE = LabelEncoder()
#     df['RainToday'] = LE.fit_transform(df['RainToday'])
#     df['RainTomorrow'] = LE.fit_transform(df['RainTomorrow'])

#     # Split the data into features (X) and target variable (y)
#     X = df.drop(['RainTomorrow'], axis=1)
#     y = df['RainTomorrow']
    
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     X_train = pd.get_dummies(X_train)
#     X_test = pd.get_dummies(X_test)
#     print("Preprocessing completed...")
#     X_train = X_train.to_numpy()
#     y_train = y_train.to_numpy()
#     return X_train, X_test, y_train, y_test

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_rain(df):
    print("Preprocessing started...")

    # Replace non-numeric values with NaN
    df.replace('M', np.nan, inplace=True)

    # Drop rows with missing values in 'RainToday' and 'RainTomorrow'
    df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

    # Fill NaN values in numeric columns with the average of the column
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert to numeric, replacing non-numeric with NaN
            df[column].fillna(df[column].mean(), inplace=True)

    # Drop specified columns
    columns_to_drop = ['Date', 'Pressure3pm', 'Pressure9am', 'Rainfall', 'WindSpeed9am', 'WindSpeed3pm']
    df = df.drop(columns=columns_to_drop)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], drop_first=True)

    # Convert boolean columns to numeric
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)

    # Standardize numeric columns
    numeric_columns = df.select_dtypes(include=['float64']).columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Encode binary categorical variables using LabelEncoder
    LE = LabelEncoder()
    df['RainToday'] = LE.fit_transform(df['RainToday'])
    df['RainTomorrow'] = LE.fit_transform(df['RainTomorrow'])

    # Split the data into features (X) and target variable (y)
    X = df.drop(['RainTomorrow'], axis=1)
    y = df['RainTomorrow']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(X_train.dtypes)
    # Convert to NumPy arrays
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    print("Preprocessing completed...")
    return X_train, X_test, y_train, y_test




########################################################################################### model - 8
import ast
from datetime import datetime

def clean_numeric(x):
    try:
        return float(x)
    except:
        return np.nan
        
def preprocess_movie(df):
    # Dropping duplicates and unnecessary columns
    print("Preprocess started...")
    df = df.drop_duplicates()
    df = df.drop(['imdb_id', 'original_title', 'adult'], axis=1)

    # Handling missing values and converting data types
    df['revenue'] = df['revenue'].replace(0, np.nan)
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['budget'] = df['budget'].replace(0, np.nan)
    df['return'] = df['revenue'] / df['budget']
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    df = df.drop('release_date', axis=1)
    df['title'] = df['title'].astype('str')
    df['overview'] = df['overview'].astype('str')

    # Engineering features
    df['production_countries'] = df['production_countries'].fillna('[]').apply(ast.literal_eval)
    df['production_countries'] = df['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    s = df.apply(lambda x: pd.Series(x['production_countries']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'countries'
    con_df = df.drop('production_countries', axis=1).join(s)
    con_df = pd.DataFrame(con_df['countries'].value_counts())
    con_df['country'] = con_df.index
    con_df.columns = ['num_movies', 'country']
    # con_df = con_df.reset_index().drop('index', axis=1)
    con_df = con_df.reset_index(drop=True)
    con_df = con_df[con_df['country'] != 'United States of America']

    # df_fran = df[df['belongs_to_collection'].notnull()]
    # df_fran['belongs_to_collection'] = df_fran['belongs_to_collection'].apply(ast.literal_eval).apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)
    df_fran = df.loc[df['belongs_to_collection'].notnull(), :].copy()
    df_fran['belongs_to_collection'] = df_fran['belongs_to_collection'].apply(ast.literal_eval).apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)

    df_fran = df_fran[df_fran['belongs_to_collection'].notnull()]
    fran_pivot = df_fran.pivot_table(index='belongs_to_collection', values='revenue', aggfunc={'revenue': ['mean', 'sum', 'count']}).reset_index()
    fran_pivot.sort_values('sum', ascending=False)
    fran_pivot.sort_values('mean', ascending=False)
    fran_pivot.sort_values('count', ascending=False)

    df['production_companies'] = df['production_companies'].fillna('[]').apply(ast.literal_eval)
    df['production_companies'] = df['production_companies'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    s = df.apply(lambda x: pd.Series(x['production_companies']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'companies'
    com_df = df.drop('production_companies', axis=1).join(s)
    com_sum = pd.DataFrame(com_df.groupby('companies')['revenue'].sum().sort_values(ascending=False))
    com_sum.columns = ['Total']
    com_mean = pd.DataFrame(com_df.groupby('companies')['revenue'].mean().sort_values(ascending=False))
    com_mean.columns = ['Average']
    com_count = pd.DataFrame(com_df.groupby('companies')['revenue'].count().sort_values(ascending=False))
    com_count.columns = ['Number']

    com_pivot = pd.concat((com_sum, com_mean, com_count), axis=1)
    com_pivot.sort_values('Total', ascending=False)
    com_pivot[com_pivot['Number'] >= 15].sort_values('Average', ascending=False)

    df['original_language'].drop_duplicates()
    lang_df = pd.DataFrame(df['original_language'].value_counts())
    lang_df['language'] = lang_df.index
    lang_df.columns = ['number', 'language']

    

    df['popularity'] = df['popularity'].apply(clean_numeric).astype('float')
    df['vote_count'] = df['vote_count'].apply(clean_numeric).astype('float')
    df['vote_average'] = df['vote_average'].apply(clean_numeric).astype('float')

    df[['title', 'popularity', 'year']].sort_values('popularity', ascending=False)
    df[['title', 'vote_count', 'year']].sort_values('vote_count', ascending=False)
    df['vote_average'] = df['vote_average'].replace(0, np.nan)
    df[df['vote_count'] > 2000][['title', 'vote_average', 'vote_count' ,'year']].sort_values('vote_average', ascending=False)

    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def get_month(x):
        try:
            return month_order[int(str(x).split('-')[1]) - 1]
        except:
            return np.nan

    def get_day(x):
        try:
            year, month, day = (int(i) for i in x.split('-'))
            answer = datetime.date(year, month, day).weekday()
            return day_order[answer]
        except:
            return np.nan

    # df['day'] = df['release_date'].apply(get_day)
    # df['month'] = df['release_date'].apply(get_month)
    df[df['year'] != 'NaT'][['title', 'year']].sort_values('year')

    months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    df_21 = df.copy()
    df_21['year'] = df_21[df_21['year'] != 'NaT']['year'].astype(int)
    df_21 = df_21[df_21['year'] >= 2000]
    # hmap_21 = pd.pivot_table(data=df_21, index='month', columns='year', aggfunc='count', values='title')
    # hmap_21 = hmap_21.fillna(0)

    df['spoken_languages'] = df['spoken_languages'].fillna('[]').apply(ast.literal_eval).apply(lambda x: len(x) if isinstance(x, list) else np.nan)
    df[df['spoken_languages'] >= 10][['title', 'year', 'spoken_languages']].sort_values('spoken_languages', ascending=False)

    df['runtime'] = df['runtime'].astype('float')
    df[df['runtime'] > 0][['runtime', 'title', 'year']].sort_values('runtime')
    df[df['runtime'] > 0][['runtime', 'title', 'year']].sort_values('runtime', ascending=False)

    df[df['budget'].notnull()][['title', 'budget', 'revenue', 'return', 'year']].sort_values('budget', ascending=False)

    df[(df['return'].notnull()) & (df['budget'] > 5e6)][['title', 'budget', 'revenue', 'return', 'year']].sort_values('return', ascending=False)

    df[(df['return'].notnull()) & (df['budget'] > 5e6) & (df['revenue'] > 10000)][['title', 'budget', 'revenue', 'return', 'year']].sort_values('return')

    df['year'] = df['year'].replace('NaT', np.nan)
    df['year'] = df['year'].apply(clean_numeric)
    df['genres'] = df['genres'].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    s = df.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'
    gen_df = df.drop('genres', axis=1).join(s)
    gen_df['genre'].value_counts().shape[0]

    pop_gen = pd.DataFrame(gen_df['genre'].value_counts()).reset_index()
    pop_gen.columns = ['genre', 'movies']
    genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Adventure', 'Science Fiction', 'Mystery', 'Fantasy', 'Mystery', 'Animation']
    violin_genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Science Fiction', 'Fantasy', 'Animation']
    violin_movies = gen_df[(gen_df['genre'].isin(violin_genres))]

    def convert_int(x):
        try:
            return int(x)
        except:
            return np.nan

    df['id'] = df['id'].apply(convert_int)
    df = df.drop([19730, 29503, 35587])
    df['id'] = df['id'].astype('int')
# , 'release_date'
    rgf = df[df['return'].notnull()]
    rgf = rgf.drop(['id', 'overview', 'poster_path', 'status', 'tagline', 'video', 'return'], axis=1)
    s = rgf.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'
    gen_rgf = rgf.drop('genres', axis=1).join(s)

    genres_train = gen_rgf['genre'].drop_duplicates()

    cls = df[df['return'].notnull()]
    # , 'release_date'
    cls = cls.drop(['overview', 'status', 'tagline', 'revenue'], axis=1)
    cls['return'] = cls['return'].apply(lambda x: 1 if x >= 1 else 0)
    cls['belongs_to_collection'] = cls['belongs_to_collection'].fillna('').apply(lambda x: 0 if x == '' else 1)
    s = cls.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'
    gen_cls = cls.drop('genres', axis=1).join(s)

    def classification_engineering(df):
        for genre in genres_train:
            df['is_' + str(genre)] = df['genres'].apply(lambda x: 1 if (isinstance(x, list) and genre in x) else 0)

        df['genres'] = df['genres'].apply(lambda x: len(x) if isinstance(x, list) else x)
        df = df.drop('homepage', axis=1)
        df['is_english'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)
        df = df.drop('original_language', axis=1)
        df['production_companies'] = df['production_companies'].apply(lambda x: len(x))
        df['production_countries'] = df['production_countries'].apply(lambda x: len(x))
        # df['is_Friday'] = df['day'].apply(lambda x: 1 if x == 'Fri' else 0)
        # df = df.drop('day', axis=1)
        # df['is_Holiday'] = df['month'].apply(lambda x: 1 if x in ['Apr', 'May', 'Jun', 'Nov'] else 0)
        # df = df.drop('month', axis=1)
        df = df.drop(['title'], axis=1)
        df['runtime'] = df['runtime'].fillna(df['runtime'].mean())
        df['vote_average'] = df['vote_average'].fillna(df['vote_average'].mean())
        return df

    cls = classification_engineering(cls)
    X, y = cls.drop('return', axis=1), cls['return']
    X = X.drop(['poster_path', 'video'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, stratify=y)
    print("preporcessing completed...")
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values
    return X_train, X_test, y_train, y_test

######################################################################################################### model - 9
def preprocess_general(df, target_column):
    # Step 1: Remove duplicates
    print("Preprocessing started...")
    df = df.drop_duplicates()

    # Step 2: Fill null values with mode for categorical columns and median for numerical columns
    for column in df.columns:
        if df[column].dtype == 'object':
            # Categorical column, fill with mode
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            # Numerical column, fill with median
            df[column].fillna(df[column].median(), inplace=True)

    # Step 3: Convert categorical values to numerical using Label Encoding
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Step 4: Train-test split
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Normalize X_train and X_test using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Preprocessing completed...")
    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = preprocess_general(df, 'your_target_column')