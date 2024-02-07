from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

def find_category_mappings(df, variable):
    return {k: i for i, k in enumerate(df[variable].dropna().unique(), 0)}

def integer_encode(df , variable, ordinal_mapping):
    df[variable] = df[variable].map(ordinal_mapping)

def clean(df1 , cols):
    mm = MinMaxScaler()
    mappin = dict()
    df = df1.copy()
    
    for variable in cols:
        mappings = find_category_mappings(df, variable)
        mappin[variable] = mappings

    
    for variable in cols:
        integer_encode(df, variable, mappin[variable])

    #Minmaxscaler and KNN imputation
    sca = mm.fit_transform(df)
    knn_imputer = KNNImputer()
    knn = knn_imputer.fit_transform(sca)
    df.iloc[:,:] = mm.inverse_transform(knn)
    for i in cols:
        df[i] = round(df[i]).astype('int')
    
    print(f'\n\nTotal duplicate rows: {df.duplicated().sum()}')
    df= df.drop_duplicates()
    print("removed duplicates")

    print("\n\nUnique Values:\n")
    [print(f"{col}: {df[col].unique()}") for col in df.columns]
    print("\n\nCorrelation:\n",df.corr())
    df  = df.apply(lambda col: mm.fit_transform(col.values.reshape(-1, 1)).flatten())
    
    return df

def balance_class(X,y):
    oversample = SMOTE()
    X,y = oversample.fit_resample(X,y)
    return X,y

def preprocess_bank(df):
    for column in df.columns:
        if df[column].dtype == 'object':  
            df[column].replace('unknown', np.NaN, inplace=True)  

    df=clean(df,['job','marital','education','default','housing','loan','y']) 
    X=df.drop(['y'],axis=1)
    y=df['y']
    X,y=balance_class(X,y)  
    return X,y
    
def preprocess_apple(df):
    df=clean(df,['Quality']) 
    X = df.drop(['A_id','Weight','Crunchiness','Acidity','Quality'],axis=1)
    y = df['Quality']
    X,y=balance_class(X,y)  
    return X,y

def preprocess_emp(df):
    df=clean(df,['Education','City','Gender','EverBenched'])  
    X = df.drop(['Education','City','EverBenched','ExperienceInCurrentDomain','LeaveOrNot'],axis=1)
    y = df['LeaveOrNot']
    X,y=balance_class(X,y)   
    return X,y

def preprocess_customer(df):
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    df.dropna(inplace=True)

    df['MultipleLines']=df['MultipleLines'].map({'No':0,'Yes':1,'No phone service':0})
    df['OnlineSecurity']=df['OnlineSecurity'].map({'No':0,'Yes':1,'No internet service':0})
    df['OnlineBackup']=df['OnlineBackup'].map({'No':0,'Yes':1,'No internet service':0})
    df['DeviceProtection']=df['DeviceProtection'].map({'No':0,'Yes':1,'No internet service':0})
    df['TechSupport']=df['TechSupport'].map({'No':0,'Yes':1,'No internet service':0})
    df['StreamingTV']=df['StreamingTV'].map({'No':0,'Yes':1,'No internet service':0})
    df['StreamingMovies']=df['StreamingMovies'].map({'No':0,'Yes':1,'No internet service':0})

    df=clean(df,['customerID','gender','Partner','Dependents','PhoneService','InternetService','Contract','PaperlessBilling','PaymentMethod','Churn'])   
    X = df.drop(['gender','PhoneService','MultipleLines','OnlineBackup','DeviceProtection','StreamingTV','StreamingMovies','customerID','Churn'],axis=1)
    y = df['Churn']
    X,y=balance_class(X,y)     
    return X,y

def preprocess_mushroom(df):
    df=clean(df,['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat'])
   
    X=df.drop(['veil-type','class'],axis=1)
    y=df['class']
    print("\nFeatures:\n",X)
    print("\nTarget\n",y)
    X,y=balance_class(X,y) 
    return X,y

def preprocess_car(df):
    df = df.drop(['Ad ID','Car Profile','Condition','Description','Car Features',"Images URL's"], axis=1)
    df=clean(df,['Car documents', 'Assembly','Transmission','Car Name','Make','Model','Fuel','Registration city','Seller Location'])
    
    X=df.drop(['Price'],axis=1)
    y=df['Price']
    print("\nFeatures:\n",X)
    print("\nTarget\n",y)
    return X,y

def preprocess_house(df): 
    df=df.drop(['Address'],axis=1) 
    df=clean(df,[])  
    df= df.drop_duplicates()
    
    X=df.drop(['Price'],axis=1)
    y = df['Price']
    return X,y

def preprocess_games(df): 
    # df=df.drop(['Address'],axis=1) 
    df=clean(df,['Name', 'Platform','Genre','Publisher','User_Score','Developer','Rating'])  
    df= df.drop_duplicates()
    
    X=df.drop(['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'],axis=1)
    y = df['Global_Sales']
    return X,y

def preprocess(df,dst_no):
    if dst_no=='1':
        X,y=preprocess_apple(df)
    elif dst_no=='2':
        X,y=preprocess_emp(df)
    elif dst_no=='3':
        X,y=preprocess_bank(df)
    elif dst_no=='4':
        X,y=preprocess_customer(df)
    elif dst_no=='5':
        X,y=preprocess_mushroom(df)
    elif dst_no=='6':
        X,y=preprocess_car(df)
    elif dst_no=='7':
        X,y=preprocess_house(df)
    elif dst_no=='8':
        X,y=preprocess_games(df)
  
    print(f'\nFeatures:\n{X} \n\nTarget:\n{y}')
    return X,y