from dj import preprocess, predict

dataset = {
    "flight": "./datasets/flight.csv",  #Logistic Regression
    "wine_quality": "./datasets/winequality.csv", #Decision Tree
    "house_price": "./datasets/houseprice.csv", #Ridge Regression
    "bike_share": "./datasets/london_merged.csv", #knnreg
    "water_quality": "./datasets/waterquality.csv", #SVM 
    "emotion_classify": "./datasets/Emotion_classify_Data.csv", #NB -
    "traffic": "./datasets/traffic.csv", #XGB -
    "income":"./datasets/income.csv"#knncla -
}

model_name = "Logistic Regression"
learning_rate = 0.002
num_iterations = 1000
sklearn=True
target_column = ""

def main():
    X_train,X_test,y_train,y_test = preprocess.prepreocess(dataset["flight"],target_column)
    predict.predict(model_name,X_train,X_test,y_train,y_test,learning_rate,num_iterations,sklearn)

if __name__ == "__main__":
    main()