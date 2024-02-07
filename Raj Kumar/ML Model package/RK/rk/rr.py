import numpy as np

class Ridge_Regression:
    def __init__(self,alpha=0.1):
        self.alpha=alpha
        self.coef_=None
        self.intercept_=None
        
    def __str__(self):
            return f"Ridge_Regression"

    def cfit(self,X_train,y_train):
        X_train=X_train.to_numpy()
        y_train=y_train.to_numpy()
        X_train=np.insert(X_train,0,1,axis=1)
        I=np.identity(X_train.shape[1])
        I[0][0]=0
        result=np.linalg.inv(np.dot(X_train.T,X_train)+self.alpha*I).dot(X_train.T).dot(y_train)
        self.intercept_=result[0]
        self.coef_=result[1:]

    def cpredict(self,X_test):
        return np.dot(X_test,self.coef_)+self.intercept_
    