# -*- coding: utf-8 -*-

from preprocessing_dataset import preprocess_dataset
from preprocessing_dataset import train_test
from preprocessing_dataset import scale_dataset
from optimizing_best_algorithm import backward_elimination
from preprocessing_dataset import cost_accuracy
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import statsmodels.api as sm

from optimizing_best_algorithm import backward_elimination
from sklearn.svm import SVR

#using support vector regression for predict probability
def support_vector_regression():
    
    dataset = preprocess_dataset()
    
   
    X = dataset.iloc[:,:-1]
    Y = dataset.iloc[:,-1:]
    #backward elimination with statsmodel
    X=sm.add_constant(X)
    model=sm.OLS(Y,X).fit()
    #print(model.summary())
                
    #1- Other Debtors/Guarantors have highest p-value
    X=X.drop(['Other Debtors/Guarantors'],axis=1)
    model=sm.OLS(Y,X).fit()
    #print(model.summary())
    #2- Job
    X=X.drop(['Job'],axis=1)
    model=sm.OLS(Y,X).fit()
    #print(model.summary())
    
    #3 - Present Residence Since
    X=X.drop(['Present Residence Since'],axis=1)
    model=sm.OLS(Y,X).fit()
    #print(model.summary())
    
    #4 - Savings Account
    X=X.drop(['Savings Account'],axis=1)
    model=sm.OLS(Y,X).fit()
    #print(model.summary())
              
    #5 - Liable to Provide
    X=X.drop(['Liable to Provide'],axis=1)
    model=sm.OLS(Y,X).fit()
    #print(model.summary())
                
    #6 - Foreign Worker
    X=X.drop(['Foreign Worker'],axis=1)
    model=sm.OLS(Y,X).fit()
    #print(model.summary())
                
    #7 - Checking Account
    X=X.drop(['Checking Account'],axis=1)
    model=sm.OLS(Y,X).fit()
    #print(model.summary())
               
    #8 - A91:Male Div
    X=X.drop(['A91:Male Div'],axis=1)
    model=sm.OLS(Y,X).fit()
    #print(model.summary())
                
    #9 - A92:Female Div
    X=X.drop(['A92:Female Div'],axis=1)
    model=sm.OLS(Y,X).fit()
    #print(model.summary())
                
    #10 - A94:Male Married
    X=X.drop(['A94: Male Married'],axis=1)
    model=sm.OLS(Y,X).fit()
     #print(model.summary())
                
     #11 - Age
    X=X.drop(['Age'],axis=1)
    model=sm.OLS(Y,X).fit()
     #print(model.summary())
                
     #12 - Existing Credits
    X=X.drop(['Existing Credits'],axis=1)
    model=sm.OLS(Y,X).fit()
      #print(model.summary())
            
    after_bw_dataset = pd.concat([X,Y],axis = 1)
    
    x_train, x_test,y_train,y_test = train_test(after_bw_dataset)
   
    X_train,X_test = scale_dataset(x_train,x_test)
    
    #support vector regression
    svr = SVR(kernel = 'rbf',gamma = 'scale') # rbf = radial bases function
    svr.fit(X_train,y_train)
    
    y_pred = svr.predict(X_test)
    
    
    return y_pred,y_test






































































#from sklearn.svm import SVR
#
#svr_reg = SVR(kernel = 'rbf',gamma='scale')
#svr_reg.fit(X_train,y_train)
#
#y_pred = svr_reg.predict(X_test)
#
#y_pred = (y_pred > 0.50)
#
#cm_svr= confusion_matrix(y_test,y_pred)
#
#cost,accuracy = cost_accuracy(cm_svr)