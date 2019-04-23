# -*- coding: utf-8 -*-

from preprocessing_dataset import preprocess_dataset
from preprocessing_dataset import train_test
from preprocessing_dataset import scale_dataset
from preprocessing_dataset import cost_accuracy
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



dataset = preprocess_dataset()

#making a function for support vector classifier
def support_vector_classifier(dataset):
  
    x_train, x_test,y_train,y_test = train_test(dataset)
    
    X_train, X_test = scale_dataset(x_train,x_test)    
    # SVC (SVM classifier)    
    svc = SVC(kernel='rbf',random_state=0,gamma='scale')
    svc.fit(X_train,y_train)
    svc_y_pred = svc.predict(X_test)
        
    cm_svc= confusion_matrix(y_test,svc_y_pred)
    cost,accuracy = cost_accuracy(cm_svc)
    
    return cost,accuracy,svc

#making backward elimination for eliminating features , based on p-value
def backward_elimination():

    before_bw_cost,before_bw_accuracy,svc = support_vector_classifier(dataset)
    
    X = dataset.iloc[:,:-1]
    Y = dataset.iloc[:,-1:]

    import statsmodels.api as sm
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
    #bw means backward elimination
    after_bw_dataset = pd.concat([X,Y],axis = 1)
   
    after_bw_cost,after_bw_accuracy,svc = support_vector_classifier(after_bw_dataset)
    
    return before_bw_cost,before_bw_accuracy,after_bw_cost,after_bw_accuracy,after_bw_dataset


#before_bw_cost,before_bw_accuracy,after_bw_cost,after_bw_accuracy,after_bw_dataset = backward_elimination()


#checking correctness of the model
def k_fold_cross_validation():
    before_bw_cost,before_bw_accuracy,after_bw_cost,after_bw_accuracy,after_bw_dataset = backward_elimination()
    x_train, x_test,y_train,y_test = train_test(after_bw_dataset)
    after_bw_cost,after_bw_accuracy,svc = support_vector_classifier(after_bw_dataset)
    X_train, X_test = scale_dataset(x_train,x_test) 
    success = cross_val_score(estimator = svc, X=X_train, y=y_train , cv = 4)
    
    return(success.mean())
    


#optimizing hyperparameters based on Grid Search
   
#    I made comment out optimizing parameters because
#    its decreasing accuracy rate and increasing cost rate ,this is not make sense.
      
   
    
#def optimizing_parameters():
#    
#    after_bw_cost,after_bw_accuracy,svc = support_vector_classifier(after_bw_dataset)
#    
#    p = [{'kernel':('linear','rbf','poly'),     
#          'gamma':('scale','auto'),
#          
#          } ]
#    
#    gs = GridSearchCV(estimator= svc, 
#                  param_grid = p,
#                  scoring =  'accuracy',
#                  cv = 10,
#                  n_jobs = -1)
#    x_train, x_test,y_train,y_test = train_test(after_bw_dataset)
#    X_train, X_test = scale_dataset(x_train,x_test)
#    
#    grid_search = gs.fit(X_train,y_train)
#    
#    best_parameters = grid_search.best_params_
#    
#    return best_parameters



#def final_svc():
#    
#    best_parameters = optimizing_parameters()
#
#    x_train, x_test,y_train,y_test = train_test(dataset)
#    
#    X_train, X_test = scale_dataset(x_train,x_test)    
#    # SVC (SVM classifier)    
#    svc = SVC(kernel = 'rbf', gamma= 'scale')
#    svc.fit(X_train,y_train)
#    svc_y_pred = svc.predict(X_test)
#        
#    cm_svc= confusion_matrix(y_test,svc_y_pred)
#    final_cost,final_accuracy = cost_accuracy(cm_svc)
#    
#    return final_cost,final_accuracy


















