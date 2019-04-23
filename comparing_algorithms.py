# -*- coding: utf-8 -*-
from sklearn.metrics import confusion_matrix
from preprocessing_dataset import preprocess_dataset
from preprocessing_dataset import train_test
from preprocessing_dataset import scale_dataset
from preprocessing_dataset import cost_accuracy
import keras
from keras.models import Sequential 
from keras.layers import Dense  
from sklearn.linear_model import LogisticRegression
dataset = preprocess_dataset()

x_train, x_test,y_train,y_test = train_test(dataset)

X_train, X_test = scale_dataset(x_train,x_test)






def algorithm_comparision():
    cost_of_algorithms = {}
    accuracy_of_algorithms= {}
    #1. Logistic Regression
    logr = LogisticRegression(random_state=0)
    logr.fit(X_train,y_train) 
                        
    logr_y_pred = logr.predict(X_test) 
    #confusion matrix of logistic regression          
    cm_logr = confusion_matrix(y_test,logr_y_pred)    
    cost,accuracy = cost_accuracy(cm_logr)        
    cost_of_algorithms['Cost of Logistic Regression'] = cost 
    accuracy_of_algorithms['Accuracy of Logistic Regression'] = accuracy    
   
    # 2. KNN    
    from sklearn.neighbors import KNeighborsClassifier    
    knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski')
    knn.fit(X_train,y_train)   
    knn_y_pred = knn.predict(X_test)   
    cm_knn = confusion_matrix(y_test,knn_y_pred)
    cost,accuracy = cost_accuracy(cm_knn)
    cost_of_algorithms['Cost of K Nearest Neighborhood'] = cost 
    accuracy_of_algorithms['Accuracy of K Nearest Neighborhood'] = accuracy 
    
    
    # 3. SVC (SVM classifier)
    from sklearn.svm import SVC
    svc = SVC(kernel='rbf',random_state=0,gamma='scale')
    svc.fit(X_train,y_train)
    
    svs_y_pred = svc.predict(X_test)
    
    cm_svc= confusion_matrix(y_test,svs_y_pred)
    cost,accuracy = cost_accuracy(cm_svc)
    cost_of_algorithms['Cost of Support Vector Classifier'] = cost 
    accuracy_of_algorithms['Accuracy of Support Vector Classifier'] = accuracy 
    
    # 4. NAive Bayes
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)    
    gnb_y_pred = gnb.predict(X_test)
    cm_gnb= confusion_matrix(y_test,gnb_y_pred)
    cost,accuracy = cost_accuracy(cm_gnb)
    cost_of_algorithms['Cost of Naive Bayes'] = cost 
    accuracy_of_algorithms['Accuracy of Naive Bayes'] = accuracy 
    
    # 5. Decision tree
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion = 'entropy')    
    dtc.fit(X_train,y_train)
    dtc_y_pred = dtc.predict(X_test)    
    cm_dtc = confusion_matrix(y_test,dtc_y_pred)
    cost,accuracy = cost_accuracy(cm_dtc)
    cost_of_algorithms['Cost of Decision Tree'] = cost
    accuracy_of_algorithms['Accuracy of Decision Tree'] = accuracy
    
    
    # 6. Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=85)
    rfc.fit(X_train,y_train)    
    rfc_y_pred = rfc.predict(X_test)
    cm_rfc = confusion_matrix(y_test,rfc_y_pred)
    cost,accuracy = cost_accuracy(cm_rfc)
    cost_of_algorithms['Cost of Random Forest'] = cost
    accuracy_of_algorithms['Accuracy of Random Forest'] = accuracy
    
    # 7. XGBoost
    from xgboost import XGBClassifier
    xgb = XGBClassifier()    
    xgb.fit(X_train,y_train)    
    y_pred = xgb.predict(X_test)
    cm_xgb = confusion_matrix(y_test,y_pred)
    cost,accuracy = cost_accuracy(cm_xgb)
    cost_of_algorithms['Cost of XGBoost'] = cost
    accuracy_of_algorithms['Accuracy of XGBoost'] = accuracy    
    
    
    #8 Artificial Neural Network
    ann = Sequential()
    ann.add(Dense(10, init = 'uniform', activation = 'linear' , input_dim = 23))
    ann.add(Dense(10, init = 'uniform', activation = 'linear'))


    ann.add(Dense(1, init = 'uniform', activation = 'sigmoid'))

    ann.compile(optimizer = 'rmsprop', loss =  'binary_crossentropy' , metrics = ['accuracy'] )

    ann.fit(x_train, y_train, epochs=50)

    y_pred = ann.predict(x_test)

    y_pred = (y_pred > 0.50)
    
    cm_ann = confusion_matrix(y_test,y_pred)
    cost,accuracy = cost_accuracy(cm_ann)
    cost_of_algorithms['Cost of Artificial Neural Network'] = cost
    accuracy_of_algorithms['Accuracy of Artificial Neural Network'] = accuracy    
    

    return cost_of_algorithms,accuracy_of_algorithms

if __name__ == "__main__":
    
    cost_of_algorithms,accuracy_of_algorithms = algorithm_comparision()





