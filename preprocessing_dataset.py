# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
def preprocess_dataset():
    
    dataset = pd.read_csv('german.csv')
    
    
    #encoding categoric to numeric
    dataset['Customer Risk Type'] = dataset['Customer Risk Type'].map({1:1,2:0 });

    dataset['Checking Account'] = dataset['Checking Account'].map({'A11':0,'A14':0,'A12':1,'A13':2})

    dataset['Credit History'] = dataset['Credit History'].map({'A34':0,'A30':1,'A33':2,'A31':3,'A32':4})

    #A47 doesn't exist
    dataset['Purpose'] = dataset['Purpose'].map({'A410':0,'A47':0,'A43':1,'A46':2,'A45':3,'A44':4,'A41':5,'A40':6,'A42':7,'A49':8,'A48':9})

    dataset['Savings Account'] = dataset['Savings Account'].map({'A65':0,'A61':1,'A62':2,'A63':3,'A64':4})

    dataset['Present employment since'] = dataset['Present employment since'].map({'A71':0,'A72':1,'A73':2,'A74':3,'A75':4})

    dataset['Personal Status and Sex'] = dataset['Personal Status and Sex'].map({'A91':0,'A92':1,'A93':2,'A95':3,'A94':4})

    dataset['Other Debtors/Guarantors'] = dataset['Other Debtors/Guarantors'].map({'A101':0,'A102':1,'A103':2})
    
    dataset['Property'] = dataset['Property'].map({'A124':0,'A123':1,'A122':2,'A121':3})

    dataset['Other installment plans'] = dataset['Other installment plans'].map({'A143':0,'A141':1,'A142':2})

    dataset['Housing'] = dataset['Housing'].map({'A153':0,'A151':1,'A152':2})

    dataset['Job'] = dataset['Job'].map({'A171':0,'A172':1,'A173':2,'A174':3})

    dataset['Telephone'] = dataset['Telephone'].map({'A191':0,'A192':1})

    dataset['Foreign Worker'] = dataset['Foreign Worker'].map({'A201':0,'A202':1})
    
    #splitting personal status and sex column for one hot encoding
    
    before_perstatus = dataset.iloc[:,:8]
    perstatus = dataset.iloc[:,8:9]
    after_perstatus = dataset.iloc[:,9:]
    
    ohe = OneHotEncoder(categorical_features='all')
    perstatus=ohe.fit_transform(perstatus).toarray()
    perstatus = pd.DataFrame(data = perstatus, index = range(1000), columns=['A91:Male Div','A92:Female Div','A93:Male Sing','A94: Male Married'])
    #combining columns again
    dataset = pd.concat([before_perstatus,perstatus,after_perstatus],axis = 1)
    
    #eliminating features with correlation
    corr = dataset.corr()

    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = dataset.columns[columns]
    dataset = dataset[selected_columns]
        
    return dataset
    
  
    
#Splitting dataset to train and test 
def train_test(dataset):
             
    X = dataset.iloc[:,:-1]
    Y = dataset.iloc[:,-1:]
    x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
     
    return x_train, x_test,y_train,y_test
     
    


def scale_dataset(x_train,x_test):

        #Scaling dataset, Min-Max scaler have better results than standart scaler
        #scaler = preprocessing.StandardScaler() 
        scaler = preprocessing.MinMaxScaler()       
        X_train = scaler.fit_transform(x_train)        
        X_test = scaler.transform(x_test)
        
        return X_train,X_test
    
  
#finding cost of confusion matrix and accuracy,in this specific dataset:
#cost is --> (0*TruePositive + 0*TrueNegative + 5*FalseNegative + 1*FalsePositive)/200 
def cost_accuracy(cm):
    
    accuracy = (cm[0][0]+cm[1][1]) / (cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
    cost =          ((cm[1][0] * 5) + (cm[0][1]*1) ) / 200
    return cost,accuracy   
    
    



    
    
    
    
    
    
