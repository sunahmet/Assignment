# -*- coding: utf-8 -*-

from preprocessing_dataset import preprocess_dataset
from preprocessing_dataset import train_test
from preprocessing_dataset import scale_dataset
from optimizing_best_algorithm import backward_elimination
from preprocessing_dataset import cost_accuracy
from optimizing_best_algorithm import support_vector_classifier
from optimizing_best_algorithm import k_fold_cross_validation
from support_vector_regression import support_vector_regression

#cost is --> (0*TruePositive + 0*TrueNegative + 5*FalsePositive + 1*FalseNegative)/200
#200 is my sample size
#accuracy is --> (TruePositive + TrueNegative) / (TruePositive+TrueNegative+FalsePositive+FalseNegative)
def run():
    #loading dataset with preprocess operations from my library
    dataset = preprocess_dataset()
    
    #finding first cost and first accuracy from support vector classifier
    cost,accuracy,svc = support_vector_classifier(dataset)
    #you can see backward elimination effects   
    before_bw_cost,before_bw_accuracy,after_bw_cost,after_bw_accuracy,after_bw_dataset = backward_elimination()
    #checking correctness of the model based on k fold cross validation
    success_k_fold = k_fold_cross_validation()
    
    # After the backward elimination has finished I have 12 features. And this is the final results for classification
    final_cost,final_accuracy,final_svc = support_vector_classifier(after_bw_dataset)
    
    #predicting probability with regression
    y_pred_reg,y_test_reg= support_vector_regression()
    
    
    return cost,accuracy,final_cost,final_accuracy,success_k_fold,y_pred_reg,y_test_reg
    
 
    
if __name__ == "__main__":    
    cost,accuracy,final_cost,final_accuracy,success_k_fold,y_pred_reg,y_test_reg = run()

