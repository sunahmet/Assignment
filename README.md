# Assignment
Hello,
I've tried to code this assignment in Python. I used Pandas, Numpy, Scikit-Learn, Keras, XGBoost libraries, and statsmodel API. On a computer with these tools, just run 'run.py' to see the results. I want to quickly mention "run.py" and other files in my folder. I'll also answer the pdf questions while explaining the files. First, let me talk about the file 'datatocsv.py', which I started first. This file simply converts 'german' file with '.data' extension to '.csv' format. That was the first challenge I had in the assignment.
I did the preprocessing operations in 'preprocessing_dataset.py'.
Specifically, I converted the categorical values to numerical values in increasing order of well-being. I've assigned 1 to Type of Good Risk and 0 to the Type of Bad Risk. It felt more comfortable this way. I used 'one hot encoder'
to handle the 'Personal status and Sex' column because I didn't want to assign increasing numerical values when I don't know how each one should affect the result.
Then I did a correlation analysis between columns. If I found a correlation above 0.9, I would do some deletion.
Because they would be linearly dependent on each other and they would have the same effect in this case it would be reasonable to delete one from the dataset, but no correlation was found. I've answered half of the 2nd problem right now. I'll talk about the P-value section later.
I will continue to explain the first problem. After looking at the correlation, I divided my dataset to train and test. Then I applied MinMaxScaler to these parts and I've got better results than StandardScaler. Then I wanted to calculate the cost and accuracy of the algorithms with the function 'cost_accuracy (cm)'. The cost was more important to me than accuracy because classfying someone who is bad risk type as good risk type has higher cost, and this was stated in the file 'german.doc' in the dataset and gave coefficient as 5. Let me continue with another file. In the file 'comparing_algorithms.py' I wrote the function 'algorithm_comparision()'. In this function, I created a dictionary to keep the accuracy and cost of the algorithms.
I compared classification algorithms such as Logistic Regression, KNN, Support Vector Classifier, Naive Bayes, Decision Tree, Random Forest, XGBoost, and Artificial Neural Networks. The Support Vector Classifier was seleted with the smallest cost (0.076), but an accuracy ratio close to others (0.72). Afterwards, I tried to optimize this algorithm in the file 'optimizing_best_algorithm.py'.
I started optimizing with Backward Elimination.
I checked the p-values on the dataset using the statsmodels API. Starting with the greatest one, I deleted the columns until the p-value is no larger than significance level (0.05).
I have 12 columns left. And so I get a %1 increase in accuracy and a %5 cost reduction. Although I could not get higher accuracy rates, I was delighted to bring the data to less cost and to increase the accuracy.Then I tried to check the correctness of my model by using the 'k fold cross validation' method.It would be wrong to think that the model I have randomly divided the data will show the same success everywhere. K fold cross validation is k times changing test part in dataset. And then we can check the correctness by looking at the average of these values. Then I tried to use grid search to optimize the parameters. However, even the best parameters I have found have decreased my accuracy rate and increased my cost rate.Then, in the file 'support_vector_regression.py' I tried to predict the probability that a customer would have a good risk or bad risk type. And finally, in the file 'run.py', I called the functions I created before and returned the following outputs:

cost : Cost ratio when I first use SVC,
accuracy : Accuracy ratio when I first use SVC,
final_cost : Cost ratio after the backward elimination,
final_accuracy : Accuracy ratio after the backward elimination,
success_k_fold : After the applied k fold cross validation, summing all parts and taking mean of this summing,

y_pred_reg	: Predict value based on Support Vector Machine,
y_test_reg : The real values against y_pred_reg
