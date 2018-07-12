# DecisionTreeLearning

The class file assignment2.java has logic to perfom the follwing tasks:
 
 
1.Implementation of the ID3 algorithm as described in Machine Learning by Tom Mitchell. The code handles continuous attributes as detailed in section 3.7.2 of the textbook (choosing the split with max information gain )and handles missing values approaches given in section 3.7.4.(in this case setting the maximum domain value)

2.Implement Reduced Error Pruning for the decision tree .Decision trees learnt using ID3 tend to overfit to the training data. One of the ways to overcome overfitting is Reduced Error Pruning. This is detailed in section 3.7.1.1 of the textbook.The validation set for this task is obtained by splitting the training data into the actual training set and the validation set.

3.Implementation of Random Forests .Though pruning can improve accuracy, one of the better ways to avoid overfitting is to construct Random Forests. A Random Forest is a bunch of decision trees, each learnt by making use of the dataset (containing N data points) that is randomly sampled from the training dataset (containing N data points). In the sampled dataset, a data point may be selected more than once from the training data set or the data point may not be selected.After obtaining the sampled data set as explained above, at each decision node, a subset of the attributes which haven’t been used at previous levels is chosen. The attribute among the sampled  attributes which has the highest information gain is chosen. The number of attributes sampled is either √p or log p where p is the total number of remaining attributes.The final output of a Random forest is the mode of the outputs of the individual trees.For more details refer to the following book: “Introduction to Data Mining” by Michael Steinbach, Pang-Ning Tan, and Vipin Kumar. 

Trained and tested each of the classifiers on the same dataset. Comparision of their performance is done in terms of
accuracy, precision, recall, F-measure and training time. Also mention the values of the various
hyper-parameters you’ve chosen. 

The code is generic and can handle any dataset.

Dataset: UCI Census Income dataset to evaluate the learning algorithms. It can be found at https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
A sample data point is:
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174,0, 40, United-States, <=50K


There are 14 attributes for each data point and the learning algorithms predict the final target attribute which indicates the income of a person (It takes one of two values - <=50K or >50K ). Missing attributes are giving the value of the most occuring domain value.

The adult.data file is used for training and the adult.test file is used while testing.
Additional information regarding the various attributes can be found at -
https://archive.ics.uci.edu/ml/datasets/Census+Income


#Outcome :

The accuracy obtained on the census data set is :
ID3 -> 82.03%
ID3 with reduced error pruning -> 83.48%
Random forest -> 1 tree - 82.22%
                 5 trees - 83.71%
                 10 trees - 84.28%
                 100 trees - 84.49%



