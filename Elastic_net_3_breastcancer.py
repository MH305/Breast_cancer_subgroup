from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
import numpy as np
import pandas as pd
from math import floor
from sklearn.linear_model import SGDClassifier

#Load data and transpose genome data, isolate class labels
data = pd.read_table('Train_call.txt')
labels = pd.read_table('Train_clinical.txt')
data = data.iloc[:,4:]
data = data.T
labels = labels.Subgroup.astype('category')
print(data.head())

# Number of trials
NUM_TRIALS = 2

# Data splitting in outer (3) and inner (CV -> 10) loop
skf_3 = StratifiedKFold(n_splits=3)
skf_10 = StratifiedKFold(n_splits=10)
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats= NUM_TRIALS)
skf = StratifiedKFold(n_splits=10)

# Set up possible values of parameters to optimize over
p_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
          "l1_ratio": [.5]}

# Stochastic gradient descent for a linear SVM
svm = SGDClassifier(loss="hinge", penalty= "elasticnet",max_iter=200)
clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=skf_10, return_train_score=False)

# Arrays to store scores
block6_scores = np.zeros(NUM_TRIALS)
block5_scores = np.zeros(NUM_TRIALS*3)
training_scores = np.zeros(NUM_TRIALS*3)

number_outer = 1
training_set = 3
for train_index, test_index in rskf.split(data, labels):
        print("Loop number:", number_outer, "Training set number:", training_set%3+1)
        #print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf.fit(X_train, y_train)
        print("Best CV score:", clf.best_score_)
        print("Best parameters:", clf.best_params_)
        print("Mean CV score", pd.DataFrame.from_dict(clf.cv_results_)['mean_test_score'].mean())
        print("Validaion score:", clf.score(X_test, y_test))
        block5_scores[training_set-3] = clf.score(X_test, y_test)
        training_scores[training_set-3] = clf.best_score_
        training_set +=1
        if training_set%3 ==0 :
            block6_score = block5_scores[training_set-6:training_set-4].mean()
            block6_scores[number_outer-1] = block6_score
            number_outer+=1
print("Blok 5 scores:", block5_scores)
print("Block 6 scores:", block6_scores)
print("Training scores:", training_scores)
print("Average training performance:", training_scores.mean())
print("Overall validation performance:", block6_scores.mean())
