a#!/usr/bin/python3

"""
Time to do some moddeling. This script will contain a simple model using our
features to calculated the popularity score. However, before this can be done
we need to split the data into a test and a training set

XBG boost
"""

# imports
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

def model_gen(csv):
    data = pd.read_csv(str(csv))
    data(csv)
    # Labels are the values we want to predict
    labels = np.array(data['mood (target)'])
    # Remove the labels from the features
    # axis 1 refers to the columns
    features = data.drop(columns="mood (target)")
    # Saving feature names for later use
    feature_list = list(features.columns)
    print (feature_list)
    # Convert to numpy array
    features = np.array(features)
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.33, random_state = 42)
    # Instantiate model with 1000 decision trees

    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    linear = LinearRegression()
    tree = DecisionTreeRegressor()
    knn = KNeighborsRegressor(10)
    xgbr = xgb.XGBRegressor(learning_rate=0.08, max_depth=10, objective='reg:linear', gamma=0.2, subsample=0.9, n_estimators=74)

    # run the cross_val_score function on all four regression models and print them
    rf_scores = cross_val_score(rf, train_features, train_labels, cv=3, scoring='neg_mean_absolute_error' )
    linear_scores = cross_val_score(linear, train_features, train_labels, cv=3, scoring='neg_mean_absolute_error' )
    tree_scores = cross_val_score(tree, train_features, train_labels, cv=3, scoring='neg_mean_absolute_error' )
    knn_scores = cross_val_score(knn, train_features, train_labels, cv=3, scoring='neg_mean_absolute_error' )
    xbg_scores = cross_val_score(xgbr, train_features, train_labels, cv=3, scoring='neg_mean_absolute_error')
    scores_list = [rf_scores, linear_scores, tree_scores, knn_scores, xbg_scores]
    for n in scores_list:
        print('scores per fold ', n)
        print('  mean score    ', np.mean(n))
        print('  standard dev. ', np.std(n))

    #Fit
    rf.fit(train_features,train_labels)
    xgbr.fit(train_features,train_labels)
    linear.fit(train_features,train_labels)

    #Predict
    rf_output = rf.predict(test_features)
    xgbr_output = xgbr.predict(data=test_features)
    linear_output = linear.predict(test_features)

    #Output
    #rf
    rf_df = pd.DataFrame()
    rf_df["Prediction"] = rf_output
    rf_df["Actual"] = test_labels
    rf_df.to_csv('rfr_output.csv')
    #xgb
    xgbr_df = pd.DataFrame()
    xgbr_df["Prediction"] = xgbr_output
    xgbr_df["Actual"] = test_labels
    xgbr_df.to_csv('xgbr_output.csv')
    #linear
    linear_df = pd.DataFrame()
    linear_df["Prediction"] = linear_output
    linear_df["Actual"] = test_labels
    linear_df.to_csv('linear_output.csv')

    #calculate loss

    #final_df.to_csv("Output_1.csv")
#    tree.fit(train_features,train_labels)
 #   Y=tree.predict(test_features)
  #  plt.figure()
#    plt.scatter(np.array(features),labels,s=20, edgecolor="black",
#            c="darkorange", label="data")
#    plt.plot(test_features,Y,color="cornflowerblue",
#         label="max_depth=2", linewidth=2)
#    plt.xlabel("data")
#    plt.ylabel("target")
#    plt.title("Decision Tree Regression")
#    plt.legend()
#    plt.show()

    #calculate mae score
    xgbr_mae_score = np.mean(abs(xgbr_output - test_labels))
    print ('Mean Absolute Error:', xgbr_mae_score)
    rf_mae_score = np.mean(abs(rf_output - test_labels))
    print ('Mean Absolute Error:', rf_mae_score)



def shuffle_data(csv):
    data = pd.read_csv(str(csv))
    data_shuf = data.sample(frac=1)
    data_shuf.to_csv(r'normalized3_csv.csv', index = None, header = True)

def main():
    csv = sys.argv[1]
    model_gen(csv)

if __name__ == '__main__':
    main()
