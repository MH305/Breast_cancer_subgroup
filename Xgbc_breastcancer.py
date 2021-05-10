#!/usr/bin/env python
# coding: utf-8

# In[2]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3
from numpy import loadtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV



def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
client_8e021aad30464e47871fd7d98c455d7d = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='LSRRj_wjJbLtxQr4T4UxOQRxcAd7TH9XmqiVfW8dchZK',
    ibm_auth_endpoint="https://iam.ng.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

# Your data file was loaded into a botocore.response.StreamingBody object.
# Please read the documentation of ibm_boto3 and pandas to learn more about the possibilities to load the data.
# ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
# pandas documentation: http://pandas.pydata.org/
streaming_body_1 = client_8e021aad30464e47871fd7d98c455d7d.get_object(Bucket='b4tm-donotdelete-pr-n5ggpcuynpqozw', Key='Train_call.txt')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(streaming_body_1, "__iter__"): streaming_body_1.__iter__ = types.MethodType( __iter__, streaming_body_1 ) 
train_data = pd.read_csv(streaming_body_1, delimiter='\t').T
#train_data = train_data.sort_values(by='Sample', axis=0)
train_data = train_data[4:]

#Normalizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(train_data)
print(X)


# Your data file was loaded into a botocore.response.StreamingBody object.
# Please read the documentation of ibm_boto3 and pandas to learn more about the possibilities to load the data.
# ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
# pandas documentation: http://pandas.pydata.org/
streaming_body_2 = client_8e021aad30464e47871fd7d98c455d7d.get_object(Bucket='b4tm-donotdelete-pr-n5ggpcuynpqozw', Key='Train_clinical.txt')['Body']
# add missing __iter__ method so pandas accepts body as file-like object
if not hasattr(streaming_body_2, "__iter__"): streaming_body_2.__iter__ = types.MethodType( __iter__, streaming_body_2 ) 
labels = pd.read_csv(streaming_body_2, delimiter = '\t', index_col = 0)
#from sklearn.preprocessing import OneHotEncoder
#encoder = OneHotEncoder()
#Y = encoder.fit_transform(labels).toarray()
Y = labels['Subgroup']
print(X)
print(Y.shape)
print(type(Y))
print(type(X))



# Your data file was loaded into a botocore.response.StreamingBody object.
# Please read the documentation of ibm_boto3 and pandas to learn more about the possibilities to load the data.
# ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
# pandas documentation: http://pandas.pydata.org/
streaming_body_3 = client_8e021aad30464e47871fd7d98c455d7d.get_object(Bucket='b4tm-donotdelete-pr-n5ggpcuynpqozw', Key='Validation_call.txt')['Body']
# add missing __iter__ method so pandas accepts body as file-like object
if not hasattr(streaming_body_3, "__iter__"): streaming_body_3.__iter__ = types.MethodType( __iter__, streaming_body_3 ) 
Test_data = pd.read_csv(streaming_body_3, delimiter = '\t', index_col = 0)
print(Test_data)



#write a function to take the following arguments: hidden_layers, number of node in each layer, 
# define the keras model


def ML_models(X,Y):
    
    train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.3, random_state = 42)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)  
    
    #models
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    #tree = DecisionTreeClassifier()
    #knn = KNeighborsClassifier(10)
    xgbc = xgb.XGBClassifier(learning_rate=0.001, max_depth=15, objective = 'multi:softmax', colsample_bytree = 0.9, gamma=0.5, subsample=0.3, lamda = 1, n_estimators=3000)

    # run the cross_val_score function on all four regression models and print them
    rf_scores = cross_val_score(rf, train_features, train_labels, cv=cv, scoring='accuracy' )
    #tree_scores = cross_val_score(tree, train_features, train_labels, cv=cv, scoring='accuracy' )
    #knn_scores = cross_val_score(knn, train_features, train_labels, cv=cv, scoring='accuracy' )
    xbg_scores = cross_val_score(xgbc, train_features, train_labels, cv=cv, scoring='accuracy')
    scores_list = [rf_scores, xbg_scores] #tree_scores,knn_scores
    for n in scores_list:
        print('scores per fold ', n)
        print('  mean score    ', np.mean(n))
        print('  standard dev. ', np.std(n))

    #Fit
    rf.fit(train_features,train_labels)
    xgbc.fit(train_features,train_labels)

    #Predict
    rf_output = rf.predict(test_features)
    xgbc_output = xgbc.predict(data=test_features)
    
    #Output
    #rf
    rf_df = pd.DataFrame()
    rf_df["Prediction"] = rf_output
    rf_df["Actual"] = test_labels
    rf_df.to_csv('rfr_output.csv')
    #xgb
    xgbc_df = pd.DataFrame()
    xgbc_df["Prediction"] = xgbc_output
    xgbc_df["Actual"] = test_labels
    xgbc_df.to_csv('xgbc_output.csv')

    #calculate accuracy score
    xgbc_accuracy = accuracy_score(xgbc_output,test_labels)
    print ('Accuracy_XGBC:', xgbc_accuracy*100)
    rf_acc_score = accuracy_score(rf_output,test_labels)
    print ('Accuracy_RF:', rf_acc_score*100)
    


# In[15]:


# Run
ML_models(X,Y)


# In[14]:


#write a function to take the following arguments: hidden_layers, number of node in each layer, 
# define the keras model


def ML_models_2(X,Y,Test_data):
    
    train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.3)    
    xgbc = xgb.XGBClassifier()
    rf = RandomForestClassifier()

    
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    parameters = {'objective':['multi:softmax'], 'learning_rate': [0.001],'max_depth': [15], 'subsample': [0.3], 'gamma':[0.5],'reg_lambda' : [1], 'colsample_bytree': [0.9], 'n_estimators': [3000]}
    #parameters_rf = {'n_estimators':[1000], 'random_state':[42]}
    clf_xgbc = GridSearchCV(xgbc, param_grid=parameters, n_jobs=5, cv=cv, scoring='accuracy', verbose=2, refit=True)
    #clf_rf = GridSearchCV(rf, param_grid=parameters_rf, verbose=2, refit=True, scoring='accuracy')
    

    xbg_scores = clf_xgbc.fit(train_features, train_labels)
    scores_list = [xbg_scores] #tree_scores,knn_scores

    #Fit
    xgbc.fit(train_features,train_labels)

    #Predict
    xgbc_output = xgbc.predict(data=test_features)
    
    #calculate accuracy score
    xgbc_accuracy = accuracy_score(xgbc_output,test_labels)
    print ('Accuracy_XGBC:', xgbc_accuracy*100)

    xgbc_predict = xgbc.predict(dtest)
    

    
    #Get scores
    for score in scores_list:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        print("Best parameters set found on development set for XGBC:")
        print()
        print(clf_xgbc.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf_xgbc.cv_results_['mean_test_score']
        stds = clf_xgbc.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf_xgbc.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        print(xgbc_output)
        print()


# In[15]:


ML_models_2(X,Y,Test_data)




print (xgbc.predict(Test_data))





