import numpy as np
import pandas as pd
import math

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

workfolder = "C:\\Users\\orent\\Documents\\Data Science Exercise\\"
all_data_path = workfolder + "hw_analysis.csv"
test_set_path = workfolder + "hw_analysis_test_set.csv"
training_set_path = workfolder + "hw_analysis_training_set.csv"

TEST_SET_SIZE = 0.2 

data_all_original_df = pd.read_csv(all_data_path)

########################################################################
# Improvements for future: this should be implemented as a pipeline.

# Data Transformations: 
# 1. Remove missing values
# 2. Drop the outliers
# 3. Drop X10 column
# 4. Use one-hot encoding for the features 
# 5. Scale all values to be between 0 - 1
# 6. Separate Outcome variable from the independent variables
# 7. Combine one-hot-encoded columns with the other independent variables

data_all_df = data_all_original_df.dropna()
data_all_df = data_all_df[data_all_df.X1 != -17.3] 
data_all_df = data_all_df.drop(['X10'], axis=1)

# One hot encoding for X8:
X8_all_df = data_all_df['X8']
X8_encoded = X8_all_df.factorize(sort=True)
encoder_X8 = OneHotEncoder()
X8_one_hot = encoder_X8.fit_transform(X8_encoded[0].reshape(-1,1))

# One hot encoding for X7:
encoder_X7 = OneHotEncoder()
X7_one_hot = encoder_X7.fit_transform(data_all_df['X7'].values.reshape(-1,1))

scaler = MinMaxScaler(feature_range=(0,1))
data_all_np = scaler.fit_transform(data_all_df[['Outcome', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X9']] )

data_all_np = np.c_[data_all_np, X7_one_hot.toarray()]
data_all_np = np.c_[data_all_np, X8_one_hot.toarray()]

data_final_for_modeling_df = pd.DataFrame(data_all_np)

data_final_for_modeling_df.columns = ['Outcome', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X9', 'X7_1', 'X7_2', 'X7_3', 'X8_A', 'X8_B', 'X8_C']

# X1 and X2 separates the data perfectly, no need for other variables for final modeling
data_final_for_modeling_df = data_final_for_modeling_df[['Outcome', 'X1', 'X2']]

########################################################################


########################################################################
# Modeling Begins
# 1. Split data into training and test set
# 2. Run ExtraTreesClassifier to estimate feature importance
# 3. Run cross validation until optimal parameters are found 
# Improvement idea: use randomized grid search for hyperparameters


# Training-Test split
data_final_for_modeling_df = data_final_for_modeling_df.sample(frac=1.0, replace=False, random_state=42)
training_set_size = math.floor((1-TEST_SET_SIZE) * data_final_for_modeling_df.shape[0])
training_set_full_df, test_set_df = data_final_for_modeling_df[:training_set_size], data_final_for_modeling_df[training_set_size:] 


# Feature Importance for the full training dataset.
X_train_full, Y_train_full = training_set_full_df.drop(['Outcome'], axis=1).values, training_set_full_df['Outcome'].values
X_test, Y_test = test_set_df.drop(['Outcome'], axis=1).values, test_set_df['Outcome'].values

model_feature_extraction = ExtraTreesClassifier()
model_feature_extraction.fit(X_train_full, Y_train_full)
Y_train_full_pred = model_feature_extraction.predict_proba(X_train_full)

roc_auc_score(Y_train_full, Y_train_full_pred[:,1])
model_feature_extraction.feature_importances_

##########################################################################
# Grid search for hyperparameters for different models

scoring = {'AUC': 'roc_auc'}

#SVC_params = [
#    {'kernel':['linear'], 'C':[1, 10, 100, 1000]},
#    {'kernel':['rbf'], 'C':[1, 10, 100, 1000], 'gamma': [0.001, 0.0001]},
#]

SVC_params = [
    {'kernel':['linear'], 'C':[1, 100]},
    {'kernel':['rbf'], 'C':[1, 100], 'gamma': [0.001]},
]


Logistic_params = [
    {'tol':[1, 10, 25], 'penalty':['l1', 'l2']}, 
]

RandomForest_params = [
    {   'n_estimators': [10,200,500],
        'bootstrap': [True],
        'max_depth': [10, 50, 100, None],
        'max_features': ['auto'],
        'min_samples_leaf': [2, 4],
        'min_samples_split': [2, 10],
    },
]

###########################################################################
# SVM 
svc = svm.SVC(gamma='scale')
GSsvc = GridSearchCV(svc, SVC_params, cv=5, scoring=scoring, refit='AUC')
GSsvc.fit(X_train_full, Y_train_full)
GSsvc.best_params_
GSsvc.best_score_
# Best params: {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
# AUC: 0.733


# Logistic Regression 
lr = LogisticRegression(solver='liblinear', multi_class='ovr')
GSlr = GridSearchCV(lr, Logistic_params, cv=5, scoring=scoring, refit='AUC')
GSlr.fit(X_train_full, Y_train_full)
GSlr.best_params_
GSlr.best_score_
# Best parameters: {'penalty': 'l1', 'tol': 1}
# AUC: 0.597

# Random Forest
rdf = RandomForestClassifier()
GSrdf = GridSearchCV(rdf, RandomForest_params, cv=5, scoring=scoring, refit='AUC')
GSrdf.fit(X_train_full, Y_train_full)
GSrdf.best_params_
GSrdf.best_score_
# Best Parameters: {'bootstrap': True, 'max_depth': 100, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
# AUC: 0.999

################################################################################


################################################################################
# RandomForest's performance with optimal hyperparameters

rdf = RandomForestClassifier(bootstrap=True, max_depth=100, max_features='auto', min_samples_leaf=2, min_samples_split=2, n_estimators=200)
rdf.fit(X_train_full, Y_train_full)
Y_train_full_pred = rdf.predict(X_train_full)
roc_auc_score(Y_train_full, Y_train_full_pred)

Y_test_pred = rdf.predict(X_test)
roc_auc_score(Y_test, Y_test_pred)
# Test AUC: 0.996

confusion_matrix(Y_test, Y_test_pred)

###########################################################################   
  