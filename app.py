# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 13:22:15 2021

@author: Sumon Dey
"""

# Imported Libraries

import base64
import streamlit as st
import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
import timeit

# Classifier Libraries

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

# Other Libraries

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import norm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import warnings

warnings.filterwarnings("ignore")

st.title('Credit Card Fraud Detection!')

# set background image

main_bg = "1.jpg"
main_bg_ext = "jpg"

side_bg = "1.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})

    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})

    }}
    </style>
    """,
    unsafe_allow_html=True
)

df = pd.read_csv('creditcard.csv')

# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()


@st.cache(allow_output_mutation=True)
def scale_dataset(acc_df):
    acc_df['scaled_amount'] = rob_scaler.fit_transform(acc_df['Amount'].values.reshape(-1, 1))
    acc_df['scaled_time'] = rob_scaler.fit_transform(acc_df['Time'].values.reshape(-1, 1))

    acc_df.drop(['Time', 'Amount'], axis=1, inplace=True)

    scaled_amount = acc_df['scaled_amount']
    scaled_time = acc_df['scaled_time']

    acc_df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    acc_df.insert(0, 'scaled_amount', scaled_amount)
    acc_df.insert(1, 'scaled_time', scaled_time)
    return acc_df
    # Amount and Time are Scaled!

    # Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

    # Lets shuffle the data before creating the subsamples


@st.cache(allow_output_mutation=True)
def normalize_dataset(acc_df):
    acc_df = acc_df.sample(frac=1)
    # amount of fraud classes 492 rows.
    fraud_df = acc_df.loc[acc_df['Class'] == 1]
    non_fraud_df = acc_df.loc[acc_df['Class'] == 0][:492]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    # Shuffle dataframe rows
    new_df = normal_distributed_df.sample(frac=1, random_state=42)
    return new_df


acc_df = df.copy(deep=True)
acc_df = scale_dataset(acc_df)
new_df = normalize_dataset(acc_df)

# Print shape and description of the data
if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(df.head(100))
    st.write('Shape of the dataframe: ', df.shape)
    st.write('Data decription: \n', df.describe())
    st.write('Number of null values: ', df.isnull().sum().max())

# Print valid and fraud transactions
fraud = df[df.Class == 1]
valid = df[df.Class == 0]
colors = ["#0101DF", "#DF0101"]
outlier_percentage = round(df['Class'].value_counts()[1] / len(df) * 100, 2)
nonoutlier_percentage = round(df['Class'].value_counts()[0] / len(df) * 100, 2)
if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write('Non-Fraudulent transactions are: %.3f%%' %
             (nonoutlier_percentage))
    st.write('Fraudulent transactions are: %.3f%%' % outlier_percentage)
    st.write('Fraud Cases: ', len(fraud))
    st.write('Valid Cases: ', len(valid))
    sns.countplot('Class', data=df, palette=colors)
    plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
    st.pyplot(plt)

if st.sidebar.checkbox('Show Distribution of Transaction Amount and Time'):
    plt.clf()
    # Print distribution
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))

    amount_val = df['Amount'].values
    time_val = df['Time'].values

    sns.distplot(amount_val, ax=ax[0], color='r')
    ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
    ax[0].set_xlim([min(amount_val), max(amount_val)])

    sns.distplot(time_val, ax=ax[1], color='b')
    ax[1].set_title('Distribution of Transaction Time', fontsize=14)
    ax[1].set_xlim([min(time_val), max(time_val)])
    st.pyplot(plt)

if st.sidebar.checkbox('Equal Distribution after Scaling'):
    st.write('Given DataSet: \n', df.head(100))
    st.write('After Scaling: \n', new_df.head(100))

    plt.clf()
    st.write('Distribution of the Classes in the subsample dataset')
    st.write(new_df['Class'].value_counts() / len(new_df))
    sns.countplot('Class', data=new_df, palette=colors)
    plt.title('Equally Distributed Classes', fontsize=14)
    st.pyplot(plt)

# Obtaining X (features) and y (labels)
X = new_df.drop(['Class'], axis=1)
y = new_df.Class

# Split the data into training and testing sets
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=size, random_state=42)

# Print shape of train and test sets
if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
    st.write(acc_df.head(5))
    st.write('X_train: ', X_train.shape)
    st.write('y_train: ', y_train.shape)
    st.write('X_test: ', X_test.shape)
    st.write('y_test: ', y_test.shape)

# Let's implement simple classifiers

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier()
}

features = X_train.columns.tolist()


# Feature selection through feature importance
@st.cache
def feature_sort(model, X_train, y_train):
    # feature selection
    mod = model
    # fit the model
    mod.fit(X_train, y_train)
    # get importance
    imp = mod.feature_importances_
    return imp


# Classifiers for feature importance
clf = ['LogisiticRegression', 'Decision Tree', 'Random Forest']
mod_feature = st.sidebar.selectbox('Which model for feature importance?', clf)

start_time = timeit.default_timer()
if mod_feature == 'Decision Tree':
    model = classifiers['DecisionTreeClassifier']
    importance = feature_sort(model, X_train, y_train)
elif mod_feature == 'Random Forest':
    model = classifiers['RandomForestClassifier']
    importance = feature_sort(model, X_train, y_train)
elif mod_feature == 'LogisiticRegression':
    model = classifiers['LogisiticRegression']
    importance = model.fit(X_train, y_train).coef_.reshape(30, )

elapsed = timeit.default_timer() - start_time
st.write('Execution Time for feature selection: %.2f minutes' % (elapsed / 60))

# Plot of feature importance
if st.sidebar.checkbox('Show plot of feature importance'):
    plt.clf()
    plt.bar([x for x in range(len(importance))], importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature (Variable Number)')
    plt.ylabel('Importance')
    st.pyplot(plt)

feature_imp = list(zip(features, importance))
feature_sort = sorted(feature_imp, key=lambda x: x[1])

n_top_features = st.sidebar.slider(
    'Number of top features', min_value=5, max_value=20)

top_features = list(list(zip(*feature_sort[-n_top_features:]))[0])

if st.sidebar.checkbox('Show selected top features'):
    st.write('Top %d features in order of importance are: %s' %
             (n_top_features, top_features[::-1]))

X_train_sfs = X_train[top_features]
X_test_sfs = X_test[top_features]

X_train_sfs_scaled = X_train_sfs
X_test_sfs_scaled = X_test_sfs

# ------------playing with different models start here------------

# Use GridSearchCV to find the best parameters.

# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)

# Logistic best estimator
log_reg = grid_log_reg.best_estimator_

# KNeighbors Classifier
knears_params = {"n_neighbors": list(range(2, 5, 1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)

# KNears best estimator
knears_neighbors = grid_knears.best_estimator_

# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)

# SVC best estimator
svc = grid_svc.best_estimator_

# Decision Tree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)),
               "min_samples_leaf": list(range(5, 7, 1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)

# tree best estimator
tree_clf = grid_tree.best_estimator_

# Random Forest Classifier
forest_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)),
                 "min_samples_leaf": list(range(5, 7, 1))}
grid_forest = GridSearchCV(RandomForestClassifier(), tree_params)
grid_forest.fit(X_train, y_train)

# forest best estimator
forest_clf = grid_forest.best_estimator_

# Overfitting Case

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')

forest_score = cross_val_score(forest_clf, X_train, y_train, cv=5)
print('RandomForest Classifier Cross Validation Score', round(forest_score.mean() * 100, 2).astype(str) + '%')


# training score vs cross validation score
@st.cache
def plot_learning_curve_oneByone(estimator1, estimator2, estimator3, estimator4, estimator5, model, X, y, ylim=None,
                                 cv=None,
                                 n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    f, (ax) = plt.subplots(1, 1, figsize=(10, 4), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    # First Estimator
    if model == 'log':
        train_sizes, train_scores, test_scores = learning_curve(
            estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax.set_title("Logistic Regression Learning Curve", fontsize=14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc="best")

    # Second Estimator 
    if model == 'knn':
        train_sizes, train_scores, test_scores = learning_curve(
            estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax.set_title("Knears Neighbors Learning Curve", fontsize=14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc="best")

    # Third Estimator
    if model == 'svm':
        train_sizes, train_scores, test_scores = learning_curve(
            estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc="best")

    # Fourth Estimator
    if model == 'tree':
        train_sizes, train_scores, test_scores = learning_curve(
            estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc="best")

    # Fifth Estimator
    if model == 'forest':
        train_sizes, train_scores, test_scores = learning_curve(
            estimator5, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="#ff9124")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                label="Cross-validation score")
        ax.set_title("Random Forest Classifier \n Learning Curve", fontsize=14)
        ax.set_xlabel('Training size (m)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc="best")
    return plt


clf = ['LogisiticRegression', 'KNN', 'SVM', 'Decision Tree', 'Random Forest']
mod_tvscore = st.sidebar.selectbox('Which model for training score vs cross validation score?', clf)

# Plot of feature importance
if st.sidebar.checkbox('Show plot of training score vs cross validation score'):
    start_time = timeit.default_timer()
    if mod_tvscore == 'Decision Tree':
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
        st.pyplot(
            plot_learning_curve_oneByone(log_reg, knears_neighbors, svc, tree_clf, forest_clf, "tree", X_train, y_train,
                                         (0.87, 1.01), cv=cv, n_jobs=1))
    elif mod_tvscore == 'Random Forest':
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
        st.pyplot(plot_learning_curve_oneByone(log_reg, knears_neighbors, svc, tree_clf, forest_clf, "forest", X_train,
                                               y_train, (0.87, 1.01), cv=cv, n_jobs=1))
    elif mod_tvscore == 'LogisiticRegression':
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
        st.pyplot(
            plot_learning_curve_oneByone(log_reg, knears_neighbors, svc, tree_clf, forest_clf, "log", X_train, y_train,
                                         (0.87, 1.01), cv=cv, n_jobs=1))
    elif mod_tvscore == 'KNN':
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
        st.pyplot(
            plot_learning_curve_oneByone(log_reg, knears_neighbors, svc, tree_clf, forest_clf, "knn", X_train, y_train,
                                         (0.87, 1.01), cv=cv, n_jobs=1))
    elif mod_tvscore == 'SVM':
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
        st.pyplot(
            plot_learning_curve_oneByone(log_reg, knears_neighbors, svc, tree_clf, forest_clf, "svm", X_train, y_train,
                                         (0.87, 1.01), cv=cv, n_jobs=1))

    elapsed = timeit.default_timer() - start_time
    st.write('Execution Time for training score vs cross validation score: %.2f minutes' % (elapsed / 60))

# Create a DataFrame with all the scores and the classifiers names.

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5, method="decision_function")

knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)

svc_pred = cross_val_predict(svc, X_train, y_train, cv=5, method="decision_function")

tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)

forest_pred = cross_val_predict(forest_clf, X_train, y_train, cv=5)

if st.sidebar.checkbox('Show Classification Reports'):
    y_pred_log_reg = log_reg.predict(X_test)
    y_pred_knear = knears_neighbors.predict(X_test)
    y_pred_svc = svc.predict(X_test)
    y_pred_tree = tree_clf.predict(X_test)
    y_pred_forest = forest_clf.predict(X_test)
    start_time = timeit.default_timer()
    if mod_tvscore == 'LogisiticRegression':
        st.write('Logistic Regression:')
        st.write(classification_report(y_test, y_pred_log_reg))
    elif mod_tvscore == 'KNN':
        st.write('KNears Neighbors:')
        st.write(classification_report(y_test, y_pred_knear))
    elif mod_tvscore == 'SVM':
        st.write('Support Vector Classifier:')
        st.write(classification_report(y_test, y_pred_svc))

    elif mod_tvscore == 'Decision Tree':
        st.write('DecisionTree Classifier:')
        st.write(classification_report(y_test, y_pred_tree))
    elif mod_tvscore == 'Random Forest':
        st.write('RandomForest Classifier:')
        st.write(classification_report(y_test, y_pred_forest))
    elapsed = timeit.default_timer() - start_time
    st.write('Execution Time for Classification Reports: %.2f minutes' % (elapsed / 60))

if st.sidebar.checkbox('Show ROC curve of TOP 5 Classifiers '):
    log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
    knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
    svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
    tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)
    forest_fpr, forest_tpr, forest_threshold = roc_curve(y_train, forest_pred)


    @st.cache
    def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr,
                                 forest_fpr, forest_tpr):
        plt.figure(figsize=(20, 30))
        plt.title('ROC Curve \n Top 5 Classifiers', fontsize=18)
        plt.plot(log_fpr, log_tpr,
                 label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
        plt.plot(knear_fpr, knear_tpr,
                 label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
        plt.plot(svc_fpr, svc_tpr,
                 label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
        plt.plot(tree_fpr, tree_tpr,
                 label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
        plt.plot(forest_fpr, forest_tpr,
                 label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_train, forest_pred)))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([-0.01, 1, 0, 1])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                     arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                     )
        plt.legend()


    graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr, forest_fpr,
                             forest_tpr)
    st.pyplot(plt)

if st.sidebar.checkbox('Show Confussion Matrixes of TOP 5 Classifiers '):
    y_pred_log_reg = log_reg.predict(X_test)
    y_pred_knear = knears_neighbors.predict(X_test)
    y_pred_svc = svc.predict(X_test)
    y_pred_tree = tree_clf.predict(X_test)
    y_pred_forest = forest_clf.predict(X_test)

    log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
    kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
    svc_cf = confusion_matrix(y_test, y_pred_svc)
    tree_cf = confusion_matrix(y_test, y_pred_tree)
    forest_cf = confusion_matrix(y_test, y_pred_forest)

    fig, ax = plt.subplots(3, 2, figsize=(22, 12))

    sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.Blues)
    ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
    ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, cmap=plt.cm.Blues)
    ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix", fontsize=14)
    ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(svc_cf, ax=ax[1][0], annot=True, cmap=plt.cm.Blues)
    ax[1][0].set_title("Suppor Vector Classifier \n Confusion Matrix", fontsize=14)
    ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(tree_cf, ax=ax[1][1], annot=True, cmap=plt.cm.Blues)
    ax[1][1].set_title("DecisionTree Classifier \n Confusion Matrix", fontsize=14)
    ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(forest_cf, ax=ax[2][0], annot=True, cmap=plt.cm.Blues)
    ax[2][0].set_title("RandomForest Classifier \n Confusion Matrix", fontsize=14)
    ax[2][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[2][0].set_yticklabels(['', ''], fontsize=14, rotation=360)
    st.pyplot(plt)