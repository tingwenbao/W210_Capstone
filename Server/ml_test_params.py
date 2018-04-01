from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    IsolationForest)
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import (
    #HuberClassifier,
    PassiveAggressiveClassifier,
    #TheilSenClassifier,
    #RANSACClassifier
    )
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

Huber_params = {
    'epsilon': [1.1, 1.35, 2.0],
    'max_iter': [100, 300]}
PassiveAggressive_params = {
    'C': [10.0, 1.0, 0.01],
    #'max_iter ': [100],
    'verbose': [True]}
RANSAC_params = {
    'min_samples': [0.1, 0.5, 1.0]}
LinearSVC_params = {
    'C': [0.01, 0.5, 1, 10, 100]}
Multinomial_params = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]}
Bernoulli_params = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]}
KNeighbors_params = {
    'n_neighbors': [30, 10, 2],
    'weights': ['uniform', 'distance']}
DecisionTree_params = {
    'min_samples_leaf': [5, 2, 1],
    'max_features': ['auto', 0.2, 0.02]}
Bagging_params = {
    'n_estimators': [200, 100],
    'max_samples': [1.0, 0.25],
    #'oob_score': [False],
    'max_features': [0.65, 0.25]}
RandomForest_params = {
    'n_estimators': [100, 50, 5],
    'min_samples_leaf': [1, 2, 5],
    'oob_score': [False],
    'max_features': [0.2, 'auto', 0.02]}
AdaBoost_params = {
    'n_estimators': [10, 100, 500]}
IsolationForest_params = {
    'n_estimators': [500, 100],
    'contamination': [0.01, 0.1]}

est_dicts = [
    #{
    #    'name': 'HuberClassifier',
    #    'params': Huber_params,
    #    'callable': HuberClassifier()},
    {
        'name': 'PassiveAggressiveClassifier',
        'params': PassiveAggressive_params,
        'callable': PassiveAggressiveClassifier()},
    #{
    #    'name': 'RANSACClassifier',
    #    'params': RANSAC_params,
    #    'callable': RANSACClassifier()},
    #{
    #    'name': 'BaggingClassifier',
    #    'params': Bagging_params,
    #    'callable': BaggingClassifier()},
    {
        'name': 'IsolationForest',
        'params': IsolationForest_params,
        'callable': IsolationForest()},
    {
        'name': 'RandomForestClassifier',
        'params': RandomForest_params,
        'callable': RandomForestClassifier()},
    #{
    #    'name': 'AdaBoostClassifier',
    #    'params': AdaBoost_params,
    #    'callable': AdaBoostClassifier()},
    {
        'name': 'LinearSVC',
        'params': LinearSVC_params,
        'callable': LinearSVC()},
    {
        'name': 'MultinomialNB',
        'params': Multinomial_params,
        'callable': MultinomialNB()},
    {
        'name': 'BernoulliNB',
        'params': Bernoulli_params,
        'callable': BernoulliNB()},
    #{
    #    'name': 'KNeighborsClassifier',
    #    'params': KNeighbors_params,
    #    'callable': KNeighborsClassifier()},
    {
        'name': 'DecisionTreeClassifier',
        'params': DecisionTree_params,
        'callable': DecisionTreeClassifier()}
    ]
