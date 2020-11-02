"""
The presented CNN-XGBoost model provides more precise output by integrating CNN as a trainable feature
extractor to automatically obtain features from input and XGBoost as a recognizer in the top level of the
network to produce results.
"""
import pandas as pd
from sklearn import ensemble, linear_model, naive_bayes, neighbors, svm, tree, model_selection
from xgboost import XGBClassifier
from time import time
from sklearn.metrics import accuracy_score, confusion_matrix
from config import *


MLA = {
    # Ensemble Methods
    ensemble.AdaBoostClassifier().__class__.__name__: ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier().__class__.__name__: ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier().__class__.__name__: ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier().__class__.__name__: ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier().__class__.__name__: ensemble.RandomForestClassifier(),

    # GLM
    linear_model.LogisticRegression().__class__.__name__: linear_model.LogisticRegression(),

    # Navies Bayes
    naive_bayes.BernoulliNB().__class__.__name__: naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB().__class__.__name__: naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier().__class__.__name__: neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC().__class__.__name__: svm.SVC(),

    # Trees
    tree.DecisionTreeClassifier().__class__.__name__: tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier().__class__.__name__: tree.ExtraTreeClassifier(),

    XGBClassifier().__class__.__name__: XGBClassifier()
}
# Hyperparameter Tune with GridSearchCV:
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]
grid_param = [
    [{
        # AdaBoostClassifier
        # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        'n_estimators': grid_n_estimator,  # default=50
        'learning_rate': grid_learn,  # default=1
        # 'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
        'random_state': grid_seed
    }],

    [{
        # BaggingClassifier
        # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
        'n_estimators': grid_n_estimator,  # default=10
        'max_samples': grid_ratio,  # default=1.0
        'random_state': grid_seed
    }],

    [{
        # ExtraTreesClassifier
        # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        'n_estimators': grid_n_estimator,  # default=10
        'criterion': grid_criterion,  # default=”gini”
        'max_depth': grid_max_depth,  # default=None
        'random_state': grid_seed
    }],

    [{
        # GradientBoostingClassifier
        # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        # 'loss': ['deviance', 'exponential'], #default=’deviance’
        'learning_rate': [.05],
        'n_estimators': [300],
        # 'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
        'max_depth': grid_max_depth,  # default=3
        'random_state': grid_seed
        # The best parameter for GradientBoostingClassifier is
        # {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} runtime of 264.45 seconds.
    }],

    [{
        # RandomForestClassifier
        # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'n_estimators': grid_n_estimator,  # default=10
        'criterion': grid_criterion,  # default=”gini”
        'max_depth': grid_max_depth,  # default=None
        'oob_score': [True],
        'random_state': grid_seed
        # The best parameter for RandomForestClassifier
        # {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0}
        # with a runtime of 146.35 seconds.
    }],

    [{
        # GaussianProcessClassifier
        'max_iter_predict': grid_n_estimator,  # default: 100
        'random_state': grid_seed
    }],

    [{
        # LogisticRegressionCV
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
        'fit_intercept': grid_bool,  # default: True
        # 'penalty': ['l1','l2'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # default: lbfgs
        'random_state': grid_seed
    }],

    [{
        # BernoulliNB
        # http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
        'alpha': grid_ratio,  # default: 1.0
    }],

    # GaussianNB -
    [{}],

    [{
        # KNeighborsClassifier
        # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7],  # default: 5
        'weights': ['uniform', 'distance'],  # default = ‘uniform’
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }],

    [{
        # SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
        # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [1, 2, 3, 4, 5],  # default=1.0
        'gamma': grid_ratio,  # edfault: auto
        'decision_function_shape': ['ovo', 'ovr'],  # default:ovr
        'probability': [True],
        'random_state': grid_seed
    }],

    [{
        # XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
        'learning_rate': grid_learn,  # default: .3
        'max_depth': [1, 2, 4, 6, 8, 10],  # default 2
        'n_estimators': grid_n_estimator,
        'seed': grid_seed
    }]
]


# index through MLA and save performance to table
def gridsearch_cls(X_train, y_train, X_test, y_test, MLA=MLA):
    """
    function takes train, test data set and run gridsearch over basic classifier from MLA dict
    """
    cv_split = model_selection.ShuffleSplit(n_splits=5, test_size=.2, train_size=.8, random_state=39)

    # create table to compare MLA metrics
    MLA_columns = ['MLA Name', 'MLA Test Accuracy Mean', 'MLA Time', 'MLA pred']
    MLA_compare = pd.DataFrame(columns=MLA_columns)

    row_index = 0
    MLA_predict = y_test.copy()
    for alg in MLA.values():
        tic = time()
        # set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

        # score model with cross validation:
        # alg = make_pipeline(preprocessing.PolynomialFeatures(), preprocessing.MinMaxScaler(),alg)
        cv_results = model_selection.cross_validate(alg, X_train, y_train, cv=cv_split)

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()

        # save MLA predictions
        alg.fit(X_train, y_train)
        MLA_compare.loc[row_index, 'MLA pred'] = alg.predict(X_test)

        row_index += 1
        toc = time()
        # print(f'Time run {MLA_name}:\t{tic - toc}')
    name_best_model = MLA_compare['MLA Name'].values[0]
    MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)

    # plot confusion matrix and acc score
    cm = confusion_matrix(y_test, MLA_compare['MLA pred'].values[0]) / len(y_test)
    accuracy = accuracy_score(y_test, MLA_compare['MLA pred'].values[0])
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(cm, annot=True, cmap='Wistia')
    plt.title(f'{name_best_model}\n\nAccuracy:\t{accuracy * 100:.2f}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # ax.set_xticklabels(labels)
    # ax.set_yticklabels(labels)
    plt.show()
    return MLA_compare


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
"""
model = Sequential()
# input: 28x28 images with 1 channels -> (28, 28, 1) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=5, activation='relu', padding='same'))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(128, kernel_size=5, activation='relu', padding='same'))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=5, activation='relu', padding='same'))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(256, kernel_size=5, activation='relu', padding='same'))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu', name='my_dense'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
"""
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd)


model_file = '/Users/tal/Dropbox/Projects/vgg_face_Eyeglasses.h5'

