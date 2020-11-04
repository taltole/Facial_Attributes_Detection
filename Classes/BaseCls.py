"""
The presented CNN-XGBoost model provides more precise output by integrating CNN as a trainable feature
extractor to automatically obtain features from input and XGBoost as a recognizer in the top level of the
network to produce results.
"""
from sklearn import ensemble, linear_model, naive_bayes, neighbors, svm, tree, model_selection, metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from time import time
from sklearn.metrics import accuracy_score, confusion_matrix
from config import *

# Hyper_parameter Tune with GridSearchCV:
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.01, .1, .5, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [39]

# Classifiers Models
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
    # naive_bayes.BernoulliNB().__class__.__name__: naive_bayes.BernoulliNB(),
    # naive_bayes.GaussianNB().__class__.__name__: naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier().__class__.__name__: neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC().__class__.__name__: svm.SVC(),

    # XGB
    XGBClassifier().__class__.__name__: XGBClassifier()
}

# Classifiers Models HyperParameters
grid_param = [
    [{
        # AdaBoostClassifier
        'n_estimators': grid_n_estimator,  # default=50
        'learning_rate': grid_learn,       # default=1
        'random_state': grid_seed,
        'n_jobs': [-1]

        # 'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
    }],

    [{
        # BaggingClassifier
        'n_estimators': grid_n_estimator,  # default=10
        'max_samples': grid_ratio,         # default=1.0
        'random_state': grid_seed,
        'n_jobs': [-1]

    }],

    [{
        # ExtraTreesClassifier
        'n_estimators': grid_n_estimator,  # default=10
        'criterion': grid_criterion,       # default=”gini”
        'max_depth': grid_max_depth,       # default=None
        'random_state': grid_seed,
        'n_jobs': [-1]

    }],

    [{
        # GradientBoostingClassifier
        # 'loss': ['deviance', 'exponential'],         # default=’deviance’
        # 'criterion': ['friedman_mse', 'mse', 'mae'], # default=”friedman_mse”
        'learning_rate': grid_learn,                   # default=0.1
        'n_estimators': grid_n_estimator,              # default=100
        'max_depth': grid_max_depth,                   # default=3
        'random_state': grid_seed
        # {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0}
    }],

    [{
        # RandomForestClassifier
        'n_estimators': [100, 300],  # default=10
        'criterion': grid_criterion,  # default=”gini”
        'max_depth': grid_max_depth,  # default=None
        'oob_score': [True],  # default=False
        'random_state': grid_seed,
        'n_jobs': [-1]

        # {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0}
        # {'criterion': 'gini', 'max_depth': 4, 'n_estimators': 100, 'oob_score': True, 'random_state': 39}

    }],

    [{
        # LogisticRegressionCV
        'fit_intercept': grid_bool,  # default: True
        # 'penalty': ['l1','l2'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # default: lbfgs
        'random_state': grid_seed,
        'n_jobs': [-1]

    }],

    # [{
    #     # BernoulliNB
    #     # 'alpha': grid_ratio,
    #     'prior': grid_bool,
    #     'random_state': grid_seed,
    # }],

    [{
        # KNeighborsClassifier
        'n_neighbors': [3, 4, 5, 6, 7],  # default: 5
        'weights': ['uniform', 'distance'],  # default = ‘uniform’
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'n_jobs': [-1]

    }],

    [{
        # SVC
        # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [1, 2, 3, 4, 5],  # default: 1.0
        'gamma': grid_ratio,   # default: auto
        'decision_function_shape': ['ovo', 'ovr'],  # default:ovr
        'probability': [True],
        'random_state': grid_seed

        # The best parameter for SVC() is
        # {'C': 1, 'decision_function_shape': 'ovo', 'gamma': 0.1, 'probability': True, 'random_state': 39}
        # {'C': 1, 'decision_function_shape': 'ovo', 'gamma': 0.1, 'probability': True, 'random_state': 39}
        # {'C': 1, 'decision_function_shape': 'ovo', 'gamma': 0.01, 'probability': True, 'random_state': 39}
    }],

    [{
        # XGBClassifier
        'learning_rate': grid_learn,  # default: .3
        'max_depth': [1, 2, 4, 6, 8, 10],  # default 2
        'n_estimators': grid_n_estimator,
        'seed': grid_seed,
        'nthread': [4],  # when use hyperthread, xgboost may become slower
        'objective': ['binary:logistic'],
        'min_child_weight': [11],
        # 'silent': [0],
        'subsample': [0.8],
        'colsample_bytree': [0.7],
        'missing': [-999],
        'n_jobs': [-1]

    }]
]


# index through MLA and save performance to table
def gridsearch_cls(X_train, y_train, X_test, y_test, model):
    """
    function takes train, test data set and run gridsearch over basic classifier from MLA dict
    """
    cv_split = model_selection.ShuffleSplit(n_splits=5, test_size=.2, train_size=.8, random_state=39)

    # create table to compare MLA metrics
    MLA_columns = ['MLA Name', 'MLA Test Accuracy Mean', 'MLA Time', 'MLA pred']
    MLA_compare = pd.DataFrame(columns=MLA_columns)

    row_index = 0
    MLA_predict = y_test.copy()
    if model == MLA:
        algo = model.values()
    else:
        algo = model['param']
    for alg in algo:
        # set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

        # score model with cross validation:
        cv_results = model_selection.cross_validate(alg, X_train, y_train, cv=cv_split, scoring='roc_auc')
        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()

        # save MLA predictions
        alg.fit(X_train, y_train)
        MLA_compare.loc[row_index, 'MLA pred'] = alg.predict(X_test)
        row_index += 1

    MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)

    return MLA_compare


def gridsearch_params(MLA_compare, X_train, y_train):
    cv_split = model_selection.ShuffleSplit(n_splits=5, test_size=.2, train_size=.8, random_state=39)
    best_classifiers = MLA_compare['MLA Name'].values[:3]
    best_cls_ind = MLA_compare['MLA Name'].index[:3]
    best_params_dict = {'cls': best_classifiers, 'param': [], 'score': []}
    start_total = time()

    for ind, clf in zip(best_cls_ind, best_classifiers):
        start = time()
        param = grid_param[ind]
        best_search = model_selection.GridSearchCV(estimator=MLA[clf], param_grid=param, cv=cv_split, scoring='roc_auc')
        best_search.fit(X_train, y_train)
        run = time() - start
        best_param = best_search.best_params_
        best_params_dict['param'].append(MLA[clf].set_params(**best_param))
        best_params_dict['score'].append(best_search.best_score_)
        print(f'{clf}\nBest Parameters: {best_param}\nRuntime: {run:.2f} seconds.')
        print('-' * 10)

    run_total = time() - start_total
    print(f'Total optimization time was {(run_total / 60):.2f} minutes.')
    return best_params_dict


def plot_best_model(df):
    plt.figure(figsize=(16, 7))
    ax = plt.subplot(1, 1, 1)
    sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=df, color='m', ax=ax)
    plt.title('Machine Learning Algorithm Accuracy Score \n')
    plt.xlabel('Accuracy Score (%)')
    plt.xticks(np.arange(0, 1, 0.1))
    plt.ylabel('Algorithm')
    plt.tight_layout()
    plt.show()


def find_best_threshold(thresholds, fpr, tpr):
    """
    find the best threshold from the roc curve. by finding the threshold for the point which
    is closest to (fpr=0,tpr=1)
    """
    fpr_tpr = pd.DataFrame({'thresholds': thresholds, 'fpr': fpr, 'tpr': tpr})
    fpr_tpr['dist'] = (fpr_tpr['fpr']) ** 2 + (fpr_tpr['tpr'] - 1) ** 2
    return fpr_tpr.ix[fpr_tpr.dist.idxmin(), 'thresholds']


def get_model_results(model, X_train, X_test, y_train, y_test):
    probabilities = model.predict_proba(np.array(X_test))[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probabilities)
    threshold = find_best_threshold(thresholds, fpr, tpr)
    predictions = probabilities > threshold
    plt.figure()
    plt.plot(fpr, tpr, label='test')
    roc_auc = metrics.roc_auc_score(y_test, probabilities)
    probabilities = model.predict_proba(np.array(X_train))[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_train, probabilities)
    plt.plot(fpr, tpr, label='train')
    plt.plot([0, 1], [0, 1], 'r--', label='random guess')
    plt.title("area under the ROC curve = {:.3f}".format(roc_auc), fontsize=18)
    print(metrics.classification_report(y_test, predictions))
    plt.legend()


def check_xgb(X_train, y_train):
    xgb_model = XGBClassifier()

    # brute force scan for all parameters, here are the tricks
    # usually max_depth is 6,7,8
    # learning rate is around 0.05, but small changes may make big diff
    # tuning min_child_weight subsample colsample_bytree can have
    # much fun of fighting against overfit
    # n_estimators is how many round of boosting
    # finally, ensemble xgboost with multiple seeds may reduce variance
    parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
                  'objective': ['binary:logistic'],
                  'learning_rate': [0.05],  # so called `eta` value
                  'max_depth': [6],
                  'min_child_weight': [11],
                  'silent': [0],
                  'subsample': [0.8],
                  'colsample_bytree': [0.7],
                  'n_estimators': [5],  # number of trees, change it to 1000 for better results
                  'missing': [-999],
                  'seed': [1337]}

    clf = GridSearchCV(xgb_model,
                       parameters,
                       n_jobs=5,
                       cv=StratifiedKFold(5, shuffle=True),
                       scoring='roc_auc',
                       verbose=2, refit=True)

    clf.fit(X_train, y_train)

    # trust your CV!
    best_parameters, score = max(clf.get_params(), key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters):
        print("%s: %r" % (param_name, score))

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
# model.compile(loss='categorical_crossentropy', optimizer=sgd
