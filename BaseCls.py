"""
The presented CNN-XGBoost model provides more precise output by integrating CNN as a trainable feature
extractor to automatically obtain features from input and XGBoost as a recognizer in the top level of the
network to produce results.
"""
import pandas as pd
from sklearn import ensemble, linear_model, naive_bayes, neighbors, svm, tree, model_selection
from xgboost import XGBClassifier
from time import time

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

cv_split = model_selection.ShuffleSplit(n_splits=5, test_size=.2, train_size=.8, random_state=39)

# create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Test Accuracy Mean', 'MLA Time']
MLA_compare = pd.DataFrame(columns=MLA_columns)

# create table to compare MLA predictions


# index through MLA and save performance to table
def run_ensemble(X_train, y_train, X_test, y_test):
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
        MLA_predict[MLA_name] = alg.predict(X_test)

        row_index += 1
        toc = time()
        print(f'Time run {MLA_name}:\t{tic - toc}')
    MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)
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

