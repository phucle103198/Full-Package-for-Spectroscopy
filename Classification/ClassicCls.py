from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sklearn.svm as svm
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
import pandas  as pd

def ANN(X_train, X_test, y_train, y_test, StandScaler=None):

    if StandScaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf =  MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=False, epsilon=1e-08,
                  hidden_layer_sizes=(10, 8), learning_rate='constant',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                  solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                  warm_start=False)

    clf.fit(X_train,y_train.ravel())
    predict_results=clf.predict(X_test)
    acc = accuracy_score(predict_results, y_test.ravel())

    return acc

def SVM(X_train, X_test, y_train, y_test):

    clf = svm.SVC(C=1, gamma=1e-3)
    clf.fit(X_train, y_train)

    predict_results = clf.predict(X_test)
    acc = accuracy_score(predict_results, y_test.ravel())

    return acc

def PLS_DA(X_train, X_test, y_train, y_test):

    y_train = pd.get_dummies(y_train)
    model = PLSRegression(n_components=228)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.array([np.argmax(i) for i in y_pred])
    acc = accuracy_score(y_test, y_pred)

    return acc

def RF(X_train, X_test, y_train, y_test):

    RF = RandomForestClassifier(n_estimators=15,max_depth=3,min_samples_split=3,min_samples_leaf=3)
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc
