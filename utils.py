import itertools
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics

def prepare_data_splits(data, labels, test_fraction, dev_fraction):
    X_train_dev, X_test, Y_train_dev, Y_test = train_test_split(
        data, labels, test_size=test_fraction, shuffle=True
    )
    X_train, X_dev, Y_train, Y_dev = train_test_split(
        X_train_dev, Y_train_dev, test_size=dev_fraction, shuffle=True
    )
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

def hyperparameter_tuning(X_train, Y_train, X_dev, Y_dev, param_combinations):
    best_accuracy = -1
    best_train_accuracy = -1
    best_model = None
    best_gamma = None
    best_C = None

    for gamma, C in param_combinations:
        model = svm.SVC(C=C, gamma=gamma)
        model.fit(X_train, Y_train)

        predicted_dev = model.predict(X_dev)
        current_accuracy = metrics.accuracy_score(y_pred=predicted_dev, y_true=Y_dev)

        predicted_train = model.predict(X_train)
        train_accuracy = metrics.accuracy_score(y_pred=predicted_train, y_true=Y_train)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_train_accuracy = train_accuracy
            best_model = model
            best_gamma = gamma
            best_C = C

    return best_model, best_gamma, best_C, best_accuracy, best_train_accuracy