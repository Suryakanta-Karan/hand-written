import itertools
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, GridSearchCV


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

# Read digits
def read_digits():
    digits = datasets.load_digits()
    x = digits.images
    y = digits.target
    return x, y

# We will define utils here:
def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into training, development, and test subsets
def split_train_dev_test(X, y, test_size, dev_size, random_state=1):
    # First, split data into training and temporary test subsets
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Next, split the remaining data (X_train_dev, y_train_dev) into training and development subsets
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_dev, y_train_dev, test_size=dev_size, random_state=random_state
    )
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test

# Create a classifier: a support vector classifier
def train_model(X, y, model_params, model_type='svm'):
    if model_type == 'svm':
        clf = svm.SVC(**model_params)
    clf.fit(X, y)
    return clf

# Function to predict and evaluate the model
def predict_and_eval(model, X_test, y_test):
    # Predict the value of the digit on the test subset
    predicted = model.predict(X_test)

    # Quantitative sanity check
    return predicted

# Function for hyperparameter tuning
def tune_hparams(X, y, X_dev, y_dev, param_grid, model_type='svm'):
    if model_type == 'svm':
        clf = svm.SVC()
    
    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(X, y)
    
    best_hparams = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_
    
    return best_hparams, best_model, best_accuracy