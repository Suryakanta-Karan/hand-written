# Import datasets, classifiers and performance metrics
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, GridSearchCV

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
