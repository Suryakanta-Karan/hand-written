from sklearn import  svm, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
# Put the Utils here 
def read_digits():
    digits = datasets.load_digits()
    X= digits.images
    y= digits.target
    return X, y

def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets
def split_data(x,y,test_size,random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# Train the model of choice with model parameter 
def train_model(x,y, model_params, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC;    
    model = clf(**model_params)
    #Train the model
    model.fit(x,y)
    return model

#Assignment2 - Added below functions
#Added spaces to capture pull request for Assignemnt2 documentation
def split_train_dev_test(X, y, test_size=0.2, dev_size=0.25, random_state=1):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=dev_size / (dev_size + test_size), random_state=random_state)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def predict_and_eval(model, X_test, y_test):
    # Predict the value of the digit on the test subsetconda activate 
    predicted = model.predict(X_test)
    

    # Quantitative sanity check
    return metrics.accuracy_score(y_test, predicted)

def tune_hyperparameters(X_train, y_train, X_dev, y_dev, hyperparameter_combinations):
    best_hyperparameters = None
    best_model = None
    best_dev_accuracy = 0.0
    
    # Iterate through each set of hyperparameters in the list
    for hyperparameters in hyperparameter_combinations:
        # Train a model with the current set of hyperparameters
        current_model = train_model(X_train, y_train, hyperparameters)

        # Evaluate the model's accuracy on the training dataset
        train_accuracy = predict_and_eval(current_model, X_train, y_train)  
        
        # Evaluate the model on the development dataset
        dev_accuracy = predict_and_eval(current_model, X_dev, y_dev)  
        
        # Check if this model's accuracy is better than the current best
        if dev_accuracy > best_dev_accuracy:
            best_hyperparameters = hyperparameters
            best_model = current_model
            best_dev_accuracy = dev_accuracy
    
    return train_accuracy, best_hyperparameters, best_model, best_dev_accuracy

def create_hparam_combo(gamma_range, C_range):
    return [{'gamma': gamma, 'C': C} for gamma in gamma_range for C in C_range]