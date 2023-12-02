from sklearn import datasets, preprocessing
import itertools
import joblib
from utils import hyperparameter_tuning, prepare_data_splits

def main():
    # Load the digits dataset
    digits_data = datasets.load_digits()

    # Flatten the images
    n_samples = len(digits_data.images)
    flattened_data = digits_data.images.reshape((n_samples, -1))
    X_data = flattened_data
    y_data = digits_data.target

    # Normalize the data
    X_data_normalized = preprocessing.normalize(X_data, norm='l2')

    # Define parameter ranges
    gamma_values = [0.001, 0.01, 0.1, 1, 100]
    C_values = [0.1, 1, 2, 5, 10]
    all_param_combinations = list(itertools.product(gamma_values, C_values))

    # Define test and dev set sizes
    test_sizes = [0.1, 0.2, 0.3]
    dev_sizes = [0.1, 0.2, 0.3]
    size_combinations = list(itertools.product(test_sizes, dev_sizes))

    for test_frac, dev_frac in size_combinations:
        # Split the normalized data into train, dev, and test sets
        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = prepare_data_splits(X_data_normalized, y_data, test_frac, dev_frac)

        # Tune hyperparameters
        trained_model, _, _, _, _ = hyperparameter_tuning(X_train, Y_train, X_dev, Y_dev, all_param_combinations)

        # Print model training logs to the console
        print("Model training complete for test_frac={}, dev_frac={}".format(test_frac, dev_frac))

    # Save the trained model (outside the loop)
    model_path = '/home/suryakantak/hand-written/API/best_model.pkl'
    joblib.dump(trained_model, model_path)

if __name__ == "__main__":
    main()
