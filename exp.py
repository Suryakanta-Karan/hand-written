from sklearn import metrics, svm
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier

from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations, tune_hparams
from joblib import dump, load

# 1. Get the dataset
X, y = read_digits()

# 2. Hyperparameter combinations for SVM
svm_gamma_list = [0.001, 0.01, 0.1, 1]
svm_C_list = [1, 10, 100, 1000]
svm_h_params = {}
svm_h_params['gamma'] = svm_gamma_list
svm_h_params['C'] = svm_C_list
svm_h_params_combinations = get_hyperparameter_combinations(svm_h_params)

# Hyperparameter combinations for Decision Tree
tree_max_depth_list = [None, 10, 20, 30]
tree_min_samples_split_list = [2, 5, 10]
tree_h_params = {}
tree_h_params['max_depth'] = tree_max_depth_list
tree_h_params['min_samples_split'] = tree_min_samples_split_list
tree_h_params_combinations = get_hyperparameter_combinations(tree_h_params)

test_sizes = [0.1, 0.2, 0.3, 0.45]
dev_sizes = [0.1, 0.2, 0.3, 0.45]

for test_size in test_sizes:
    for dev_size in dev_sizes:
        train_size = 1 - test_size - dev_size
        # 3. Data splitting -- to create train and test sets
        X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
        # 4. Data preprocessing
        X_train = preprocess_data(X_train)
        X_test = preprocess_data(X_test)
        X_dev = preprocess_data(X_dev)

        # SVM
        svm_best_hparams, svm_best_model_path, svm_best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, svm_h_params_combinations, model_type="svm")

        # Decision Tree
        tree_best_hparams, tree_best_model_path, tree_best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, tree_h_params_combinations, model_type="decision_tree")

        # Loading SVM and Decision Tree models
        svm_best_model = load(svm_best_model_path)
        tree_best_model = load(tree_best_model_path)

        svm_test_acc = predict_and_eval(svm_best_model, X_test, y_test)
        tree_test_acc = predict_and_eval(tree_best_model, X_test, y_test)

        svm_train_acc = predict_and_eval(svm_best_model, X_train, y_train)
        tree_train_acc = predict_and_eval(tree_best_model, X_train, y_train)

        svm_dev_acc = svm_best_accuracy
        tree_dev_acc = tree_best_accuracy

        print(f"test_size={test_size:.2f} dev_size={dev_size:.2f} train_size={train_size:.2f} "
              f"svm_train_acc={svm_train_acc:.2f} svm_dev_acc={svm_dev_acc:.2f} svm_test_acc={svm_test_acc:.2f} "
              f"tree_train_acc={tree_train_acc:.2f} tree_dev_acc={tree_dev_acc:.2f} tree_test_acc={tree_test_acc:.2f}")
