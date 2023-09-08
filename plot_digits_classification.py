# This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import metrics
from utils import preprocess_data, train_model, split_train_dev_test, read_digits, predict_and_eval, tune_hparams

# The digits dataset consists of 8x8 pixel images of digits. The images attribute of the dataset stores 8x8 arrays of grayscale values for each image. We will use these arrays to visualize the first 4 images. The target attribute of the dataset stores the digit each image represents and this is included in the title of the 4 plots below.
# Note: if we were working from image files (e.g., ‘png’ files), we would load them using matplotlib.pyplot.imread.

# 1. Data Loading
x, y = read_digits()

# Define hyperparameter grid for tuning
param_grid = {'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}

# 2. Vary test_size and dev_size
test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

for test_size in test_sizes:
    for dev_size in dev_sizes:
        train_size = 1 - test_size - dev_size
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(x, y, test_size=test_size, dev_size=dev_size)

        # 3. Data Preprocessing
        X_train = preprocess_data(X_train)
        X_dev = preprocess_data(X_dev)
        X_test = preprocess_data(X_test)

        # 4. Hyperparameter Tuning
        best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, param_grid, model_type='svm')

        # 5. Train the data with best_hparams
        model = train_model(X_train, y_train, best_hparams, model_type='svm')

        # 6. Model Prediction
        predicted = predict_and_eval(model, X_test, y_test)

        # 7. Visualize Predictions
        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, prediction in zip(axes, X_test, predicted):
            ax.set_axis_off()
            image = image.reshape(8, 8)
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title(f"Prediction: {prediction}")

        # 8. Model Evaluation
        print(f"Test size={test_size} dev_size={dev_size} train_size={train_size} train_acc={best_accuracy:.2f} dev_acc={model.score(X_dev, y_dev):.2f} test_acc={model.score(X_test, y_test):.2f}")
        print(f"Best hyperparameters: {best_hparams}")
        print("-" * 50)

# Note: In a real-world scenario, you might want to save the best model for future use.
