import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets, metrics
from utils import preprocess_data
import os

os.makedirs("q2_models", exist_ok=True)

def load_data():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y 

def save_trained_model(model, filename):
    joblib.dump(model, filename)

# Read and preprocess the dataset
X, y = load_data()
X = preprocess_data(X)

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

print(f"*"*50)
print(f"[Q2] ANSWER IS FOLLOWING:\n\n")
for solver in solvers:
    # Initialize and train logistic regression model
    logistic_model = LogisticRegression(solver=solver, max_iter=1000)
    logistic_model.fit(X_train, y_train)

    # Predict and evaluate
    predicted = logistic_model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted)
    print(f"\t", "*"*25)
    print(f"\t Accuracy with solver '{solver}': {accuracy}")

    # Cross-validation
    cv_scores = cross_val_score(logistic_model, X, y, cv=5)
    print(f"\t Mean and Std with solver and 5-CROSS CV '{solver}': {cv_scores.mean()}, {cv_scores.std()}")
    print(f"\t", "*"*25, "\n")
    
    # Save the trained model
    filename = f"M22AIE207_lr_{solver}.joblib"  # Replace 'YourRollNumber' with your actual roll number
    save_trained_model(logistic_model, os.path.join("q2_models", filename))
print(f"\n\n[Q2] END OF ANSWER")
print(f"*"*50)
