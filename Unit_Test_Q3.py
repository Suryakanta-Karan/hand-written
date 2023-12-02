import joblib
from sklearn.linear_model import LogisticRegression
import os
import pytest

@pytest.mark.parametrize("solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
def test_loaded_model(solver):
    # Replace <rollno> with your actual value
    rollno = "M22AIE207"
    model_name = f"{rollno}_lr_{solver}.joblib"
    model_path = os.path.join('/home/suryakantak/hand-written/API/', model_name)

    # Load the model from the file
    loaded_model = joblib.load(model_path)

    # Check if the loaded model is an instance of Logistic Regression
    assert isinstance(loaded_model, LogisticRegression) == True
    print(f"----[DEBUG] ASSERT SUCCESS FOR test_loaded_model: {loaded_model}")

@pytest.mark.parametrize("solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
def test_solver_name_match(solver):
    # Replace <rollno> with your actual value
    rollno = "M22AIE207"
    model_name = f"{rollno}_lr_{solver}.joblib"
    model_path = os.path.join('/home/suryakantak/hand-written/API/', model_name)

    # Load the model from the file
    loaded_model = joblib.load(model_path)

    # Check if the solver name in the model file name matches the solver used in the model
    model_solver = loaded_model.get_params()['solver']
    assert solver == model_solver
    print(f"----[DEBUG] ASSERT SUCCESS FOR test_solver_name_match: {loaded_model}")
