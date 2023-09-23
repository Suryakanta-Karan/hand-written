# content of test_sample.py
# def inc(x):
#     return x + 1


# def test_answer():
#     assert inc(4) == 5
from utils import create_hparam_combo

def test_for_hparam_combinations_count():
    # A test case to check the all possible combinations of parameters are indeed generated 
    gamma_range = [0.001,0.01,0.1,1]
    C_range = [1,10,100,1000]
    
    h_params_combinations = create_hparam_combo(gamma_range,C_range )
    
    assert len(h_params_combinations) == len(gamma_range) * len(C_range)
    
def test_for_hparam_combinations_values():
    # A test case to check the all possible combinations of parameters are indeed generated 
    gamma_range = [0.001,0.01,0.1,1]
    C_range = [1,10,100,1000]
    
    h_params_combinations = create_hparam_combo(gamma_range,C_range )
    expected_param_combo1 = {'gamma':0.01, 'C':10}
    expected_param_combo2 = {'gamma':0.01, 'C':1 }
    
    assert (expected_param_combo1 in h_params_combinations) and (expected_param_combo2 in h_params_combinations)