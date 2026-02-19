import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_frame = pd.read_csv("backend/Housing.csv")
model_data = np.load('backend/house_train_para.npz')

my_weights = model_data["weights"]
my_bias = model_data["my_bias"]
sk_weights = model_data["sk_weights"]
sk_bias = model_data["sk_bias"]
x_mean = model_data["x_mean"]
x_std = model_data["x_std"]
y_mean = model_data["y_mean"]
y_std =model_data["y_std"]
split_index = model_data["split_index"]
w_simple = model_data["simple_weights"]
intercept_simple = model_data["simple_intercept"]
x_test = model_data["x_test"]
y_test = model_data["y_test"]


print(x_test.shape , y_test.shape)

print(w_simple.shape , intercept_simple.shape)


# first we calculate the mse and rmse for 1 feature and multiple feature 


simple_prediction = ( ( ( x_test[:,0:1] - x_mean[0]) / x_std[0] ) @ w_simple ) + intercept_simple 
multiple_prediction = ( ( (x_test -x_mean) / x_std ) @ my_weights ) + my_bias 


simple_prediction_denormalized = (simple_prediction * y_std ) + y_mean
multiple_prediction_denormalized = (multiple_prediction * y_std ) + y_mean

mse_simple = ((y_test - simple_prediction_denormalized)**2).sum() / y_test.shape[0]
mse_multiple = ((y_test - multiple_prediction_denormalized) **2 ).sum() / y_test.shape[0]
rmse_multiple = np.sqrt(mse_multiple)
rmse_simple = np.sqrt(mse_simple)

print(f"the mse of simple prediction is :{mse_simple} ")
print(f"the mse of multiple prediction is :{mse_multiple} ")
print(f"the RMSE OF simple lr ={rmse_simple}")
print(f"the RMSE of multiple lr = {rmse_multiple}")

SSres = ((y_test - multiple_prediction_denormalized  )**2).sum()
SStot = ((y_test - y_test.mean()) **2 ).sum()
r_square = 1 - SSres / SStot 


MAE = (np.abs(y_test - multiple_prediction_denormalized)).sum() / y_test.shape[0]

print("Our Mae is =" , MAE)
print("the RMSE is = ", rmse_multiple)
print("the R square is : " , r_square)


