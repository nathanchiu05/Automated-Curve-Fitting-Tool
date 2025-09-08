import os
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib


#Import stress equations from stress_equations.py
from stress_equations import (
    ogden1_stress, ogden2_stress, ogden3_stress,
    red_poly1_stress, red_poly2_stress, red_poly3_stress,
    poly1_stress, poly2_stress, poly3_stress
)

df = pd.read_csv('training_dataset.csv')

#model registry used to train networks one model at a time
MODEL_REGISTRY = {"ogden1":{"output_neurons": 2, "param_columns": ['param_1', 'param_2'], "nn": ogden1_stress},
              "ogden2":{"output_neurons": 4, "param_columns": ['param_1', 'param_2', 'param_3', 'param_4'], "nn": ogden2_stress},
              "ogden3":{"output_neurons": 6, "param_columns": ['param_1', 'param_2', 'param_3', 'param_4', 'param_5', 'param_6'], "nn": ogden3_stress},
              "red_poly1":{"output_neurons": 1, "param_columns": ['param_1'], "nn": red_poly1_stress},
              "red_poly2":{"output_neurons": 2, "param_columns": ['param_1', 'param_2'], "nn": red_poly2_stress},
              "red_poly3":{"output_neurons": 3, "param_columns": ['param_1', 'param_2', 'param_3'], "nn": red_poly3_stress},
              "poly1":{"output_neurons": 2, "param_columns": ['param_1', 'param_2'], "nn": poly1_stress},
              "poly2":{"output_neurons": 5, "param_columns": ['param_1', 'param_2', 'param_3', 'param_4', 'param_5'], "nn": poly2_stress},
              "poly3":{"output_neurons": 9, "param_columns": ['param_1', 'param_2', 'param_3', 'param_4', 'param_5', 'param_6', 'param_7', 'param_8', 'param_9'], "nn": poly3_stress}}

#====================
#choose model to train
#====================

choose_model = "ogden2" #use the model names in the MODEL_REGISTRY
model_info = MODEL_REGISTRY[choose_model]

#================================
#Split training/test + Normalize
#================================
X_train_dict, X_test_dict, y_train_dict, y_test_dict = {}, {}, {}, {} #these values will be scaled...
stress_test_dict_unscaled, strain_test_dict_unscaled, y_test_dict_unscaled = {}, {}, {} #...hence why we have this so we dont have to unscale later. more efficient
stress_min_dict, stress_max_dict, y_min_dict, y_max_dict = {}, {}, {}, {} 

strain_min = 0.0 #strain min/max same for every value
strain_max = 3.0

model_names = df['model'].unique()

#Loops through each equation name (9 times), splits, normalizes, and adds to the training/test set
for model in model_names:

    mask = df['model'] == model #boolean indexing. Filter rows by model name.
    X = df[mask].drop(columns = ['model', 'param_1','param_2','param_3','param_4','param_5','param_6','param_7','param_8', 'param_9']) #Training Features [1000x100]
    y = df[mask][['param_1','param_2','param_3','param_4','param_5','param_6','param_7','param_8', 'param_9']] #Target Features Shape[1000x9]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    #split X into stress and strain
    stress_train = X_train.iloc[:, 0:100]
    stress_test = X_test.iloc[:, 0:100]
    strain_train = X_train.iloc[:, 100:200] #grab strain columns 100-200
    strain_test = X_test.iloc[:, 100:200]

    #Get min/max from training data only
    stress_min_dict[model] = stress_train.min(axis=0) #Axis 0 = grab min/max column wise
    stress_max_dict[model] = stress_train.max(axis=0)
    y_min_dict[model] = y_train.min(axis=0)
    y_max_dict[model] = y_train.max(axis=0)

    #Normalize training and target features using normalization formula on scale [0,1]
    stress_train_scaled = (stress_train - stress_min_dict[model]) / (stress_max_dict[model] - stress_min_dict[model])
    stress_test_scaled = (stress_test - stress_min_dict[model]) / (stress_max_dict[model] - stress_min_dict[model])

    strain_train_scaled = (strain_train - strain_min)/(strain_max - strain_min)
    strain_test_scaled = (strain_test - strain_min)/(strain_max - strain_min)

    y_train_scaled = (y_train - y_min_dict[model]) / (y_max_dict[model] - y_min_dict[model]) 
    y_test_scaled = (y_test - y_min_dict[model]) / (y_max_dict[model] - y_min_dict[model])

    #These dictionaries contain SCALED train/test values
    X_train_dict[model] = np.stack([stress_train_scaled, strain_train_scaled], axis=-1)
    X_test_dict[model] = np.stack([stress_test_scaled, strain_test_scaled], axis=-1)
    y_train_dict[model] = y_train_scaled
    y_test_dict[model] = y_test_scaled

    #These dictionaries contain UNSCALED train/test values. Useful for interperating results, removes the need to unscale
    stress_test_dict_unscaled [model] = stress_test
    strain_test_dict_unscaled [model]= strain_test
    y_test_dict_unscaled[model] = y_test 

#================================
#Define and train the neural network
#================================

#define the model
nn = models.Sequential([
    layers.Input(shape=(100,2)), 
    layers.Flatten(), #Flatten input shape from 2D array to 1D array [[stress1, strain1], [stress2,strain2]] -> [stress1, strain1, stress2, strain2]
    layers.Dense(128, activation = 'relu'), #ReLu activation function introduced non-linearity
    layers.Dense(64, activation = 'relu'), 
    layers.Dense(32, activation = 'relu'),
    layers.Dense(model_info["output_neurons"]
)
])

optimizer = Adam(learning_rate=0.0001) #Adam (Adaptive Moment Estimation) adjusts the weights and biases
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) #stops training once model stops improving
nn.compile(optimizer=optimizer, loss='mae')

# store_history = nn.fit(
nn.fit(
    X_train_dict[choose_model], y_train_dict[choose_model][model_info["param_columns"]].to_numpy(), 
    epochs=100, #epoch = one complete pass through entire training data
    batch_size=8, #9000 samples / batch size 8 = 1125 batches per epoch
    validation_data=(X_test_dict[choose_model], y_test_dict[choose_model][model_info["param_columns"]].to_numpy()),  #validate model using test split
    callbacks=[early_stop]
)

#================================
#Save neural networks and scaling data (max/min values) -> files save to /training_files folder
#================================

# #Save training (stress/strain) and target (y) max/mins as a single dict
# #only need to save scaling data once because the max/mins will stay constant during every run
# scaling_data = {
#     "stress_min": stress_min_dict,
#     "stress_max": stress_max_dict,
#     "strain_min": strain_min,
#     "strain_max": strain_max,
#     "y_min": y_min_dict,
#     "y_max": y_max_dict
# }

# joblib.dump(scaling_data, "scaling_data.pkl")
# print("Scaling data saved to scaling_data.pkl")

# #save trained neural networks as keras
# nn.save(f"{choose_model}_model.keras")
# print (f"{model} nn saved to {choose_model}_model.keras ")

#================================
#Evaluate the neural network on the Test split
#================================

#Predict on test set
predicted_scaled = nn.predict(X_test_dict[choose_model])

y_max = y_max_dict[choose_model][model_info["param_columns"]].to_numpy() #[param_columns] ensures y_max/min is the same shape as predicted_scaled (removes NaN in the unused params)
y_min = y_min_dict[choose_model][model_info["param_columns"]].to_numpy() #to_numpy converts pandas df into a numpy array so we can perform operations

#unscale predicted parameters back to its original range
y_pred = predicted_scaled * (y_max - y_min) + y_min
y_true = y_test_dict_unscaled[choose_model].to_numpy()

#================================
#Get R^2 scores for the entire test set
#================================

#this function loops through the 200 test split samples, calculates the predicted stress, and compares them to the true stress
def get_r2_score():

    total = 0
    r2_score_list = []

    print (f"{choose_model} R^2 Scores:")

    for i in range (200):
        true_stress = stress_test_dict_unscaled[choose_model].iloc[i].to_numpy()
        strain_vals = strain_test_dict_unscaled[choose_model].iloc[i].to_numpy()
        lambda_vals = strain_vals + 1

        predicted_stress = model_info["nn"](lambda_vals, *y_pred[i])
        
        #R^2 for current sample
        r2 = r2_score(true_stress, predicted_stress)
        r2_score_list.append(r2) #add r^2 to list
        print (f"Sample {i}: {r2}")
        total = total + r2 #sum of r^2 used to calculate mean

    print (f"\nmean R^2: {total/200}")
    print(f"median R^2: {np.median(r2_score_list)}")

    return r2_score_list

#================================
#Compare predicted and true stress-strain
#================================

#this function displays the results between predicted and true stress-strian
def results(sample, best_worst, n):

    strain_vals = strain_test_dict_unscaled[choose_model].iloc[sample].to_numpy()
    lambda_vals = strain_vals + 1
        
    #calculate stress using predicted params. model_info["nn"] provides the stress equation
    pred_stress = model_info["nn"](lambda_vals, *y_pred[sample]).flatten()
    true_stress = stress_test_dict_unscaled[choose_model].to_numpy() #true stress stored in dict

    #evaluate stress using r^2  and MAE
    r2 = r2_score(true_stress[sample], pred_stress)
    mae = mean_absolute_error(true_stress[sample], pred_stress)

    print (f"\n------------------------------\nFIGURE {n} - {best_worst} Sample:\n\nModel: {choose_model}")
    print(f"Sample {sample}\nR-squared score: {r2}")
    print(f"MAE: {mae:.4f}")
    print(f"\nTrue Parameters: {y_true[sample][:model_info["output_neurons"]]}\nPredicted Parameters: {y_pred[sample]}\n")
    # print(f"\nPredicted Stress: {[pred_stress[:50]]}")
    # print(f"\nTrue Stress: {[true_stress[sample][:50]]}")

    #plot true and predicted stress-strain
    plt.figure(n)
    plt.plot(strain_vals, true_stress[sample], label=f'{choose_model} True Stress', linewidth=2)
    plt.plot(strain_vals, pred_stress, label=f'{choose_model} Predicted Stress',linestyle='--', color='orange')
    plt.title(f"Predicted vs True Stress-Strain ({best_worst} Sample, R^2 = {r2:.6f})")
    plt.xlabel("Strain (λ - 1)")
    plt.ylabel("Stress")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.figtext(0.95, 0.02, f"FIGURE {n}", ha="right", va="bottom", fontsize=10)
    plt.show()

#find the best and worst predictions
r2_list = get_r2_score()
highest_sample = np.argmax(r2_list)
lowest_sample = np.argmin(r2_list)

#call the results function to display the best and worst predictions
results(highest_sample, "Best", 1)
results(lowest_sample, "Worst", 2)






