import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, r2_score

#====================
#import evaluation dataset, trained networks, and max/mins for scaling
#====================

#import evaluation dataset
df = pd.read_excel('evaluation_dataset/modified.xlsx')

#import max/mins from training
scaling_dicts = joblib.load("trained_ann/scaling_data.pkl")

#import trained neural networks
ogden1_nn = keras.models.load_model("trained_ann/ogden1_model.keras")
ogden2_nn = keras.models.load_model("trained_ann/ogden2_model.keras")
ogden3_nn = keras.models.load_model("trained_ann/ogden3_model.keras")
poly1_nn = keras.models.load_model("trained_ann/poly1_model.keras")
poly2_nn = keras.models.load_model("trained_ann/poly2_model.keras")
poly3_nn = keras.models.load_model("trained_ann/poly3_model.keras")
red_poly1_nn = keras.models.load_model("trained_ann/red_poly1_model.keras")
red_poly2_nn = keras.models.load_model("trained_ann/red_poly2_model.keras")
red_poly3_nn = keras.models.load_model("trained_ann/red_poly3_model.keras")

#====================
#normalization functions
#====================

#scales the input stress-strain using the same max/min from ann training
def scale(predict_parameters_and_stress, strain_interp, stress_interp):
    stress_min = scaling_dicts["stress_min"][predict_parameters_and_stress] 
    stress_max = scaling_dicts["stress_max"][predict_parameters_and_stress]  
    y_min = scaling_dicts["y_min"][predict_parameters_and_stress].to_numpy() 
    y_max = scaling_dicts["y_max"][predict_parameters_and_stress].to_numpy()
    strain_min = scaling_dicts["strain_min"] 
    strain_max = scaling_dicts["strain_max"] 
    
    #normalizes the inputs to [0,1] range
    strain_interp_scaled = (strain_interp - strain_min)/(strain_max - strain_min)
    stress_interp_scaled = (stress_interp - stress_min)/(stress_max - stress_min)
    lambda_vals = 1.0 + strain_interp
    
    #X is input to NN: combines the interperated, scaled strain+stress
    X = np.stack([stress_interp_scaled, strain_interp_scaled], axis=-1)[None,]
    return y_min, y_max, X, lambda_vals

#convert NN predictions back to original parameter range
def unscale(y_pred_scaled, y_max, y_min, param_size):
    return y_pred_scaled * (y_max[:param_size] - y_min[:param_size]) + y_min[:param_size]

#====================
#stress functions
#====================

#calculate stress using predicted params
def ogden1_stress(lambda_vals, mu1, alpha1):
    sigma = (2 * mu1 / alpha1) * (lambda_vals ** alpha1 - lambda_vals ** (-alpha1 / 2))
    return sigma

def ogden2_stress(lambda_vals, mu1, alpha1, mu2, alpha2):
    term1 = (2 * mu1 / alpha1) * (lambda_vals ** alpha1 - lambda_vals ** (-0.5* alpha1))
    term2 = (2 * mu2 / alpha2) * (lambda_vals ** alpha2 - lambda_vals ** (-0.5*alpha2))
    return term1 + term2

def ogden3_stress(lambda_vals, mu1, alpha1, mu2, alpha2, mu3, alpha3):
    term1 = (2 * mu1 / alpha1) * (lambda_vals ** alpha1 - lambda_vals ** (-alpha1 / 2))
    term2 = (2 * mu2 / alpha2) * (lambda_vals ** alpha2 - lambda_vals ** (-alpha2 / 2))
    term3 = (2 * mu3 / alpha3) * (lambda_vals ** alpha3 - lambda_vals ** (-alpha3 / 2))
    return term1 + term2 + term3

def red_poly1_stress(lambda_vals, C10):
    return 2 * C10 * (lambda_vals**2 - lambda_vals**(-1))

def red_poly2_stress(lambda_vals, C10, C20):
    term1 = lambda_vals**2 - lambda_vals**(-1)
    term2 = lambda_vals**2 + 2 * lambda_vals**(-1)
    stress = 2 * C10 * term1 + 4 * C20 * term1 * (term2 - 3)
    return stress

def red_poly3_stress(lambda_vals, C10, C20, C30):
    term1 = lambda_vals**2 - lambda_vals**(-1)
    term2 = lambda_vals**2 + 2 * lambda_vals**(-1)
    stress = 2 * term1 * (C10 + 2 * C20 * (term2 - 3) + 3 * C30 * (term1 - 3)**2)
    return stress

def poly1_stress(lambda_vals, C10, C01):
    stress = 2*(lambda_vals**2 - lambda_vals**-1)*C10 - 2*(lambda_vals**-2 - lambda_vals)*C01
    return stress

def poly2_stress(lambda_vals, C10, C01, C20, C11, C02):
    I1 = lambda_vals**2 + 2*lambda_vals**-1
    I2 = lambda_vals**-2 + 2*lambda_vals
    A  = I1 - 3
    B  = I2 - 3
    stress = 2*(lambda_vals**2 - lambda_vals**-1)*(C10 + 2*C20*A + C11*B) \
           - 4*(lambda_vals**-2 - lambda_vals)*(C01 + C11*A + 2*C02*B)
    return stress

def poly3_stress(lambda_vals, C10, C01, C20, C11, C02, C30, C21, C12, C03):
    I1 = lambda_vals**2 + 2*lambda_vals**-1
    I2 = lambda_vals**-2 + 2*lambda_vals
    A  = I1 - 3
    B  = I2 - 3
    stress = 2*(lambda_vals**2 - lambda_vals**-1)*(C10 + 2*C20*A + C11*B +
                                   3*C30*A**2 + 2*C21*A*B + C12*B**2) \
           - 4*(lambda_vals**-2 - lambda_vals)*(C01 + C11*A + 2*C02*B +
                                C21*A**2 + 2*C12*A*B + 3*C03*B**2)
    return stress

#====================
#predict parameters
#====================

MODEL_REGISTRY = {
    "ogden1":{"nn": ogden1_nn, "param_size": 2, "stress": ogden1_stress},
    "ogden2":{"nn": ogden2_nn, "param_size": 4, "stress": ogden2_stress},
    "ogden3":{"nn": ogden3_nn, "param_size": 6, "stress": ogden3_stress},
    "red_poly1":{"nn": red_poly1_nn, "param_size": 1, "stress": red_poly1_stress},
    "red_poly2":{"nn": red_poly2_nn, "param_size": 2, "stress": red_poly2_stress},
    "red_poly3":{"nn": red_poly3_nn, "param_size": 3, "stress": red_poly3_stress},
    "poly1":{"nn": poly1_nn, "param_size": 2, "stress": poly1_stress},
    "poly2":{"nn": poly2_nn, "param_size": 5, "stress": poly2_stress},
    "poly3":{"nn": poly3_nn, "param_size": 9, "stress": poly3_stress},
}

#make predictions
#Given a model name, runs its NN, rescales parameters, computes predicted stress-strain curve        
def predict_parameters_and_stress (model, X, y_max, y_min, lambda_vals):
    
    entry = MODEL_REGISTRY[model]
    y_pred_scaled = entry["nn"](X, training=False).numpy()[0] #training = false avoids retracing warnings
    y_pred = unscale (y_pred_scaled, y_max, y_min, entry["param_size"])
    stress_pred = entry["stress"](lambda_vals, *y_pred)
    return stress_pred, y_pred

#====================
#plotting functions
#====================

#plot interpolation
def plot_interpolation(strain, stress, strain_interp, stress_interp):
    plt.figure(figsize=(8, 5))
    plt.plot(strain, stress, 'o-', label='Original', alpha=0.7)
    plt.plot(strain_interp, stress_interp, '--', label='Interpolated (100 points)', color='orange')
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.title('Original vs Interpolated Stress–Strain')
    plt.legend()
    plt.grid(True)
    plt.show()

#plot results
def plot_stress_strain(strain_interp, stress_interp, stress_pred, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(strain_interp, stress_interp, label="True (Interpolated)", linewidth=2)
    plt.plot(strain_interp, stress_pred, '--', color='orange', label="Predicted", linewidth=2)
    plt.title(f"Predicted {model_name} vs True Stress–Strain Curve")
    plt.xlabel("Strain (λ - 1)")
    plt.ylabel("Stress")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

#====================
#main function
#====================

def main():

    for num in range (15):

        model_list = ["ogden1", "ogden2", "ogden3", "poly1", "poly2", "poly3", "red_poly1", "red_poly2", "red_poly3"] #list of models
        top_model = None
        top_r2 = float("-inf")
        top_y_pred = None
        top_stress_pred = None

        #load both stress and strain columns, drop NaN cells
        subset = df[[f"strain_{num+1}", f"stress_{num+1}"]].dropna()

        strain = subset.iloc[:,0].values
        stress = subset.iloc[:,1].values
        print(strain.shape, stress.shape)

        #interpolate 100 stress-strain to match neural network input shape
        strain_interp = np.linspace(strain.min(), strain.max(), 100)
        stress_interp = np.interp(strain_interp, strain, stress)

        #plot interpolated and original stress-strain points
        plot_interpolation(strain, stress, strain_interp, stress_interp)

        #test all models to find the most accurate one
        for model in model_list:
            y_min, y_max, X, lambda_vals = scale(model, strain_interp, stress_interp)
            stress_pred, y_pred = predict_parameters_and_stress(model, X, y_max, y_min, lambda_vals)
            r2 = r2_score(stress_interp, stress_pred)
            mae = mean_absolute_error(stress_interp, stress_pred)
            print (model)
            print (f"r^2: {r2}")
            # print (f"mae: {mae}")

            #keep best model by r^2
            if r2 > top_r2:
                top_model = model
                top_r2 = r2
                top_y_pred = y_pred
                top_stress_pred = stress_pred

        print (f"\nChosen Model: {top_model}")
        print(f"R^2 Score: {top_r2}")
        print (f"\nPredicted Parameters{top_y_pred}")

        plot_stress_strain(strain_interp, stress_interp, top_stress_pred, top_model)

if __name__ == "__main__":
    main()
