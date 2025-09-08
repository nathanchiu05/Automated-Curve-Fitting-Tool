import numpy as np

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