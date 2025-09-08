This project uses neural networks to automate curve fitting for hyperelastic material stress–strain data. Unlike traditional regression-based methods, the ANN learns stress–strain relationships directly, improving efficiency and reducing the cost of material characterization.

# Dataset
The dataset was synthetically generated in MATLAB and includes nine models: Ogden, Reduced Polynomial, and Polynomial (orders 1–3). It contains 9,000 samples, each with 100 stress values, 100 strain values, and up to nine parameters. To mimic experimental uncertainty, random noise of 3–5% was added to half the samples.

# Model Architecture
The network is a fully connected feedforward model. Stress–strain curves shaped as (100, 2) are flattened into 200 features, which are passed through three hidden layers with 128, 64, and 32 neurons using ReLU activation. The output layer predicts the material parameters required by the selected model.

<img width="440" height="230" alt="Screenshot 2025-09-08 at 11 38 30 AM" src="https://github.com/user-attachments/assets/76b49844-8499-4493-a6c6-223885315ba2" />

# Training
The data was split into 80% training and 20% testing. Normalization to the range [0, 1] was applied to both the training and test features. The model was trained using the Adam optimizer with a mean absolute error (MAE) loss function for 100 epochs with a batch size of 8.

<img width="535" height="275" alt="Screenshot 2025-09-08 at 11 38 06 AM" src="https://github.com/user-attachments/assets/138b3e67-da9d-4f51-970f-96e7d29fcab0" />

# Evaluation
Before evaluation, stress–strain data were interpolated to match the input shape. The ANN predicted parameters for all models, and stress curves were reconstructed. The best-fit model was selected based on the R² Score, which measures how closely the predicted curves match the experimental data.

<img width="372" height="260" alt="Screenshot 2025-09-08 at 11 37 43 AM" src="https://github.com/user-attachments/assets/3a3a65fe-1e30-47f6-ac4d-135d85d4bcdd" />

# Conclusion
This project demonstrates that neural networks can complement and extend traditional curve-fitting methods, offering greater flexibility, automation, and reliability for analyzing complex material behaviour. The model achieved an average accuracy of 93% (R² = 0.93) in recreating stress–strain curves, showing strong potential for real-world material characterization.


