# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Initialize data, parameters, and learning rate
3. Repeat for given iterations: • Predict output • Compute loss • Update weight and bias
4. Plot results and display parameters
5. Stop

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: MOHAMMED AFSAL S
RegisterNumber:  212225040247
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("C:/Users/acer/Documents/50_Startups.csv")

x = data["R&D Spend"].values
y = data["Profit"].values

x = (x - np.mean(x)) / np.std(x)

w = 0.0          # weight
b = 0.0          # bias
alpha = 0.01     # learning rate
epochs = 100
n = len(x)
losses = []

# Gradient Descent
for i in range(epochs):
    # Prediction
    y_hat = w * x + b

    # Loss (MSE)
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    # Gradients
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    # Update parameters
    w = w - alpha * dw
    b = b - alpha * db

# Plots
plt.figure(figsize=(12, 5))

# Loss vs Iterations
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

# Regression Line
plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Data")
plt.plot(x, w * x + b, label="Regression Line",color="red")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")

plt.tight_layout()
plt.show()

# Final Parameters
print("Final Weight (w):", w)
print("Final Bias (b):", b)
```

## Output:

<img width="1562" height="772" alt="image" src="https://github.com/user-attachments/assets/68ddcaf1-f8b4-442b-b314-ff6649e6b094" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
