import os
import numpy as np
import pandas as pd
from ft_pypackage import ft_error
from matplotlib import pyplot as plt


# Common eval metrics for regression models
#   Mean Absolute Error (MAE)
#   Mean Squared Error (MSE)
#   Root Mean Squared Error (RMSE)
#   R-sqared
#   Adjusted R squared

def parse() -> int:
    try:
        input_data = input("Please enter a mileage(km): ").strip()  # Read a single line of input and strip any leading/trailing whitespace
        input_data = int(input_data)
    except Exception:
        print("There was an error with the input")
        exit(-1)
    return input_data

def predict() -> int:
    mileage = parse()
    if os.path.exists("weights.npy"):
        weights = np.load("weights.npy", allow_pickle=True).item()
        print(weights)
        tetha0 = weights["tetha0"]
        tetha1 = weights["tetha1"]
    else:
        tetha0 = 0
        tetha1 = 0
    
    estimate_price = tetha0 + tetha1 * mileage
    print()
    print(f"{'Predicted price:':<35} {estimate_price:.4f}")
    return mileage, estimate_price


def evaluation_metrics(tetha0: float, tetha1: float, data: pd.DataFrame) -> None:
    y_true = data["price"].values
    y_pred = tetha0 + tetha1 * data["km"].values
    
    # Measure of how far pred is from truth, lower is better
    mae = np.mean(np.abs(y_true - y_pred))
    # Measure of how far pred is from truth but penalizes larger errors, lower is better
    mse = np.mean((y_true - y_pred) ** 2)
    # same unit as targer, average deviation from the actual price, lower is better
    rmse = np.sqrt(mse)
    # between 0 and 1, 0 = doesnt predict at all, 1 = perfect predictions
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    n = len(y_true)
    p = 1  # Number of predictors
    # Sane as r2 but penalizes irrelevant predictors/features
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    print(f"{'Mean Absolute Error (MAE):':<35} {mae:.4f}")
    print(f"{'Mean Squared Error (MSE):':<35} {mse:.4f}")
    print(f"{'Root Mean Squared Error (RMSE):':<35} {rmse:.4f}")
    print(f"{'R-squared (R²):':<35} {r2:.4f}")
    print(f"{'Adjusted R-squared:':<35} {adjusted_r2:.4f}")


@ft_error()
def main():
    mileage, predicted_price = predict()
    data = pd.read_csv("data.csv")
    
    # Plot the data points
    plt.plot(data["km"], data["price"], 'o', label='Data points')
    
    # Load the weights
    if os.path.exists("weights.npy"):
        weights = np.load("weights.npy", allow_pickle=True).item()
        tetha0 = weights["tetha0"]
        tetha1 = weights["tetha1"]
        
        # Calculate the predicted prices
        x = np.linspace(data["km"].min(), data["km"].max(), 100)
        y = tetha0 + tetha1 * x
        
        # Plot the regression line
        plt.plot(x, y, '-', label='Regression line')
    
    plt.plot(mileage, predicted_price, 'ro', label='Predicted Value', markersize=10)

    if "tetha0" in locals() and "tetha1" in locals():
        evaluation_metrics(tetha0, tetha1, data)

    # Add labels and title
    plt.title("Mileage vs Price")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.legend()
    
    # Show the plot
    plt.show()



if __name__ == "__main__":
    main()