import numpy as np
import pandas as pd
from ft_pypackage import ft_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


@ft_error()
def main():
    # Load and prepare data
    data = pd.read_csv("./data.csv")
    X_train = data['km'].values.reshape(-1, 1)
    y_train = data['price'].values

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Set hyperparameters
    learning_rate = 0.1
    tetha0 = 0
    tetha1 = 0
    m = len(y_train)

    # Set up the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    # Plot the original data points
    ax.scatter(X_train.flatten(), y_train, color='blue', label='Training Data')
    ax.set_xlabel('Kilometers (Standardized)')
    ax.set_ylabel('Price')
    line, = ax.plot([], [], color='red', label='Regression Line')  # Initial empty line for the regression line
    ax.legend()

    # Initialize variables for tracking epochs and loss
    epoch = 0
    losses = []

    # Gradient Descent loop
    while True:
        predictions = tetha0 + tetha1 * X_train.flatten()

        # Compute gradients
        tmp_tetha0 = learning_rate * (1 / m) * np.sum(predictions - y_train)
        tmp_tetha1 = learning_rate * (1 / m) * np.sum((predictions - y_train) * X_train.flatten())

        # Update weights
        prev_tetha0 = tetha0
        prev_tetha1 = tetha1
        tetha0 -= tmp_tetha0
        tetha1 -= tmp_tetha1

        # Update regression line for visualization
        line.set_xdata(X_train.flatten())
        line.set_ydata(predictions)

        # Redraw the plot
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Track the loss (Mean Squared Error)
        loss = np.mean((predictions - y_train) ** 2)
        losses.append(loss)

        # Print and plot loss at intervals
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

        # Check for convergence
        if prev_tetha0 == tetha0 and prev_tetha1 == tetha1:
            break

        epoch += 1

    # Print final standardized and de-standardized parameters
    print(f"Final Std t0: {tetha0}, t1: {tetha1}")

    mean_km = scaler.mean_[0]
    std_km = scaler.scale_[0]

    tetha1_destandardized = tetha1 / std_km
    tetha0_destandardized = tetha0 - (tetha1_destandardized * mean_km)

    print(f"De-standardized t0: {tetha0_destandardized}, t1: {tetha1_destandardized}")

    # Save the final weights
    weights = {
        "tetha0": tetha0_destandardized,
        "tetha1": tetha1_destandardized
    }
    np.save('weights.npy', weights)

    # Keep the plot open after the loop
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
