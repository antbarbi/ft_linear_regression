import numpy as np
import pandas as pd
from ft_pypackage import ft_error
from sklearn.preprocessing import StandardScaler


@ft_error()
def main():
    data = pd.read_csv("./data.csv")
    X_train = data['km'].values.reshape(-1, 1)
    y_train = data['price'].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    learning_rate = 0.1
    
    tetha0 = 0
    tetha1 = 0
    m = len(y_train)

    while(True):
        predictions = tetha0 + tetha1 * X_train.flatten()

        tmp_tetha0 = learning_rate * (1 / m) * np.sum(predictions - y_train)
        tmp_tetha1 = learning_rate * (1 / m) * np.sum((predictions - y_train) * X_train.flatten())

        prev_tetha0 = tetha0
        prev_tetha1 = tetha1

        tetha0 -= tmp_tetha0
        tetha1 -= tmp_tetha1
        if prev_tetha0 == tetha0 and prev_tetha1 == tetha1:
            break

    print(f"Std t0: {tetha0}, t1: {tetha1}")

    mean_km = scaler.mean_[0]
    std_km = scaler.scale_[0]

    tetha1_destandardized = tetha1 / std_km
    tetha0_destandardized = tetha0 - (tetha1_destandardized * mean_km)

    print(f"De-standardized t0: {tetha0_destandardized}, t1: {tetha1_destandardized}")

    weights = {
        "tetha0": tetha0_destandardized,
        "tetha1": tetha1_destandardized
    }
    np.save('weights.npy', weights)

if __name__ == "__main__":
    main()
