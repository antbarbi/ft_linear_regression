import numpy as np
import pandas as pd
from ft_pypackage import ft_error
from sklearn.preprocessing import StandardScaler


@ft_error()
def main():
    data = pd.read_csv("./data.csv")
    X_train = data['km']
    y_train = data['price']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    learning_rate = 0.1
    tetha0 = 0
    tetha1 = 0




if __name__ == "__main__":
    main()
