import numpy as np

def entropy(timeseries: np.ndarray, I: list):
    """
    Calculates the entropy for a given triple based on mutual information and quantile-based discretization.

    Args:
        timeseries (np.ndarray): The time series data of the network.
        I (list): The chosen triple of indices.

    Returns:
        float: The calculated entropy value (S).
    """
    # Assign values based on indices from the triple
    timeseries[I[0] - 1, :] = X
    timeseries[I[1] - 1, :] = Y
    timeseries[I[2] - 1, :] = Z

    # Constants for calculations
    num = 5
    tlen = len(timeseries[0])
    nrunmax = 100
    P_sep = 10

    # Calculate the mutual information and related values
    MI, MIz, MIz_null, MIC, Theta_S, Theta2_T, Theta2_Tn, Sigma, Sigma_null_list, P, P_T, P_Tn = ifc.Theta_score_null_model(
        timeseries, I, num, tlen, nrunmax, True, True)

    # Decision tree to determine thresholds
    x = range(1, num + 1)
    th1, th2, c = decision_tree(x, MIz, disp_fig=False, disp_txt_rep=False, disp_tree=False)

    # Adjust indices and calculate quantiles
    I = np.array(I) - 1
    X, Y, Z = ifc.timeseries_quantile(timeseries, I, num, tlen)

    # Function to calculate entropy for a given subset of data
    def calculate_entropy(X_data: np.ndarray, Y_data: np.ndarray) -> float:
        n = np.zeros((10, 10))
        X_min, Y_min = np.min(X_data), np.min(Y_data)
        X_max, Y_max = np.max(X_data), np.max(Y_data)
        ival_X = (X_max - X_min) / P_sep
        ival_Y = (Y_max - Y_min) / P_sep

        for i in range(len(X_data)):
            x = np.floor((X_data[i] - X_min) / ival_X)
            y = np.floor((Y_data[i] - Y_min) / ival_Y)
            x = min(x, 9)  # Ensure x is within the bounds
            y = min(y, 9)  # Ensure y is within the bounds
            n[int(x), int(y)] += 1

        entropy = 0
        for i in range(10):
            for j in range(10):
                entropy += (n[i, j] / len(X_data)) ** 2
        return entropy

    # Calculate entropy for the three subsets
    X_a, Y_a = X[Z < th1], Y[Z < th1]
    Y_1 = calculate_entropy(X_a, Y_a)

    X_b, Y_b = X[np.logical_and(Z <= th2, Z >= th1)], Y[np.logical_and(Z <= th2, Z >= th1)]
    Y_2 = calculate_entropy(X_b, Y_b)

    X_c, Y_c = X[Z > th2], Y[Z > th2]
    Y_3 = calculate_entropy(X_c, Y_c)

    # Calculate entropy using the three values
    S = -(np.log(Y_1) + np.log(Y_2) + np.log(Y_3))

    return S
