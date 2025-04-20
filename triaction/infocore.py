from sklearn.feature_selection import mutual_info_regression
import sklearn.preprocessing
import scipy as sp
import numpy as np
import math

def timeseries_quantile(timeseries: np.ndarray, I: list, num: int, tlen: int):
    """
    Computes quantile labels and sorts corresponding values from a time series dataset.

    This function selects the last `tlen` time points from a 2D time series array and sorts
    values from three specified rows (via indices in `I`) based on the values of the third row.
    It then assigns quantile labels to the sorted third row values, splitting them into `num`
    quantile bins.

    Args:
        timeseries (np.ndarray): 
            A 2D NumPy array of shape (n_series, time_steps) representing the time series data.
        I (list): 
            A list of three integers specifying the indices of the rows to use. 
            - `I[0]`: used for X values
            - `I[1]`: used for Y values
            - `I[2]`: used for sorting and quantile labeling
        num (int): 
            Number of quantile bins to compute.
        tlen (int): 
            Length of the time window (from the end) to consider in the time series.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - `X_sort`: Values from the row at `I[0]`, sorted by `I[2]`
            - `Y_sort`: Values from the row at `I[1]`, sorted by `I[2]`
            - `sp`: Quantile labels (1 to `num`) corresponding to sorted values of `I[2]`
    """

    timeseries = timeseries[:,-tlen:]
    dtlen = int(np.floor(tlen/num))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    idx = Z.argsort()
    Z_sort = Z[idx]
    X_sort = X[idx]
    Y_sort = Y[idx]
    sp = np.array_split(Z, num)
    compt = 1
    for k in range(len(sp)):
        sp[k] = compt*np.ones(len(sp[k]))
        compt += 1
        
    sp = np.hstack(sp)
    
    return X_sort, Y_sort, sp

    
def timeseries_quantile_val(timeseries: np.ndarray, I: list, num: int, tlen: int):
    """
    Computes quantile labels and sorted values from specified rows of time series data.

    This function selects the last `tlen` time steps from the input 2D time series array.
    It extracts values from three specified rows (by indices in `I`), sorts them based on 
    the values from the third row, and assigns quantile labels across `num` bins.

    Args:
        timeseries (np.ndarray): 
            A 2D NumPy array of shape (n_series, time_steps) representing the time series data.
        I (list): 
            A list of three integers specifying the indices of the rows to use:
            - `I[0]`: row for X values
            - `I[1]`: row for Y values
            - `I[2]`: row used for sorting and quantile labeling
        num (int): 
            The number of quantile bins to compute.
        tlen (int): 
            The length of the trailing time window to consider.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - `X_sort`: Values from row `I[0]`, sorted by values in `I[2]`
            - `Y_sort`: Values from row `I[1]`, sorted by values in `I[2]`
            - `Z_sort`: Sorted values from row `I[2]`, used for quantile computation
            - `sp`: Quantile labels (ranging from 1 to `num`) for the values in `Z_sort`
    """

    timeseries = timeseries[:,-tlen:]
    dtlen = int(np.floor(tlen/num))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    idx = Z.argsort()
    Z_sort = Z[idx]
    X_sort = X[idx]
    Y_sort = Y[idx]
    sp = np.array_split(Z, num)
    compt = 1
    for k in range(len(sp)):
        sp[k] = compt*np.ones(len(sp[k]))
        compt += 1
        
    sp = np.hstack(sp)
    
    return X_sort, Y_sort, Z_sort, sp


def mutual_information_analysis_continuous(timeseries: np.ndarray, I: list, num: int, tlen: int):
    """
    Analyzes mutual information between continuous variables in a time series.

    This function computes the overall mutual information between two continuous variables 
    (rows of a 2D time series array), as well as how that mutual information varies across 
    intervals defined by a third variable. The time series is first truncated to the last 
    `tlen` points, then sorted based on a third row (specified in `I[2]`) and divided into 
    `num` equally sized intervals. Mutual information is computed in each interval, and 
    summary statistics are reported.

    Args:
        timeseries (np.ndarray): 
            A 2D NumPy array of shape (n_series, time_steps) representing the time series data.
        I (list): 
            A list of three integers specifying the indices of rows used in the analysis:
            - `I[0]`: row for X (input variable)
            - `I[1]`: row for Y (target variable)
            - `I[2]`: row used to sort and divide data into intervals
        num (int): 
            Number of intervals to divide the sorted data into for localized mutual information analysis.
        tlen (int): 
            Length of the time window to consider from the end of the series.

    Returns:
        Tuple[float, np.ndarray, float, float, float, float]: 
            - `MI`: Overall mutual information between X and Y across the entire time window.
            - `MIz`: Array of mutual information values computed in each interval.
            - `MIC`: Mean mutual information across all intervals.
            - `Sigma`: Standard deviation of mutual information values.
            - `T`: Total range (max - min) of mutual information values.
            - `Tn`: Maximum absolute difference between consecutive mutual information values (temporal variability).
    """

    timeseries = timeseries[:,-tlen:]
    dtlen = int(np.floor(tlen/num))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    Xa = np.zeros((tlen,2))
    Xa[:,0] = X
    MI = mutual_info_regression(Xa, Y, discrete_features = False)
    MI = MI[0]
    idx = Z.argsort()
    Z_sort = Z[idx]
    X_sort = X[idx]
    Y_sort = Y[idx]
    Xn = np.zeros((dtlen,2))
    MIz = np.zeros((num))
    
    for i in range(num):
        Xn[:,0] = X_sort[i*dtlen:(i+1)*dtlen]
        y = Y_sort[i*dtlen:(i+1)*dtlen]
        mi = mutual_info_regression(Xn, y, discrete_features = False)
        MIz[i] = mi[0]
        
    MIC = np.mean(MIz)
    Sigma = np.std(MIz)
    T = np.max(MIz)-np.min(MIz)
    Tn = np.max(abs(MIz[0:num-1]-MIz[1:num]))
    
    return MI, MIz, MIC, Sigma, T, Tn


def mutual_information_analysis_continuous_extended(timeseries: np.ndarray, I: list, num: int, tlen: int):
    """
    Extended analysis of mutual information between continuous variables in a time series.

    This function performs a more detailed analysis of mutual information (MI) between two continuous 
    variables (specified by rows in a time series). In addition to computing the overall mutual information 
    between the variables, it evaluates the MI in multiple intervals, calculates temporal statistics, and 
    provides additional metrics such as Pearson correlation and the difference between MI values at the 
    beginning and end of the time series.

    Args:
        timeseries (np.ndarray): 
            A 2D NumPy array of shape (n_series, time_steps) representing the time series data.
        I (list): 
            A list of three integers specifying the indices of rows used in the analysis:
            - `I[0]`: row for X (input variable)
            - `I[1]`: row for Y (target variable)
            - `I[2]`: row used to sort and divide data into intervals
        num (int): 
            Number of intervals to divide the sorted data into for localized mutual information analysis.
        tlen (int): 
            The length of the time window to consider from the end of the series.

    Returns:
        Tuple[float, np.ndarray, float, float, float, float, float, float, float, float]: 
            - `MI`: Overall mutual information between X and Y across the entire time window.
            - `MIz`: Array of mutual information values computed in each interval.
            - `MIC`: Mean mutual information across all intervals.
            - `Corr_Sigma`: Pearson correlation coefficient between the mean of Z values and MIz.
            - `Sigma`: Standard deviation of mutual information values.
            - `T`: Range (max - min) of mutual information values across intervals.
            - `Tn`: Maximum absolute difference between consecutive mutual information values (temporal variability).
            - `MINDY`: Absolute difference between mutual information values from the first and last third of the data.
            - `MI1`: Mutual information of the first third of the data.
            - `MI2`: Mutual information of the last third of the data.
    """

    timeseries = timeseries[:,-tlen:]
    dtlen = int(np.floor(tlen/num))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    Xa = np.zeros((tlen,2))
    Xa[:,0] = X
    MI = mutual_info_regression(Xa, Y, discrete_features = False)
    MI = MI[0]
    idx = Z.argsort()
    Z_sort = Z[idx]
    X_sort = X[idx]
    Y_sort = Y[idx]
    Xn = np.zeros((dtlen,2))
    Zn = np.zeros((dtlen,2))
    MIz = np.zeros((num))
    z = np.zeros((num))
    for i in range(num):
        Xn[:,0] = X_sort[i*dtlen:(i+1)*dtlen]
        Zn[:,0] = Z_sort[i*dtlen:(i+1)*dtlen]
        y = Y_sort[i*dtlen:(i+1)*dtlen]
        mi = mutual_info_regression(Xn, y, discrete_features = False)
        MIz[i] = mi[0]
        z[i] = np.mean(Zn)
        
    MIC = np.mean(MIz)
    Sigma = np.std(MIz)
    T = np.max(MIz)-np.min(MIz)
    Tn = np.max(abs(MIz[0:num-1]-MIz[1:num]))
    Corr_Sigma = sp.stats.pearsonr(z, MIz)[0]
    
    MINDYz = np.zeros((3))
    dtlen = int(np.floor(tlen/3))
    Xn = np.zeros((dtlen,2))
    for i in range(3):
        Xn[:,0] = X_sort[i*dtlen:(i+1)*dtlen]
        y = Y_sort[i*dtlen:(i+1)*dtlen]
        mi = mutual_info_regression(Xn, y, discrete_features = False)
        MINDYz[i] = mi[0]
    MI1 = MINDYz[0]
    MI2 = MINDYz[-1]
    MINDY = np.abs(MI1 - MI2)
        
    return MI, MIz, MIC, Corr_Sigma, Sigma, T, Tn, MINDY, MI1, MI2


def mindy_final(timeseries: np.ndarray, I: list, tlen: int):
    """
    Computes the MINDy measure between continuous variables.

    The MINDy (Mutually Information Normalized Difference Yield) measure is based on the work by Wang, 
    Saito, Bisikirska et al., and it quantifies the change in mutual information over time between two 
    continuous variables (represented by rows in a time series). This function computes the mutual information 
    for the first and last thirds of the time series and calculates the absolute difference between them.

    Reference:
        Wang, K., Saito, M., Bisikirska, B. et al. Nat Biotechnol 27, 829–837 (2009).
        https://doi.org/10.1038/nbt.1563

    Args:
        timeseries (np.ndarray): 
            A 2D NumPy array of shape (n_series, time_steps) representing the time series data.
        I (list): 
            A list of three integers specifying the indices of rows used in the analysis:
            - `I[0]`: row for X (input variable)
            - `I[1]`: row for Y (target variable)
            - `I[2]`: row used to sort and divide data
        tlen (int): 
            The length of the time window to consider for analysis (i.e., the number of time steps).

    Returns:
        Tuple[float, float, float]: 
            - `MINDY`: The absolute difference in mutual information between the first and last thirds of the data.
            - `MI1`: Mutual information of the first third of the time series.
            - `MI2`: Mutual information of the last third of the time series.
    """
    
    timeseries = timeseries[:,-tlen:]
    dtlen = int(np.floor(tlen/3))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    idx = Z.argsort()
    Z_sort = Z[idx]
    X_sort = X[idx]
    Y_sort = Y[idx]
    Xn = np.zeros((dtlen,2))
    MIz = np.zeros((3))
    for i in range(3):
        Xn[:,0] = X_sort[i*dtlen:(i+1)*dtlen]
        y = Y_sort[i*dtlen:(i+1)*dtlen]
        mi = mutual_info_regression(Xn, y, discrete_features = False)
        MIz[i] = mi[0]
    MI1 = MIz[0]
    MI2 = MIz[-1]
    MINDY = np.abs(MI1 - MI2)
        
    return MINDY, MI1, MI2


def mindy(timeseries:np.ndarray, I:list, tlen:int):
    """
    Computes the MINDy measure between continuous variables.

    The MINDy (Mutually Information Normalized Difference Yield) measure quantifies the change in mutual 
    information between two continuous variables over time. This function calculates the mutual information 
    for the first and last thirds of a time series, both using the `mutual_info_regression` method and an 
    older method (`information_mutual`) for comparison.

    Reference:
        Wang, K., Saito, M., Bisikirska, B. et al. Nat Biotechnol 27, 829–837 (2009).
        https://doi.org/10.1038/nbt.1563

    Args:
        timeseries (numpy.ndarray): 
            A 2D NumPy array of shape (n_series, time_steps) representing the time series data.
        I (list): 
            A list of three integers specifying the indices of rows used in the analysis:
            - `I[0]`: row for X (input variable)
            - `I[1]`: row for Y (target variable)
            - `I[2]`: row used to sort and divide data into intervals
        tlen (int): 
            The length of the time window to consider for analysis (i.e., the number of time steps).

    Returns:
        tuple: 
            A tuple containing six elements:
            - `MINDY` (float): The absolute difference in mutual information between the first and last thirds of the data.
            - `MI1` (float): Mutual information of the first third of the time series (using `mutual_info_regression`).
            - `MI2` (float): Mutual information of the last third of the time series (using `mutual_info_regression`).
            - `oldMINDY` (float): The absolute difference in mutual information between the first and last thirds, 
              calculated using the `information_mutual` method.
            - `oldMI1` (float): Mutual information of the first third using `information_mutual`.
            - `oldMI2` (float): Mutual information of the last third using `information_mutual`.
    """

    timeseries = timeseries[:,-tlen:]
    dtlen = int(np.floor(tlen/3))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    idx = Z.argsort()
    Z_sort = Z[idx]
    X_sort = X[idx]
    Y_sort = Y[idx]
    MINDYz = np.zeros((3))
    oldMINDYz = np.zeros((3))
    Xn = np.zeros((dtlen,2))
    for i in range(3):
        Xn[:,0] = X_sort[i*dtlen:(i+1)*dtlen]
        y = Y_sort[i*dtlen:(i+1)*dtlen]
        mi = mutual_info_regression(Xn, y, discrete_features = False)
        MINDYz[i] = mi[0]
        oldMINDYz[i] = drv.information_mutual(np.round(Xn[:,0]), np.round(y))
    MI1 = MINDYz[0]
    MI2 = MINDYz[2]
    MINDY = np.abs(MI1 - MI2)
    oldMI1 = oldMINDYz[0]
    oldMI2 = oldMINDYz[2]
    oldMINDY = np.abs(oldMI1 - oldMI2)
        
    return MINDY, MI1, MI2, oldMINDY, oldMI1, oldMI2
    

def mutual_information(timeseries: np.ndarray, I: list, num: int, tlen: int):
    """
    Computes mutual information measures between continuous variables in a time series.

    This function computes the mutual information between two continuous variables (X and Y) over the 
    entire time series, as well as the mutual information in several intervals, allowing for an analysis 
    of how this relationship evolves over time. It also returns the mean mutual information across intervals.

    Args:
        timeseries (np.ndarray): 
            A 2D NumPy array of shape (n_series, time_steps) representing the time series data.
        I (list): 
            A list of three integers specifying the indices of rows used in the analysis:
            - `I[0]`: row for X (input variable)
            - `I[1]`: row for Y (target variable)
            - `I[2]`: row used to sort and divide data into intervals
        num (int): 
            Number of intervals to divide the time series for localized mutual information analysis.
        tlen (int): 
            The length of the time window to consider for analysis (i.e., the number of time steps).

    Returns:
        Tuple[float, np.ndarray, float]: 
            - `MI` (float): Mutual information between the first and second rows of interest (over the entire time window).
            - `MIz` (np.ndarray): An array of mutual information values for each interval.
            - `MIC` (float): Mean mutual information across all intervals.
    """

    timeseries = timeseries[:,-tlen:]
    dtlen = int(np.floor(tlen/num))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    Xa = np.zeros((tlen,2))
    Xa[:,0] = X
    MI = mutual_info_regression(Xa, Y, discrete_features = False)
    MI = MI[0]
    idx = Z.argsort()
    Z_sort = Z[idx]
    X_sort = X[idx]
    Y_sort = Y[idx]
    Xn = np.zeros((dtlen,2))
    MIz = np.zeros((num))
    
    for i in range(num):
        Xn[:,0] = X_sort[i*dtlen:(i+1)*dtlen]
        y = Y_sort[i*dtlen:(i+1)*dtlen]
        mi = mutual_info_regression(Xn, y, discrete_features = False)
        MIz[i] = mi[0]
        
    MIC = np.mean(MIz)
    
    return MI, MIz, MIC


def sigma(timeseries: np.ndarray, I: list, num: int, tlen: int, null: bool = True, nrunmax: int = 5000):
    """
    Computes the standard deviation of mutual information values between continuous variables in a time series.
    This is defined as the Sigma measure.

    The function calculates the standard deviation of mutual information between two continuous variables (X and Y) 
    across several intervals of a time series. Additionally, it can simulate a null model of mutual information 
    to compare against the observed values.

    Args:
        timeseries (np.ndarray): 
            A 2D NumPy array of shape (n_series, time_steps) representing the time series data.
        I (list): 
            A list of three integers specifying the indices of rows used in the analysis:
            - `I[0]`: row for X (input variable)
            - `I[1]`: row for Y (target variable)
            - `I[2]`: row used to sort and divide data into intervals
        num (int): 
            Number of intervals to divide the time series for localized mutual information analysis.
        tlen (int): 
            The length of the time window to consider for analysis (i.e., the number of time steps).
        null (bool or int, optional): 
            If False, the function only computes the standard deviation of mutual information. 
            If an integer, simulates a null model with the specified number of runs. Defaults to True.
        nrunmax (int, optional): 
            Maximum number of runs for the null model simulation. Default is 5000.

    Returns:
        tuple: 
            A tuple with one or two elements:
            - `Sigma` (float): Standard deviation of conditional mutual information across intervals.
            - `Sigma_null` (float, optional): Standard deviation of conditional mutual information from the null model, 
              if `null` is not False. In this case, a tuple `(Sigma, Sigma_null)` is returned.

    Raises:
        TypeError: If `null` is not False or an integer.
    """

    timeseries = timeseries[:,-tlen:]
    dtlen = int(np.floor(tlen/num))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    idx = Z.argsort()
    Z_sort = Z[idx]
    X_sort = X[idx]
    Y_sort = Y[idx]
    Xn = np.zeros((dtlen,2))
    MIz = np.zeros((num))
    for i in range(num):
        Xn[:,0] = X_sort[i*dtlen:(i+1)*dtlen]
        y = Y_sort[i*dtlen:(i+1)*dtlen]
        mi = mutual_info_regression(Xn, y, discrete_features = False)
        MIz[i] = mi[0]
        
    Sigma = np.std(MIz)
    if null == False:
        return Sigma
    else:
        if type(null) == int:
            Sigma_null, T_null, Tn_null = null_model(timeseries, I, num, tlen, nrunmax, Gaussian_version = True,\
                                                                    Mutual_version = True, model = None)
            
            return Sigma, Sigma_null
    
        else:
            raise TypeError("Only integers or False accepted")
            

def t(timeseries: np.ndarray, I: list, num: int, tlen: int, null: bool = True, nrunmax: int = 5000):
    """
    Computes the range of mutual information values between continuous variables in a time series.
    This is defined as the T measure.

    The function calculates the range (i.e., the absolute difference between the maximum and minimum) 
    of mutual information between two continuous variables (X and Y) across several intervals of a time series.
    Additionally, it can simulate a null model of mutual information to compare against the observed values.

    Args:
        timeseries (np.ndarray): 
            A 2D NumPy array representing the time series data (shape: n_series, time_steps).
        I (list): 
            A list of three integers specifying the indices of rows used in the analysis:
            - `I[0]`: row for X (input variable)
            - `I[1]`: row for Y (target variable)
            - `I[2]`: row used to sort and divide data into intervals.
        num (int): 
            Number of intervals to divide the time series for localized mutual information analysis.
        tlen (int): 
            The length of the time window to consider for analysis (i.e., the number of time steps).
        null (bool or int, optional): 
            If False, the function only computes the range of mutual information. 
            If an integer, simulates a null model with the specified number of runs. Defaults to True.
        nrunmax (int, optional): 
            Maximum number of runs for the null model simulation. Default is 5000.

    Returns:
        tuple: 
            A tuple with one or two elements:
            - `T` (float): The range (absolute difference between max and min) of conditional mutual information.
            - `T_null` (float, optional): The range of conditional mutual information from the null model, 
              if `null` is not False. In this case, a tuple `(T, T_null)` is returned.

    Raises:
        TypeError: If `null` is not False or an integer.
    """

    timeseries = timeseries[:,-tlen:]
    dtlen = int(np.floor(tlen/num))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    idx = Z.argsort()
    Z_sort = Z[idx]
    X_sort = X[idx]
    Y_sort = Y[idx]
    Xn = np.zeros((dtlen,2))
    MIz = np.zeros((num))
    
    for i in range(num):
        Xn[:,0] = X_sort[i*dtlen:(i+1)*dtlen]
        y = Y_sort[i*dtlen:(i+1)*dtlen]
        mi = mutual_info_regression(Xn, y, discrete_features = False)
        MIz[i] = mi[0]
        
    T = np.max(MIz)-np.min(MIz)
    if null == False:
        return T
    else:
        if type(null) == int:
            Sigma_null, T_null, Tn_null = null_model(timeseries, I, num, tlen, nrunmax, Gaussian_version = True,\
                                                                    Mutual_version = True, model = None)
            
            return T, T_null
    
        else:
            raise TypeError("Only integers or False accepted")
    

def tn(timeseries: np.ndarray, I: list, num: int, tlen: int, null: bool = True, nrunmax: int = 5000):
    """
    Computes the maximum difference between consecutive mutual information values in a time series.
    This is defined as the Tn measure.

    The function calculates the maximum absolute difference between consecutive mutual information 
    values computed over several intervals of a time series. Additionally, it can simulate a null 
    model of mutual information to compare the observed values against.

    Args:
        timeseries (np.ndarray): 
            A 2D NumPy array representing the time series data (shape: n_series, time_steps).
        I (list): 
            A list of three integers specifying the indices of rows used in the analysis:
            - `I[0]`: row for X (input variable)
            - `I[1]`: row for Y (target variable)
            - `I[2]`: row used to sort and divide data into intervals.
        num (int): 
            Number of intervals to divide the time series for localized mutual information analysis.
        tlen (int): 
            The length of the time window to consider for analysis (i.e., the number of time steps).
        null (bool or int, optional): 
            If False, the function only computes the maximum difference (`Tn`). 
            If an integer, simulates a null model with the specified number of runs. Defaults to True.
        nrunmax (int, optional): 
            Maximum number of runs for the null model simulation. Default is 5000.

    Returns:
        tuple: 
            A tuple with one or two elements:
            - `Tn` (float): The maximum absolute difference between consecutive mutual information values.
            - `Tn_null` (float, optional): The maximum difference from the null model's mutual information values, 
              if `null` is not False. In this case, a tuple `(Tn, Tn_null)` is returned.

    Raises:
        TypeError: If `null` is not False or an integer.
    """

    timeseries = timeseries[:,-tlen:]
    dtlen = int(np.floor(tlen/num))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    idx = Z.argsort()
    Z_sort = Z[idx]
    X_sort = X[idx]
    Y_sort = Y[idx]
    Xn = np.zeros((dtlen,2))
    MIz = np.zeros((num))
    
    for i in range(num):
        Xn[:,0] = X_sort[i*dtlen:(i+1)*dtlen]
        y = Y_sort[i*dtlen:(i+1)*dtlen]
        mi = mutual_info_regression(Xn, y, discrete_features = False)
        MIz[i] = mi[0]

    Tn = np.max(abs(MIz[0:num-1]-MIz[1:num]))
    if null == False:
        return Tn
    else:
        if type(null) == int:
            Sigma_null, T_null, Tn_null = null_model(timeseries, I, num, tlen, nrunmax, Gaussian_version = True,\
                                                                    Mutual_version = True, model = None)
            
            return Tn, Tn_null
    
        else:
            raise TypeError("Only integers or False accepted")
    

def correlation_analysis_continuous(timeseries: np.ndarray, I: list, num: int, tlen: int):
    """
    Analyzes the correlation between continuous variables in a time series.

    The function calculates various correlation measures between two variables over specified intervals 
    of a time series. The primary analysis focuses on the correlation between the first and second rows of interest, 
    with additional metrics such as the range and maximum difference between consecutive correlation values.

    Args:
        timeseries (np.ndarray): 
            A 2D NumPy array representing the time series data (shape: n_series, time_steps).
        I (list): 
            A list of three integers specifying the indices of rows used in the analysis:
            - `I[0]`: row for X (first variable)
            - `I[1]`: row for Y (second variable)
            - `I[2]`: row used to sort and divide data into intervals.
        num (int): 
            Number of intervals to divide the time series for localized correlation analysis.
        tlen (int): 
            The length of the time window to consider for analysis (i.e., the number of time steps).

    Returns:
        tuple: 
            A tuple containing five elements:
            - `C` (float): The mean correlation between the first and second rows of interest.
            - `Cz` (numpy.ndarray): Array of correlation values for each interval.
            - `Sigma` (float): The variance of the correlation values.
            - `T` (float): The range of correlation values across intervals (i.e., `max(Cz) - min(Cz)`).
            - `Tn` (float): The maximum difference between consecutive correlation values.
    """

    timeseries = timeseries[:,-tlen:]
    dtlen = int(np.floor(tlen/num))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    idx = Z.argsort()
    Z_sort = Z[idx]
    X_sort = X[idx]
    Y_sort = Y[idx]
    Cz = np.zeros((num))
    Xaus = np.zeros((dtlen,2))
    
    for i in range(num):
        Xaus[:,0] = X_sort[i*dtlen:(i+1)*dtlen]
        Xaus[:,1] = Y_sort[i*dtlen:(i+1)*dtlen]
        C = np.cov(Xaus)
        Cz[i] = C[0,1]
        
    C = np.mean(Cz)
    Sigma = np.var(Cz)
    T = np.max(Cz)-np.min(Cz)
    Tn = np.max(abs(Cz[0:num-1]-Cz[1:num]))
    
    return C, Cz, Sigma, T, Tn


def null_model_results(M: np.ndarray, Cov: np.ndarray, timeseries: np.ndarray, I: list, num: int, tlen: int, 
                       Sigma: float, T: float, Tn: float, nrunmax: int, Gaussian_version: bool, 
                       Mutual_version: bool, model: int = None):
    """
    Computes the results of the null model for mutual conditional information.

    The function generates a null model for the mutual information (or correlation) analysis of continuous variables.
    It uses either a multivariate normal distribution or a randomized version of the time series to simulate null data
    and compares its properties (e.g., standard deviation, range, and maximum difference) to the original data.

    Args:
        M (np.ndarray): 
            Mean vector for the multivariate normal distribution.
        Cov (np.ndarray): 
            Covariance matrix for the multivariate normal distribution.
        timeseries (np.ndarray): 
            A 2D NumPy array representing the time series data (shape: n_series, time_steps).
        I (list): 
            A list of three integers specifying the indices of rows used in the analysis:
            - `I[0]`: row for X (first variable)
            - `I[1]`: row for Y (second variable)
            - `I[2]`: row used to sort and divide data into intervals.
        num (int): 
            Number of intervals to divide the time series for localized analysis.
        tlen (int): 
            The length of the time window to consider for analysis.
        Sigma (float): 
            Standard deviation of mutual information (or correlation) in the original data.
        T (float): 
            Range of mutual information (or correlation) values in the original data.
        Tn (float): 
            Maximum difference between consecutive mutual information (or correlation) values in the original data.
        nrunmax (int): 
            Maximum number of runs for the null model simulation.
        Gaussian_version (bool): 
            If True, uses a multivariate normal distribution for the null model.
        Mutual_version (bool): 
            If True, performs mutual information analysis; otherwise, performs correlation analysis.
        model (int, optional): 
            Null model type (0-7). Defaults to None. Determines the specific model for the null simulation.

    Returns:
        tuple: 
            A tuple containing:
            - `X_null` (float): The mean value of mutual information or correlation in the null model.
            - `Xz_null` (np.ndarray): Array of mutual information or correlation values for each interval in the null model.
            - `Theta` (float): Normalized difference in standard deviation between the original and null models.
            - `Theta_T` (float): Normalized difference in range between the original and null models.
            - `Theta_Tn` (float): Normalized difference in maximum difference between the original and null models.
            - `Sigma` (float): Standard deviation of mutual information or correlation values in the original data.
            - `Sigma_null_list` (np.ndarray): Standard deviation of mutual information or correlation values for each run in the null model.
            - `P` (float): Probability of observing a standard deviation greater than the original in the null model.
            - `P_T` (float): Probability of observing a range greater than the original in the null model.
            - `P_Tn` (float): Probability of observing a maximum difference greater than the original in the null model.

    Notes:
        - The function generates a null model using either a multivariate normal distribution (Gaussian model) or 
          randomized data (shuffling). It then calculates mutual information or correlation for the null model 
          and compares it to the original data.
        - Model types are as follows:
            - Z-Y-X-Z: 0
            - Z Y-X-Z: 1
            - Z-Y X-Z: 2
            - Z-Y-X Z: 3
            - Z Y X-Z: 4
            - Z-Y X-Z: 5
            - Z Y-X Z: 6
            - Z Y X Z: 7
    """

    I2 = I
    if (Gaussian_version==True):
        MT = M[I]
        CovT = Cov[I]
        CovT = CovT[:,I]
        if model == 1:
            Cov[1,2] = 0
            Cov[2,1] = Cov[1,2]
        elif model == 2:
            Cov[0,1] = 0
            Cov[1,0] = Cov[0,1]
        elif model == 3:
            Cov[0,2] = 0
            Cov[2,0] = Cov[0,2]
        elif model == 4:
            Cov[1,2] = 0
            Cov[2,1] = Cov[1,2]
            Cov[0,1] = 0
            Cov[1,0] = Cov[0,1]
        elif model == 5:
            Cov[0,1] = 0
            Cov[1,0] = Cov[0,1]
            Cov[0,2] = 0
            Cov[2,0] = Cov[0,2]
        elif model == 6:
            Cov[1,2] = 0
            Cov[2,1] = Cov[1,2]
            Cov[0,2] = 0
            Cov[2,0] = Cov[0,2]
        elif model == 7:
            Cov[1,2] = 0
            Cov[2,1] = Cov[1,2]
            Cov[0,1] = 0
            Cov[1,0] = Cov[0,1]
            Cov[0,2] = 0
            Cov[2,0] = Cov[0,2]
        mult_dist = sp.stats.multivariate_normal(mean = MT, cov = CovT)
        I2 = [0,1,2]
      
    Sigma_null_list = []
    T_null_list = []
    Tn_null_list = []
    
    for n in range(nrunmax):
        if(Gaussian_version==False): 
            null_timeseries = np.array(timeseries).copy()
            np.random.shuffle(null_timeseries[I2[2], :])
            
        elif(Gaussian_version==True):    
            null_timeseries = mult_dist.rvs(tlen)
            null_timeseries = np.transpose(null_timeseries)
            
        if(Mutual_version==False):
            X_null, Xz_null, Sigma_null, T_null, Tn_null = correlation_analysis_continuous(null_timeseries, I2, num, tlen)
            
        elif(Mutual_version==True):
            X_null, Xz_null, MIC_null, Sigma_null, T_null, Tn_null = mutual_information_analysis_continuous(null_timeseries, I2, num, tlen)
            
        Sigma_null_list.append(Sigma_null)
        T_null_list.append(T_null)
        Tn_null_list.append(Tn_null)
        
    Sigma_null_list = np.array(Sigma_null_list)
    T_null_list = np.array(T_null_list)
    Tn_null_list = np.array(Tn_null_list)
    Sigma_mean_null = np.mean(Sigma_null_list)
    T_mean_null = np.mean(T_null_list)
    Tn_mean_null = np.mean(Tn_null_list)
    Sigma_null = np.std(Sigma_null_list)
    T_std_null = np.std(T_null_list)
    Tn_std_null = np.std(Tn_null_list)
    Theta = abs(Sigma-Sigma_mean_null)/(Sigma_null)
    Theta_T = abs(T-T_mean_null)/T_std_null
    Theta_Tn = abs(Tn-Tn_mean_null)/Tn_std_null
    P = np.count_nonzero(Sigma_null_list[Sigma_null_list>Sigma])/nrunmax
    P = P if P>0 else 1/nrunmax
    P_T = np.count_nonzero(T_null_list[T_null_list>T])/nrunmax
    P_T = P_T if P_T>0 else 1/nrunmax
    P_Tn = np.count_nonzero(Tn_null_list[Tn_null_list>Tn])/nrunmax
    P_Tn = P_Tn if P_Tn>0 else 1/nrunmax
    
    print('Sigma, T, Tn',Sigma, T, Tn)
    print('Sigma_mean_null', Sigma_mean_null)
    print('T_mean_null', T_mean_null)
    print('Tn_mean_null', Tn_mean_null)
    print('P', P, 'P_T', P_T, 'P_Tn', P_Tn)
    print('Theta', Theta, 'Theta_T', Theta_T, 'Theta_Tn', Theta_Tn)
    return  X_null, Xz_null, Theta, Theta_T, Theta_Tn, Sigma, Sigma_null_list, P, P_T, P_Tn


def null_model(timeseries: np.ndarray, I: list, num: int, tlen: int, nrunmax: int, Gaussian_version: bool, 
               Mutual_version: bool, model: int = None):
    """
    Compute the null model for mutual conditional information using Gaussian distribution or shuffled data.

    The function computes a null model for mutual information or correlation analysis, using either a multivariate
    normal distribution or shuffled time series data. It generates null data based on these models, then calculates
    mutual information or correlation measures for each run, and returns the mean values of key statistics like
    standard deviation, range, and maximum difference for the null model.

    Args:
        timeseries (np.ndarray): 
            A 2D array representing the time series data (shape: n_series, time_steps).
        I (list): 
            A list of three integers specifying the indices of rows in the time series that are of interest:
            - `I[0]`: row for X (first variable)
            - `I[1]`: row for Y (second variable)
            - `I[2]`: row used to sort and divide data into intervals.
        num (int): 
            Number of intervals for analysis.
        tlen (int): 
            Length of the time window to consider for analysis.
        nrunmax (int): 
            Maximum number of runs for the null model simulation.
        Gaussian_version (bool): 
            If True, uses a multivariate normal distribution as a null model; otherwise, uses shuffled data.
        Mutual_version (bool): 
            If True, performs mutual information analysis; otherwise, performs correlation analysis.
        model (int, optional): 
            Null model type (0-7). Defaults to None. Determines the specific model for the null simulation.

    Returns:
        tuple: 
            A tuple containing:
            - `Sigma_mean_null` (float): The mean standard deviation of mutual information or correlation values in the null model.
            - `T_mean_null` (float): The mean range of mutual information or correlation values in the null model.
            - `Tn_mean_null` (float): The mean maximum difference between consecutive mutual information or correlation values in the null model.

    Notes:
        - The function generates a null model using either a multivariate normal distribution (Gaussian model) or 
          randomized data (shuffling). It then calculates mutual information or correlation for the null model 
          and compares it to the original data.
        - Model types are as follows:
            - Z-Y-X-Z: 0
            - Z Y-X-Z: 1
            - Z-Y X-Z: 2
            - Z-Y-X Z: 3
            - Z Y X-Z: 4
            - Z-Y X Z: 5
            - Z Y-X Z: 6
            - Z Y X Z: 7
    """

    I = np.array(I)-1
    timeseries = timeseries[:, -tlen:]
    Cov = np.cov(timeseries)
    M = np.mean(timeseries, axis = 1)
    MIC = -1
    I2 = I
    if (Gaussian_version==True):
        MT = M[I]
        CovT = Cov[I]
        CovT = CovT[:,I]
        if model == 1:
            Cov[1,2] = 0
            Cov[2,1] = Cov[1,2]
        elif model == 2:
            Cov[0,1] = 0
            Cov[1,0] = Cov[0,1]
        elif model == 3:
            Cov[0,2] = 0
            Cov[2,0] = Cov[0,2]
        elif model == 4:
            Cov[1,2] = 0
            Cov[2,1] = Cov[1,2]
            Cov[0,1] = 0
            Cov[1,0] = Cov[0,1]
        elif model == 5:
            Cov[0,1] = 0
            Cov[1,0] = Cov[0,1]
            Cov[0,2] = 0
            Cov[2,0] = Cov[0,2]
        elif model == 6:
            Cov[1,2] = 0
            Cov[2,1] = Cov[1,2]
            Cov[0,2] = 0
            Cov[2,0] = Cov[0,2]
        elif model == 7:
            Cov[1,2] = 0
            Cov[2,1] = Cov[1,2]
            Cov[0,1] = 0
            Cov[1,0] = Cov[0,1]
            Cov[0,2] = 0
            Cov[2,0] = Cov[0,2]
        mult_dist = sp.stats.multivariate_normal(mean = MT, cov = CovT)
        I2 = [0,1,2]
      
    Sigma_null_list = []
    T_null_list = []
    Tn_null_list = []
    
    for n in range(nrunmax):
        if(Gaussian_version==False): 
            null_timeseries = np.array(timeseries).copy()
            np.random.shuffle(null_timeseries[I2[2], :])
            
        elif(Gaussian_version==True):    
            null_timeseries = mult_dist.rvs(tlen)
            null_timeseries = np.transpose(null_timeseries)
            
        if(Mutual_version==False):
            X_null, Xz_null, Sigma_null, T_null, Tn_null = correlation_analysis_continuous(null_timeseries, I2, num, tlen)
            
        elif(Mutual_version==True):
            X_null, Xz_null, MIC_null, Sigma_null, T_null, Tn_null = mutual_information_analysis_continuous(null_timeseries, I2, num, tlen)
            
        Sigma_null_list.append(Sigma_null)
        T_null_list.append(T_null)
        Tn_null_list.append(Tn_null)
        
    Sigma_null_list = np.array(Sigma_null_list)
    T_null_list = np.array(T_null_list)
    Tn_null_list = np.array(Tn_null_list)
    Sigma_mean_null = np.mean(Sigma_null_list)
    T_mean_null = np.mean(T_null_list)
    Tn_mean_null = np.mean(Tn_null_list)
    
    return Sigma_mean_null, T_mean_null, Tn_mean_null


def Theta_score_null_model(timeseries: np.ndarray, I: list, num: int, tlen: int, nrunmax: int, 
                           Gaussian_version: bool = True, Mutual_version: bool = True, extended: bool = False):
    """
    Calculate Theta scores and statistical significance from null model simulations for mutual conditional information.

    This function computes Theta scores to quantify the difference between the original data and a null model
    generated from either a multivariate normal distribution or shuffled data. The null model simulations are used
    to assess the statistical significance of mutual information or correlation measures. It returns multiple Theta
    scores, including those for the original data's standard deviation, range, and maximum difference between
    consecutive values, along with associated p-values indicating their significance.

    Args:
        timeseries (np.ndarray): 
            A 2D array representing the time series data (shape: n_series, time_steps).
        I (list): 
            A list of indices for rows in the time series that are of interest:
            - `I[0]`: row for X (first variable)
            - `I[1]`: row for Y (second variable)
            - `I[2]`: row used to sort and divide data into intervals.
        num (int): 
            Number of intervals for analysis.
        tlen (int): 
            Length of the time window to consider for analysis.
        nrunmax (int): 
            Maximum number of runs for the null model simulation.
        Gaussian_version (bool, optional): 
            If True, uses a multivariate normal distribution as a null model; otherwise, uses shuffled data. Defaults to True.
        Mutual_version (bool, optional): 
            If True, performs mutual information analysis; otherwise, performs correlation analysis. Defaults to True.
        extended (bool, optional): 
            If True, performs extended mutual information analysis; otherwise, performs standard mutual information analysis. Defaults to False.

    Returns:
        tuple: 
            A tuple containing the following:
            - `X` (numpy.ndarray): Original input data for analysis.
            - `Xz` (numpy.ndarray): Sorted input data for analysis.
            - `Xz_null` (numpy.ndarray): Sorted null model input data.
            - `MIC` (float): Mutual information or correlation value for the original data.
            - `Theta` (float): Theta score, a measure of the difference between the original data and the null model.
            - `Theta_T` (float): Theta score for the range of the original data.
            - `Theta_Tn` (float): Theta score for the maximum difference between consecutive values in the original data.
            - `Sigma` (float): Standard deviation of mutual information values for the original data.
            - `Sigma_null_list` (numpy.ndarray): List of standard deviations of mutual information values from null model simulations.
            - `P` (float): P-value indicating the statistical significance of the Theta score.
            - `P_T` (float): P-value indicating the statistical significance of the Theta_T score.
            - `P_Tn` (float): P-value indicating the statistical significance of the Theta_Tn score.

            If `extended=True`, the following additional values are returned:
            - `MINDY` (numpy.ndarray): Extended mutual information values.
            - `MI1` (numpy.ndarray): First set of mutual information values.
            - `MI2` (numpy.ndarray): Second set of mutual information values.
            - `Corr_Sigma` (numpy.ndarray): Correlation sigma values.

    Notes:
        - This function allows for analysis using either standard or extended mutual information methods. 
        - The null model is computed using either a Gaussian distribution (if `Gaussian_version=True`) or shuffled data (if `Gaussian_version=False`).
        - The `Theta` score compares the original data's properties (standard deviation, range, and maximum consecutive differences) to those of the null model, with statistical significance assessed via p-values.
    """

    I = np.array(I)-1
    timeseries = timeseries[:, -tlen:]
    Cov = np.cov(timeseries)
    M = np.mean(timeseries, axis = 1)
    MIC = -1

    if (Mutual_version == True):
        if extended == False:
            X, Xz, MIC, Sigma, T, Tn = mutual_information_analysis_continuous(timeseries, I, num, tlen)
        else: 
            X, Xz, MIC, Corr_Sigma, Sigma, T, Tn, MINDY, MI1, MI2 = mutual_information_analysis_continuous_extended(timeseries, I, num, tlen)
    elif (Mutual_version == False):
        X, Xz, Sigma, T, Tn = correlation_analysis_continuous(timeseries, I, num, tlen)
    X_null, Xz_null, Theta, Theta_T, Theta_Tn, Sigma, Sigma_null_list, P,P_T,P_Tn = null_model_results(M, Cov, timeseries, I, \
                                                                                                       num, tlen, Sigma, T, Tn, nrunmax, \
                                                                                                       Gaussian_version, Mutual_version)
    if extended == False: 
        return X, Xz, Xz_null, MIC, Theta, Theta_T, Theta_Tn, Sigma, Sigma_null_list, P, P_T, P_Tn
    
    else:
        return X, Xz, Xz_null, MIC, Theta, Theta_T, Theta_Tn, Sigma, Sigma_null_list, P, P_T, P_Tn, MINDY, MI1, MI2, Corr_Sigma


def freedman_diaconis(data: np.ndarray, returnas: str = "bins"):
    """
    Compute the optimal bin width for a histogram using the Freedman-Diaconis rule.

    This function calculates the optimal number of bins or the bin width for a histogram 
    based on the Freedman-Diaconis rule, which takes into account the interquartile range 
    (IQR) of the data and the number of data points.

    The formula for the bin width is:
        bin width = (2 * IQR) / N^(1/3)
    where IQR is the interquartile range, and N is the number of data points.

    Args:
        data (np.ndarray): 
            Input data for which the bin width or number of bins is computed. 
            It should be a 1D numpy array or a sequence of numerical values.
        returnas (str, optional): 
            If "bins", returns the number of bins for the histogram. If "width", returns the bin width. Defaults to "bins".

    Returns:
        result (int or float): 
            The computed result based on the specified `returnas` parameter:
            - If `returnas="bins"`, returns the estimated number of bins for a histogram.
            - If `returnas="width"`, returns the optimal bin width for a histogram.

    Raises:
        ValueError: 
            If `returnas` is not one of ["bins", "width"].

    Notes:
        - The Freedman-Diaconis rule is often used for determining the optimal bin width in histograms.
        - This method is robust to outliers due to the use of the interquartile range (IQR).
    """

    data = np.asarray(data, dtype = np.float_)
    IQR  = sp.stats.iqr(data, rng = (25, 75), scale="raw", nan_policy = "omit")
    N    = data.size
    bw   = (2*IQR)/np.power(N, 1/3)

    if returnas=="bins":
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int(np.round(datrng / bw) + 1)
        
    else:
        result = bw
        
    return(result)


def mch_approximation(samples: np.ndarray, dlamda: np.ndarray):
    """
    Make a Monte Carlo Histogram (MCH) approximation step for the Ising model.

    This function performs an approximation step based on the Monte Carlo Histogram (MCH) method 
    for the Ising model. The MCH method computes predicted values for the mean observables of the 
    Ising model by considering changes in the parameters and their impact on the energy.

    Args:
        samples (np.ndarray): 
            An array of Ising model samples, where each row corresponds to a sample, and each column represents a spin in the system.
        dlamda (np.ndarray): 
            An array of changes in the parameters (e.g., magnetic field, interaction strengths) of the Ising model.
    
    Returns:
        predsisj (np.ndarray): 
            Predicted values for the mean Ising model observables. The values are constrained to lie within the range [-1, 1].

    Raises:
        AssertionError: 
            If the predicted values (`predsisj`) are found to be outside the valid range of [-1, 1].

    Notes:
        - The function uses the `calc_observables` function to compute the observables from the samples.
        - The `dlamda` array represents changes in model parameters, which affect the energy of the system.
        - The predicted values are normalized with a partition function (`ZFraction`), calculated using the `logsumexp` function for numerical stability.
        - The predicted values are checked to ensure they remain within the valid physical range of [-1, 1].
    """

    dE = calc_observables(samples).dot(dlamda)
    ZFraction = len(dE) / np.exp(logsumexp(-dE))
    predsisj = ( calc_observables( samples )*np.exp(-dE)[:,None] ).mean(0) * ZFraction  
    assert not (np.any(predsisj<-1.00000001) or
        np.any(predsisj>1.000000001)),"Predicted values are beyond limits, (%1.6f,%1.6f)"%(predsisj.min(), predsisj.max())
    
    return predsisj
    
    
def learn_settings(i: int):
    """
    Determine settings based on the iteration counter.

    This function calculates the settings for the learning process at a given iteration, where the 
    settings are determined by an exponential decay based on the iteration index `i`. The values 
    are used for controlling the maximum allowed change in parameters (`maxdlamda`) and a 
    multiplicative factor (`eta`) that adjusts the parameter updates.

    Args:
        i (int): 
            The iteration counter. As the counter increases, the settings (`maxdlamda` and `eta`) 
            are updated using an exponential decay formula.

    Returns:
        dict: 
            A dictionary containing two key-value pairs:
            - 'maxdlamda' (float): The maximum allowed change in any given parameter, computed 
              using an exponential decay based on the iteration.
            - 'eta' (float): The multiplicative factor, computed similarly, that adjusts the 
              changes in the parameters based on the observed error.

    Notes:
        - Both `maxdlamda` and `eta` are computed using the formula `math.exp(-i/5.) * 0.5`, 
          where `i` is the current iteration index.
        - The values of `maxdlamda` and `eta` decrease exponentially with the number of iterations, 
          implying a gradual reduction in the rate of learning.
    """

    out = {'maxdlamda':math.exp(-i/5.)*.5,'eta':math.exp(-i/5.)*.5}

    return out
     
# The following section require the package Coniii to be runned.
# But due to instability of the coniii package we opted for an alternative solution
# However, if coniii is imported the following functions work
'''
def inverse_ising_null_model(X:np.ndarray, Y:np.ndarray, Z:np.ndarray, normalised:bool = False, model:str = 'pseudo', hypothesis:int = None):
    """
    Compute the inverse Ising null model for given Ising model parameters.

    Parameters
    ----------
    X : :numpy.ndarray
        Binary data for the X variable.
    Y : :numpy.ndarray
        Binary data for the Y variable.
    Z : :numpy.ndarray
        Binary data for the Z variable.
    normalised : bool, optional
        If True, normalize the computed standard deviations.
    model : str
        Type of null model to use. Options: 'pseudo', 'MCH', 'MPF'.
    hypothesis : int
        Specify a hypothesis to constrain certain coupling terms. Options: 1, 2, 3.

    Returns
    ----------
    sigmaij : float
        Standard deviation of the difference between observed and null model correlations for X and Y.

    Notes
    ----------
    - For binary data represented by indicator matrices X, Y, and Z.
    - Model options: 'pseudo' (gradient descent), 'MCH' (Monte Carlo History), 'MPF' (Mean-Field Perturbation).
    - Hypothesis options: 1, 2, 3, to constrain specific coupling terms.
    - Standard deviation is computed for the difference between observed and null model correlations.
    - N = number of spins, t = time to converge (t<<T (iteration))
    - n = number of samples, R = a real value
    - pseudo -> O(RN^{2}) for gradient descent and O(RN^{3}) for Hessian
    - MCH -> O(tnN^{2})
    - MPF -> O(RGN^{2}) and G ~ 2^{N} (number of neighbors)
    """    

    X = 2*X - 1
    Y = 2*Y - 1
    Z = 2*Z - 1
    vec = np.array([X, Y, Z])
    size = len(vec[0])
    sample = vec.T

    n = 3  # system size
    h = np.random.normal(scale = .1, size = n)           # random couplings
    J = np.random.normal(scale = .1, size = n*(n-1)//2)  # random fields
    hJ = np.concatenate((h, J))
    p = ising_eqn_3_sym.p(hJ)  # probability distribution of all states p(s)
    sisjTrue = ising_eqn_3_sym.calc_observables(hJ)  # exact means and pairwise correlations
    sisj = coni.pair_corr(sample, concat=True)

    if model == 'pseudo':
        solver = coni.Pseudo(sample)
        params = solver.solve()
        
    elif model == 'MCH':
        calc_observables = coni.define_ising_helper_functions()[1]
        solver = coni.MCH(sample, sample_size = 1_000, rng = np.random.RandomState(0), \
                     mch_approximation = mch_approximation)
        params = solver.solve(maxiter = 30, custom_convergence_f = learn_settings, \
                     n_iters = 500, burn_in = 1_000)
    
    elif model == 'MPF':
        solver = coni.MPF(sample)
        params = solver.solve()
        
    hx = params[0]
    hy = params[1]
    hz = params[2]
    Jxy = params[3]
    Jxz = params[4]
    Jyz = params[5]
    if hypothesis == 1:
        Jxz = Jyz = 0
    elif hypothesis == 2:
        Jxz = 0
    elif hypothesis == 3:
        Jyz = 0
    Hamilton = lambda sigmax, sigmay, sigmaz: -Jxy*sigmax*sigmay -Jxz*sigmax*sigmaz -Jyz*sigmay*sigmaz -hx*sigmax -hy*sigmay -hz*sigmaz
    H = np.zeros((2,2,2))
    prob = np.zeros((2,2,2))
    for i in [0,1]:
        for j in [0,1]:
            for k in [0,1]:
                H[i,j,k] = Hamilton(i,k,j)
    Z = np.sum(np.exp(-H))
    for i in [0,1]:
        for j in [0,1]:
            for k in [0,1]:
                prob[i,j,k] = np.exp(-H[i,j,k])/Z
    
    probij = np.zeros(2)
    probik = np.zeros(2)
    probjk = np.zeros(2)
    probij[0], probij[1] = np.sum(prob[:,0,:]), np.sum(prob[:,1,:])
    probik[0], probik[1] = np.sum(prob[:,:,0]), np.sum(prob[:,:,1])
    probjk[0], probjk[1] = np.sum(prob[0,:,:]), np.sum(prob[1,:,:])
    probi = np.zeros(2)
    probj = np.zeros(2)
    probk = np.zeros(2)
    for i in range(probi.shape[0]):
        probi[i] = np.sum(prob[:,i,:])
        probj[i] = np.sum(prob[:,:,i])
        probk[i] = np.sum(prob[i,:,:])
    Cij = np.zeros((2))
    Cik = np.zeros((2))
    Cjk = np.zeros((2))
    for i in range(probi.shape[0]):
        Cij[i] = np.sum(probik[i]*np.log(probik[i]))+\
                 np.sum(probjk[i]*np.log(probjk[i]))-\
                 np.sum(prob*np.log(prob))-\
                 np.sum(probk[i]*np.log(probk[i]))
        Cik[i] = np.sum(probij[i]*np.log(probij[i]))+\
                 np.sum(probjk[i]*np.log(probjk[i]))-\
                 np.sum(prob*np.log(prob))-\
                 np.sum(probj[i]*np.log(probj[i]))
        Cjk[i] = np.sum(probij[i]*np.log(probij[i]))+\
                 np.sum(probik[i]*np.log(probik[i]))-\
                 np.sum(prob*np.log(prob))-\
                 np.sum(probi[i]*np.log(probi[i]))
    Iij = np.sum(Cij)
    Iik = np.sum(Cik)
    Ijk = np.sum(Cjk)
    if normalised == False:
        sigmaij = np.sqrt(np.sum((probij*(Cij-Iij*np.ones(len(Cij)))**2)))
        sigmaik = np.sqrt(np.sum((probik*(Cik-Iik*np.ones(len(Cik)))**2)))
        sigmajk = np.sqrt(np.sum((probjk*(Cjk-Ijk*np.ones(len(Cjk)))**2)))
    else:
        sigmaij = np.sqrt(np.sum((probij*(Cij-Iij*np.ones(len(Cij)))**2)))/Iij
        sigmaik = np.sqrt(np.sum((probik*(Cik-Iik*np.ones(len(Cik)))**2)))/Iik
        sigmajk = np.sqrt(np.sum((probjk*(Cjk-Ijk*np.ones(len(Cjk)))**2)))/Ijk
    
    return sigmaij


def inverse_ising2(X:np.ndarray, Y:np.ndarray, model:str = 'pseudo'):
    """
    Compute the inverse Ising model for given Ising model parameters in a simplified 2-spin system.

    Parameters
    ----------
    X : numpy.ndarray
        Binary data for the X variable.
    Y: numpy.ndarray
        Binary data for the Y variable.
    model : str
        Type of model to use. Options: 'pseudo', 'MCH', 'MPF'.

    Returns
    ----------
    hx : float
        Fields for X in the Ising model
    hy : float
        Fields for Y in the Ising model
    Jxy : float
        Coupling between X and Y in the Ising model

    Notes
    ----------
    - For binary data represented by indicator matrices X and Y in a 2-spin system.
    - Model options: 'pseudo' (gradient descent), 'MCH' (Monte Carlo History), 'MPF' (Mean-Field Perturbation).
    - Couplings and fields are inferred based on the selected model.
    - N = number of spins, t = time to converge (t<<T (iteration))
    - n = number of samples, R = a real value
    - pseudo -> O(RN^{2}) for gradient descent and O(RN^{3}) for Hessian
    - MCH -> O(tnN^{2})
    - MPF -> O(RGN^{2}) and G ~ 2^{N} (number of neighbors)
    """
    
    X = 2*X - 1
    Y = 2*Y - 1
    vec = np.array([X, Y])
    size = len(vec[0])
    sample = vec.T

    n = 2  # system size
    h = np.random.normal(scale = .1, size = n)           # random couplings
    J = np.random.normal(scale = .1, size = n*(n-1)//2)  # random fields
    hJ = np.concatenate((h, J))
    p = ising_eqn_2_sym.p(hJ)  # probability distribution of all states p(s)
    sisjTrue = ising_eqn_2_sym.calc_observables(hJ)  # exact means and pairwise correlations
    sisj = coni.pair_corr(sample, concat=True)

    if model == 'pseudo':
        solver = coni.Pseudo(sample)
        params = solver.solve()
        
    elif model == 'MCH':
        calc_observables = coni.define_ising_helper_functions()[1]
        solver = coni.MCH(sample, sample_size = 1_000, rng = np.random.RandomState(0), \
                     mch_approximation = mch_approximation)
        params = solver.solve(maxiter = 30, custom_convergence_f = learn_settings, \
                     n_iters = 500, burn_in = 1_000)
    
    elif model == 'MPF':
        solver = coni.MPF(sample)
        params = solver.solve()
        
    hx = params[0]
    hy = params[1]
    Jxy = params[2]
    
    return hx, hy, Jxy
'''