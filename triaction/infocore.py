from sklearn.feature_selection import mutual_info_regression
from coniii.ising_eqn import ising_eqn_3_sym
from coniii.ising_eqn import ising_eqn_2_sym
import sklearn.preprocessing
import coniii as coni
import scipy as sp
import numpy as np
import math

### Change the convention for I, then it starts at 1 and we move I-1 in null model and several other functions
### now it should be better to start from 0 and remove the I-1 everywhere.


def timeseries_quantile(timeseries, I, num, tlen):
    """
    Computes quantiles of time series data.

    Args:
    timeseries (numpy.ndarray): 2D array representing the time series data.
    I (list): List of indices specifying rows of interest in the time series.
    num (int): Number of quantiles to compute.
    tlen (int): Length of the time window considered for quantile computation.

    Returns:
    tuple: A tuple containing three arrays - X_sort, Y_sort, and sp.
        - X_sort (numpy.ndarray): Sorted values of the first row of interest.
        - Y_sort (numpy.ndarray): Sorted values of the second row of interest.
        - sp (numpy.ndarray): Quantile labels corresponding to the third row of interest.

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

    
def mutual_information_analysis_continuous(timeseries, I, num, tlen):
    """
    Analyses mutual information between continuous variables in a time series.

    Args:
    timeseries (numpy.ndarray): 2D array representing the time series data.
    I (list): List of indices specifying rows of interest in the time series.
    num (int): Number of intervals for analysis.
    tlen (int): Length of the time window considered for analysis.

    Returns:
    tuple: A tuple containing six elements - MI, MIz, MIC, Sigma, T, and Tn.
        - MI (float): Mutual information between the first and second rows of interest.
        - MIz (numpy.ndarray): Mutual information values for each interval.
        - MIC (float): Mean mutual information across intervals.
        - Sigma (float): Standard deviation of mutual information values.
        - T (float): Range of mutual information values across intervals.
        - Tn (float): Maximum difference between consecutive mutual information values.

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
    Tn = np.max(abs(MIz[0:num-1])-MIz[1:num])
    
    return MI, MIz, MIC, Sigma, T, Tn


def mutual_information_analysis_continuous_extended(timeseries, I, num, tlen):
    """
    Extended analysis of mutual information between continuous variables in a time series.

    Args:
    timeseries (numpy.ndarray): 2D array representing the time series data.
    I (list): List of indices specifying rows of interest in the time series.
    num (int): Number of intervals for analysis.
    tlen (int): Length of the time window considered for analysis.

    Returns:
    tuple: A tuple containing ten elements - MI, MIz, MIC, Corr_Sigma, Sigma, T, Tn, MINDY, MI1, and MI2.
        - MI (float): Mutual information between the first and second rows of interest.
        - MIz (numpy.ndarray): Mutual information values for each interval.
        - MIC (float): Mean mutual information across intervals.
        - Corr_Sigma (float): Pearson correlation between mean Zn values and MIz.
        - Sigma (float): Standard deviation of mutual information values.
        - T (float): Range of mutual information values across intervals.
        - Tn (float): Maximum difference between consecutive mutual information values.
        - MINDY (float): Absolute difference between mutual information values of the first and last third of the data.
        - MI1 (float): Mutual information of the first third of the data.
        - MI2 (float): Mutual information of the last third of the data.
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
    Tn = np.max(abs(MIz[0:num-1])-MIz[1:num])
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


def mindy(timeseries, I, tlen):
    """
    Computes the MINDy measure between continuous variables.
    Wang, K., Saito, M., Bisikirska, B. et al. Nat Biotechnol 27, 829–837 (2009).
    https://doi.org/10.1038/nbt.1563

    Args:
    timeseries (numpy.ndarray): 2D array representing the time series data.
    I (list): List of indices specifying rows of interest in the time series.
    tlen (int): Length of the time window considered for analysis.

    Returns:
    tuple: A tuple containing three elements - MINDY, MI1, and MI2.
        - MINDY (float): Mutually Information Normalized Difference Yield.
        - MI1 (float): Mutual information of the first third of the data.
        - MI2 (float): Mutual information of the last third of the data.

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
    Xn = np.zeros((dtlen,2))
    for i in range(3):
        Xn[:,0] = X_sort[i*dtlen:(i+1)*dtlen]
        y = Y_sort[i*dtlen:(i+1)*dtlen]
        mi = mutual_info_regression(Xn, y, discrete_features = False)
        MINDYz[i] = mi[0]
    MI1 = MINDYz[0]
    MI2 = MINDYz[-1]
    MINDY = np.abs(MI1 - MI2)
        
    return MINDY, MI1, MI2


def mutual_information(timeseries, I, num, tlen):
    """
    Computes mutual information measures between continuous variables in a time series.

    Args:
    timeseries (numpy.ndarray): 2D array representing the time series data.
    I (list): List of indices specifying rows of interest in the time series.
    num (int): Number of intervals for analysis.
    tlen (int): Length of the time window considered for analysis.

    Returns:
    tuple: A tuple containing three elements - MI, MIz, and MIC.
        - MI (float): Mutual information between the first and second rows of interest.
        - MIz (numpy.ndarray): Mutual information values for each interval.
        - MIC (float): Conditional mutual information across intervals.
        
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


def sigma(timeseries, I, num, tlen, nrunmax = 5000):
    """
    Computes the standard deviation of mutual information values between continuous variables in a time series.
    Define as the Sigma measure.

    Args:
    timeseries (numpy.ndarray): 2D array representing the time series data.
    I (list): List of indices specifying rows of interest in the time series.
    num (int): Number of intervals for analysis.
    tlen (int): Length of the time window considered for analysis.
    nrunmax (int, optional): Maximum number of runs for null model simulation. Defaults to 5000.
    null (bool or int, optional): If False, only computes the standard deviation. If an integer, simulates a null model
                                  with the specified number of runs. Defaults to False.

    Returns:
    float or tuple: If null is False, returns Sigma (standard deviation of mutual information values).
                   If null is an integer, returns a tuple containing Sigma and Sigma_null (standard deviation of null model).
                   
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
            

def t(timeseries, I, num, tlen, nrunmax = 5000):
    """
    Computes the range of mutual information values between continuous variables in a time series.
    Define as the T measure.

    Args:
    timeseries (numpy.ndarray): 2D array representing the time series data.
    I (list): List of indices specifying rows of interest in the time series.
    num (int): Number of intervals for analysis.
    tlen (int): Length of the time window considered for analysis.
    nrunmax (int, optional): Maximum number of runs for null model simulation. Defaults to 5000.
    null (bool or int, optional): If False, only computes the range. If an integer, simulates a null model
                                  with the specified number of runs. Defaults to False.

    Returns:
    float or tuple: If null is False, returns T (range of mutual information values).
                   If null is an integer, returns a tuple containing T and T_null (range of null model).

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
    

def tn(timeseries, I, num, tlen, nrunmax = 5000):
    """
    Computes the maximum difference between consecutive mutual information values in a time series.
    Define as the Tn measure.

    Args:
    timeseries (numpy.ndarray): 2D array representing the time series data.
    I (list): List of indices specifying rows of interest in the time series.
    num (int): Number of intervals for analysis.
    tlen (int): Length of the time window considered for analysis.
    nrunmax (int, optional): Maximum number of runs for null model simulation. Defaults to 5000.
    null (bool or int, optional): If False, only computes the maximum difference.
                                  If an integer, simulates a null model with the specified number of runs. Defaults to False.

    Returns:
    float or tuple: If null is False, returns Tn (maximum difference between consecutive mutual information values).
                   If null is an integer, returns a tuple containing Tn and Tn_null (maximum difference of null model).

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

    Tn = np.max(abs(MIz[0:num-1])-MIz[1:num])
    if null == False:
        return Tn
    else:
        if type(null) == int:
            Sigma_null, T_null, Tn_null = null_model(timeseries, I, num, tlen, nrunmax, Gaussian_version = True,\
                                                                    Mutual_version = True, model = None)
            
            return Tn, Tn_null
    
        else:
            raise TypeError("Only integers or False accepted")
    

def correlation_analysis_continuous(timeseries, I, num, tlen):
    """
    Analyses the correlation between continuous variables in a time series.

    Args:
    timeseries (numpy.ndarray): 2D array representing the time series data.
    I (list): List of indices specifying rows of interest in the time series.
    num (int): Number of intervals for analysis.
    tlen (int): Length of the time window considered for analysis.

    Returns:
    tuple: A tuple containing five elements - C, Cz, Sigma, T, and Tn.
        - C (float): Mean correlation between the first and second rows of interest.
        - Cz (numpy.ndarray): Correlation values for each interval.
        - Sigma (float): Variance of correlation values.
        - T (float): Range of correlation values across intervals.
        - Tn (float): Maximum difference between consecutive correlation values.

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
    Tn = np.max(abs(Cz[0:num-1])-Cz[1:num])
    
    return C, Cz, Sigma, T, Tn


def null_model_results(M, Cov, timeseries, I, num, tlen, Sigma, T, Tn, nrunmax, Gaussian_version, Mutual_version, model = None):
    """
    Compute the results of the null model for mutual conditional information.

    Args:
    M (numpy.ndarray): Mean vector for the multivariate normal distribution.
    Cov (numpy.ndarray): Covariance matrix for the multivariate normal distribution.
    timeseries (numpy.ndarray): 2D array representing the time series data.
    I (list): List of indices specifying rows of interest in the time series.
    num (int): Number of intervals for analysis.
    tlen (int): Length of the time window considered for analysis.
    Sigma (float): Standard deviation of mutual information values in the original data.
    T (float): Range of mutual information values in the original data.
    Tn (float): Maximum difference between consecutive mutual information values in the original data.
    nrunmax (int): Maximum number of runs for null model simulation.
    Gaussian_version (bool): If True, uses a multivariate normal distribution as a null model.
    Mutual_version (bool): If True, performs mutual information analysis; otherwise, performs correlation analysis.
    model (int, optional): Null model type (0-7). Defaults to None.

    Returns:
    tuple: A tuple containing nine elements - X_null, Xz_null, Theta, Theta_T, Theta_Tn, Sigma,
           Sigma_null_list, P, P_T, and P_Tn.
        - X_null (float): Mean value of mutual information or correlation in null model.
        - Xz_null (numpy.ndarray): Values of mutual information or correlation for each interval in null model.
        - Theta (float): Normalized difference in standard deviation of the original and null models.
        - Theta_T (float): Normalized difference in range of the original and null models.
        - Theta_Tn (float): Normalized difference in maximum difference of the original and null models.
        - Sigma (float): Standard deviation of mutual information values in the original data.
        - Sigma_null_list (numpy.ndarray): Standard deviation of mutual information values for each run in null model.
        - P (float): Probability of observing a standard deviation greater than the original in the null model.
        - P_T (float): Probability of observing a range greater than the original in the null model.
        - P_Tn (float): Probability of observing a maximum difference greater than the original in the null model.

    Notes: Compute the Gaussian null model for the mutual conditional information.
            Z-Y-X-Z: 0    Z Y-X-Z: 1    Z-Y X-Z: 2    Z-Y-X Z: 3
            Z Y X-Z: 4    Z-Y X Z: 5    Z Y-X Z: 6    Z Y X Z: 7
            
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
    
    return  X_null, Xz_null, Theta, Theta_T, Theta_Tn, Sigma, Sigma_null_list, P, P_T, P_Tn


def null_model(timeseries, I, num, tlen, nrunmax, Gaussian_version, Mutual_version, model = None):
    """
    Compute the null model for mutual conditional information using Gaussian distribution or shuffled data.

    Args:
    timeseries (numpy.ndarray): 2D array representing the time series data.
    I (list): List of indices specifying rows of interest in the time series.
    num (int): Number of intervals for analysis.
    tlen (int): Length of the time window considered for analysis.
    nrunmax (int): Maximum number of runs for null model simulation.
    Gaussian_version (bool): If True, uses a multivariate normal distribution as a null model; otherwise, uses shuffled data.
    Mutual_version (bool): If True, performs mutual information analysis; otherwise, performs correlation analysis.
    model (int, optional): Null model type (0-7). Defaults to None.

    Returns:
    tuple: A tuple containing three elements - Sigma_mean_null, T_mean_null, and Tn_mean_null.
        - Sigma_mean_null (float): Mean standard deviation of mutual information values in null model.
        - T_mean_null (float): Mean range of mutual information values in null model.
        - Tn_mean_null (float): Mean maximum difference between consecutive mutual information values in null model.

    Notes: Compute the Gaussian null model for the mutual conditional information.
            Z-Y-X-Z: 0    Z Y-X-Z: 1    Z-Y X-Z: 2    Z-Y-X Z: 3
            Z Y X-Z: 4    Z-Y X Z: 5    Z Y-X Z: 6    Z Y X Z: 7
            
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


def Theta_score_null_model(timeseries, I, num, tlen, nrunmax, Gaussian_version = True, Mutual_version = True, extended = False):
    """
    Calculate Theta scores and statistical significance from null model simulations for mutual conditional information.

    Args:
    timeseries (numpy.ndarray): 2D array representing the time series data.
    I (list): List of indices specifying rows of interest in the time series.
    num (int): Number of intervals for analysis.
    tlen (int): Length of the time window considered for analysis.
    nrunmax (int): Maximum number of runs for null model simulation.
    Gaussian_version (bool, optional): If True, uses a multivariate normal distribution as a null model; 
                                       otherwise, uses shuffled data. Defaults to True.
    Mutual_version (bool, optional): If True, performs mutual information analysis; 
                                     otherwise, performs correlation analysis. Defaults to True.
    extended (bool, optional): If True, performs extended mutual information analysis; 
                               otherwise, performs standard mutual information analysis. Defaults to False.

    Returns:
    tuple: A tuple containing relevant information based on the analysis.
        - X (numpy.ndarray): Original input data for analysis.
        - Xz (numpy.ndarray): Sorted input data for analysis.
        - Xz_null (numpy.ndarray): Sorted null model input data.
        - MIC (float): Mutual information or correlation value for the original data.
        - Theta (float): Theta score, a measure of the difference between the original data and the null model.
        - Theta_T (float): Theta score for the range of the original data.
        - Theta_Tn (float): Theta score for the maximum difference between consecutive values in the original data.
        - Sigma (float): Standard deviation of mutual information values for the original data.
        - Sigma_null_list (numpy.ndarray): List of standard deviations of mutual information values from null model simulations.
        - P (float): P-value indicating the statistical significance of Theta score.
        - P_T (float): P-value indicating the statistical significance of Theta_T score.
        - P_Tn (float): P-value indicating the statistical significance of Theta_Tn score.

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


def freedman_diaconis(data, returnas = "bins"):
    """
    Compute the optimal bin width for a histogram using the Freedman-Diaconis rule.

    Args:
    data (numpy.ndarray): Input data for which the bin width is computed.
    returnas (str, optional): If "bins", returns the number of bins; if "width", returns the bin width. Defaults to "bins".

    Returns:
    int or float: The computed result based on the specified returnas parameter.
        - If returnas is "bins", returns the estimated number of bins for a histogram.
        - If returnas is "width", returns the optimal bin width for a histogram.

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


def mch_approximation(samples, dlamda):
    """
    Make a Monte Carlo Histogram (MCH) approximation step for the Ising model.

    Args:
    samples (numpy.ndarray): Array of Ising model samples.
    dlamda (numpy.ndarray): Array of changes in parameters for the Ising model.

    Returns:
    numpy.ndarray: Predicted values for the mean Ising model observables.

    Raises:
    AssertionError: If the predicted values are beyond the valid range [-1, 1].

    """
    dE = calc_observables(samples).dot(dlamda)
    ZFraction = len(dE) / np.exp(logsumexp(-dE))
    predsisj = ( calc_observables( samples )*np.exp(-dE)[:,None] ).mean(0) * ZFraction  
    assert not (np.any(predsisj<-1.00000001) or
        np.any(predsisj>1.000000001)),"Predicted values are beyond limits, (%1.6f,%1.6f)"%(predsisj.min(), predsisj.max())
    
    return predsisj
    
    
def learn_settings(i):
    """
    Determine settings based on the iteration counter.

    Args:
    i (int): Iteration counter.

    Returns:
    dict: A dictionary containing the following settings:
        - 'maxdlamda' (float): Maximum change allowed in any given parameter.
        - 'eta' (float): Multiplicative factor where d(parameter) = (error in observable) * eta.
    
    """
    return {'maxdlamda':math.exp(-i/5.)*.5,'eta':math.exp(-i/5.)*.5}
        

def inverse_ising_null_model(X, Y, Z, normalised = False, model = 'pseudo', hypothesis = None):
    """
    Compute the inverse Ising null model for given Ising model parameters.

    Args:
    X (array_like): Binary data for the X variable.
    Y (array_like): Binary data for the Y variable.
    Z (array_like): Binary data for the Z variable.
    normalised (bool, optional): If True, normalize the computed standard deviations.
    model (str, optional): Type of null model to use. Options: 'pseudo', 'MCH', 'MPF'.
    hypothesis (int, optional): Specify a hypothesis to constrain certain coupling terms. Options: 1, 2, 3.

    Returns:
    float: Standard deviation of the difference between observed and null model correlations for X and Y.

    Notes:
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
#    print('h',h)
    J = np.random.normal(scale = .1, size = n*(n-1)//2)  # random fields
#    print('J',J)
    hJ = np.concatenate((h, J))
    p = ising_eqn_3_sym.p(hJ)  # probability distribution of all states p(s)
#    print('p',p)
    sisjTrue = ising_eqn_3_sym.calc_observables(hJ)  # exact means and pairwise correlations
    sisj = coni.pair_corr(sample, concat=True)
#    print('sisj', sisj)

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


def inverse_ising2(X, Y, model = 'pseudo'):
    """
    Compute the inverse Ising model for given Ising model parameters in a simplified 2-spin system.

    Args:
    X (array_like): Binary data for the X variable.
    Y (array_like): Binary data for the Y variable.
    model (str, optional): Type of model to use. Options: 'pseudo', 'MCH', 'MPF'.

    Returns:
    tuple: Tuple containing the inferred couplings and fields for X and Y in the Ising model.

    Notes:
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