from pyitlib.discrete_random_variable import _sanitise_array_input, _autocreate_alphabet
from pyitlib.discrete_random_variable import _isnan, _vstack_pad, _map_observations_to_integers
from pyitlib.discrete_random_variable import _verify_alphabet_sufficiently_large, _determine_number_additional_empty_bins
from pyitlib.discrete_random_variable import _remove_counts_at_fill_value, _estimate_probabilities
from pyitlib.discrete_random_variable import information_mutual
import pyitlibnew as drv
import scipy as sp
import numpy as np
import warnings
import os


def entropy_joint(X, base = 2, fill_value = -1, estimator = 'ML', Alphabet_X = None):
    """
    Calculate the joint entropy of a set of random variables.

    Args:
        X (array-like): Input data.
        base (float, optional): Logarithm base for entropy calculation. Default is 2.
        fill_value (float, optional): Fill value for missing data. Default is -1.
        estimator (str, optional): Estimator for probability distribution. Default is 'ML'.
        Alphabet_X (array-like, optional): Alphabet for X. If not provided, it will be automatically created.

    Returns:
        tuple: A tuple containing probability values and joint entropy.
    """

    X, fill_value_X = _sanitise_array_input(X, fill_value)
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(Alphabet_X, fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X, fill_value_X)   
    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if np.any(_isnan(X)) :
        raise ValueError("arg X contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)) :
        raise ValueError("arg Alphabet_X contains NaN values")
    if _isnan(fill_value_X) :
        raise ValueError("fill value for arg X is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0) :
        raise ValueError("arg base not a positive real-valued scalar")
    S, fill_value = _map_observations_to_integers((X, Alphabet_X), (fill_value_X, fill_value_Alphabet_X))
    X, Alphabet_X = S
    X = np.reshape(X, (-1, X.shape[-1]))
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
    _verify_alphabet_sufficiently_large(X, Alphabet_X, fill_value)
    for i in range(X.shape[0]) :
        X = X[:, X[i].argsort(kind='mergesort')]
    B = np.any(X[:, 1:] != X[:, :-1], axis = 0)
    I = np.append(np.where(B), X.shape[1]-1)
    L = np.diff(np.append(-1, I))
    alphabet_X = X[:, I]
    if estimator != 'ML':
        n_additional_empty_bins = \
            _determine_number_additional_empty_bins(L, alphabet_X, Alphabet_X, fill_value)
    else:
        n_additional_empty_bins = 0
    L, _ = _remove_counts_at_fill_value(L, alphabet_X, fill_value)
    if not np.any(L) :
        return np.float64(np.NaN)
    P, P_0 = _estimate_probabilities(L, estimator, n_additional_empty_bins)
    vd = np.unique(alphabet_X[-1])
    H = dict.fromkeys(vd)
    Proba = dict.fromkeys(vd)
    for i in vd :
        pos = np.asarray(alphabet_X[-1] == i).nonzero()[0]
        H0 = n_additional_empty_bins * P_0 * -np.log2(P_0 + np.spacing(0)) / np.log2(base)
        H[i] = H0
        Proba[i] = np.sum(P[pos])
        for j in pos :
            H[i] += -np.sum((P[j]/Proba[i])*np.log2(P[j] + np.spacing(0)), axis = -1) / np.log2(base)
            
    return Proba, H


def information_mutual_conditional(X, Y, Z, cartesian_product = False, base = 2, 
                                   fill_value = -1, estimator = 'ML', Alphabet_X = None, 
                                   Alphabet_Y = None, Alphabet_Z = None):
    """
    Calculate the mutual conditional information between three sets of random variables.

    Args:
        X (array-like): Input data for variable X.
        Y (array-like): Input data for variable Y.
        Z (array-like): Input data for variable Z.
        cartesian_product (bool, optional): Whether to compute the cartesian product. Default is False.
        base (float, optional): Logarithm base for entropy calculation. Default is 2.
        fill_value (float, optional): Fill value for missing data. Default is -1.
        estimator (str, optional): Estimator for probability distribution. Default is 'ML'.
        Alphabet_X (array-like, optional): Alphabet for X. If not provided, it will be automatically created.
        Alphabet_Y (array-like, optional): Alphabet for Y. If not provided, it will be automatically created.
        Alphabet_Z (array-like, optional): Alphabet for Z. If not provided, it will be automatically created.

    Returns:
        tuple: A tuple containing probability values, conditional entropies, and mutual conditional information.
    """
    
    X, fill_value_X = _sanitise_array_input(X, fill_value)
    Y, fill_value_Y = _sanitise_array_input(Y, fill_value)
    Z, fill_value_Z = _sanitise_array_input(Z, fill_value)
    
    if Alphabet_X is not None:
        Alphabet_X, fill_value_Alphabet_X = _sanitise_array_input(Alphabet_X, fill_value)
        Alphabet_X, _ = _autocreate_alphabet(Alphabet_X, fill_value_Alphabet_X)
    else:
        Alphabet_X, fill_value_Alphabet_X = _autocreate_alphabet(X, fill_value_X)
    if Alphabet_Y is not None:
        Alphabet_Y, fill_value_Alphabet_Y = _sanitise_array_input(Alphabet_Y, fill_value)
        Alphabet_Y, _ = _autocreate_alphabet(Alphabet_Y, fill_value_Alphabet_Y)
    else:
        Alphabet_Y, fill_value_Alphabet_Y = _autocreate_alphabet(Y, fill_value_Y)
    if Alphabet_Z is not None:
        Alphabet_Z, fill_value_Alphabet_Z = _sanitise_array_input(Alphabet_Z, fill_value)
        Alphabet_Z, _ = _autocreate_alphabet(Alphabet_Z, fill_value_Alphabet_Z)
    else:
        Alphabet_Z, fill_value_Alphabet_Z = _autocreate_alphabet(Z, fill_value_Z)
    if X.size == 0:
        raise ValueError("arg X contains no elements")
    if Y.size == 0:
        raise ValueError("arg Y contains no elements")
    if Z.size == 0:
        raise ValueError("arg Z contains no elements")
    if np.any(_isnan(X)) :
        raise ValueError("arg X contains NaN values")
    if np.any(_isnan(Y)) :
        raise ValueError("arg Y contains NaN values")
    if np.any(_isnan(Z)) :
        raise ValueError("arg Z contains NaN values")
    if Alphabet_X.size == 0:
        raise ValueError("arg Alphabet_X contains no elements")
    if np.any(_isnan(Alphabet_X)) :
        raise ValueError("arg Alphabet_X contains NaN values")
    if Alphabet_Y.size == 0:
        raise ValueError("arg Alphabet_Y contains no elements")
    if np.any(_isnan(Alphabet_Y)) :
        raise ValueError("arg Alphabet_Y contains NaN values")
    if Alphabet_Z.size == 0:
        raise ValueError("arg Alphabet_Z contains no elements")
    if np.any(_isnan(Alphabet_Z)) :
        raise ValueError("arg Alphabet_Z contains NaN values")
    if _isnan(fill_value_X) :
        raise ValueError("fill value for arg X is NaN")
    if _isnan(fill_value_Y) :
        raise ValueError("fill value for arg Y is NaN")
    if _isnan(fill_value_Z) :
        raise ValueError("fill value for arg Z is NaN")
    if X.shape[:-1] != Alphabet_X.shape[:-1]:
        raise ValueError("leading dimensions of args X and Alphabet_X do not match")
    if Y.shape[:-1] != Alphabet_Y.shape[:-1]:
        raise ValueError("leading dimensions of args Y and Alphabet_Y do not match")
    if Z.shape[:-1] != Alphabet_Z.shape[:-1]:
        raise ValueError("leading dimensions of args Z and Alphabet_Z do not match")
    if not cartesian_product and (X.shape != Y.shape or X.shape != Z.shape) :
        raise ValueError("dimensions of args X, Y, Z do not match")
    if cartesian_product and (X.shape[-1] != Y.shape[-1] or
                              X.shape[-1] != Z.shape[-1]) :
        raise ValueError("trailing dimensions of args X, Y, Z do not match")
    if not (np.isscalar(base) and np.isreal(base) and base > 0) :
        raise ValueError("arg base not a positive real-valued scalar")
        
    S, fill_value = _map_observations_to_integers((X, Alphabet_X,
                                                   Y, Alphabet_Y,
                                                   Z, Alphabet_Z),
                                                  (fill_value_X,
                                                   fill_value_Alphabet_X,
                                                   fill_value_Y,
                                                   fill_value_Alphabet_Y,
                                                   fill_value_Z,
                                                   fill_value_Alphabet_Z))
    X, Alphabet_X, Y, Alphabet_Y, Z, Alphabet_Z = S
    X = np.reshape(X, (-1, X.shape[-1]))
    Y = np.reshape(Y, (-1, Y.shape[-1]))
    Z = np.reshape(Z, (-1, Z.shape[-1]))
    vd = np.unique(Z)
    Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
    Alphabet_Y = np.reshape(Alphabet_Y, (-1, Alphabet_Y.shape[-1]))
    Alphabet_Z = np.reshape(Alphabet_Z, (-1, Alphabet_Z.shape[-1]))
    for i in range(Z.shape[0]) :        
        Pa, Ha = entropy_joint(np.vstack((X[i], Z[i])), base, fill_value,
                            estimator, _vstack_pad((Alphabet_X[i], Alphabet_Z[i]), fill_value))
        Pb, Hb = entropy_joint(np.vstack((Y[i], Z[i])), base, fill_value, 
                            estimator, _vstack_pad((Alphabet_Y[i], Alphabet_Z[i]), fill_value))
        Pc, Hc = entropy_joint(np.vstack((X[i], Y[i], Z[i])), base, fill_value,
                            estimator, _vstack_pad((Alphabet_X[i], Alphabet_Y[i],Alphabet_Z[i]), fill_value))
        Pd, Hd = entropy_joint(Z[i], base, fill_value, estimator, Alphabet_Z[i])
        H = dict.fromkeys(vd)
        for i in H.keys() :   
            H[i] = Ha[i] + Hb[i] - Hc[i] - Hd[i]
        I = 0
        for i in H.keys() :
            I += Pa[i]*H[i]
            
    return Pa, H, I


def std(X, Y, Z, normalised=False):
    """
    Calculate the standard deviation of the mutual conditional information.

    Args:
        X (array-like): Input data for variable X.
        Y (array-like): Input data for variable Y.
        Z (array-like): Input data for variable Z.
        normalised (bool, optional): Whether to normalize the result. Default is False.

    Returns:
        float: The standard deviation of the mutual conditional information.
    """

    P, H, I = information_mutual_conditional(X, Y, Z)
    P_array = np.zeros(len(P))
    H_array = np.zeros(len(H))
    compt = 0
    for i in H.keys() :
        P_array[compt] = P[i]
        H_array[compt] = H[i]
        compt += 1
    Id = np.sum(P_array*H_array)
    if normalised == False:
        sdt_vd = np.sqrt(np.sum(P_array*((H_array - Id)**2)))
    else:
        sdt_vd = np.sqrt(np.sum(P_array*((H_array - Id)**2)))/Id

    return sdt_vd


def corr(X, Y, Z):
    """
    Calculate the correlation coefficient between the mutual conditional information and the variable Z.

    Args:
        X (array-like): Input data for variable X.
        Y (array-like): Input data for variable Y.
        Z (array-like): Input data for variable Z.

    Returns:
        float: The correlation coefficient.
    """

    P, H, I = information_mutual_conditional(X, Y, Z)
    P_array = np.zeros(len(P))
    H_array = np.zeros(len(H))
    vd_array = np.zeros(len(H))
    compt = 0
    for i in H.keys() :
        P_array[compt] = P[i]
        H_array[compt] = H[i]
        vd_array[compt] = i
        compt += 1
    std_vd = std(X, Y, Z)
    a = np.sum(vd_array*P_array*H_array)
    b = np.sum(P_array*H_array)*np.sum(P_array*vd_array)
    corr_vd = (a-b)/(np.std(H_array)*np.std(vd_array))

    return corr_vd


def pele_mele(X, Y, Z):
    """
    Calculate various metrics including T, Tn, entropy, and probabilities.

    Args:
        X (array-like): Input data for variable X.
        Y (array-like): Input data for variable Y.
        Z (array-like): Input data for variable Z.

    Returns:
        tuple: A tuple containing T, Tn, entropy, probabilities, and value differences.
    """

    P, H, I = information_mutual_conditional(X, Y, Z)
    P_array = np.zeros(len(P))
    H_array = np.zeros(len(H))
    compt = 0
    for i in H.keys() :
        P_array[compt] = P[i]
        H_array[compt] = H[i]
        compt += 1

    MI_XY = information_mutual(X, Y)   
    MIz_tilde = H_array - np.ones(len(H_array))*MI_XY
    gamma = (H_array-np.ones(len(H_array))*np.sum(H_array*P_array))**2
    gamma = gamma/np.sum(gamma)
    MM = (gamma)*(P_array)
    MM = MM/np.sum(MM)
    Mn = max(abs(MM[0:len(MM)-1] - MM[1:len(MM)]))

    sig_dif_abs_dis = abs(H_array-np.ones(len(H_array))*np.sum(H_array*P_array))
    sig_dif_rel_dis = abs(H_array[0:len(H_array)-1] - H_array[1:len(H_array)])
    T = max(sig_dif_abs_dis)
    Tn = max(sig_dif_rel_dis)

    return T, Tn, H_array, P_array, np.array(list(H.keys()))-1


def preprocess_null_model(X, Y, Z, positive = False, discretized = False, tol = None):
    """
    Preprocess data for null model calculations.

    Args:
        X (array-like): Input data for variable X.
        Y (array-like): Input data for variable Y.
        Z (array-like): Input data for variable Z.
        positive (bool, optional): Whether to adjust data for positivity. Default is False.
        discretized (bool, optional): Whether data is discretized. Default is False.
        tol (int, optional): Tolerance for rounding when discretized. Default is None.

    Returns:
        tuple: A tuple containing preprocessed X, Y, and Z.
    """

    if discretized == True :
        X = np.round(X, tol)
        Y = np.round(Y, tol)
#        Z = np.round(Z, tol)
    if positive == True :
        minima_X = np.min(X)
        minima_Y = np.min(Y)
        minima_Z = np.min(Z)
        minima = min(minima_X, minima_Y, minima_Z)
        if minima < 0:
            if tol == None :
                X = X + abs(minima)
                Y = Y + abs(minima)
                Z = Z + abs(minima)
            else :
                X = np.round(X + abs(minima), tol)
                Y = np.round(Y + abs(minima), tol)
#                Z = np.round(Z + abs(minima), tol)
            
    return X, Y, Z    


def preprocess(X, Y, Z, num = 5, tol = 0, XY_sort = False):
    """
    Preprocess data for null model calculations.

    Args:
        X (array-like): Input data for variable X.
        Y (array-like): Input data for variable Y.
        Z (array-like): Input data for variable Z.
        num (int, optional): Number of splits. Default is 5.
        tol (int, optional): Tolerance for rounding. Default is 0.
        XY_sort (bool, optional): Whether to sort X and Y. Default is False.

    Returns:
        tuple: A tuple containing preprocessed X, Y, and spatial coordinate values.
    """    

    Z_s = Z.argsort()
    Z = Z[Z_s]
    X = X[Z_s]
    Y = Y[Z_s]    
    
    sp = np.array_split(Z, num)
    compt = 1
    for k in range(len(sp)):
        sp[k] = compt*np.ones(len(sp[k]))
        compt += 1
    sp = np.hstack(sp)
    if XY_sort == True:
        X_sort = X
        Y_sort = Y
        X, Y, Z = preprocess_null_model(X, Y, Z, positive = True, discretized = True, tol = tol)
        return X, Y, sp, X_sort, Y_sort
    else:
        X, Y, Z = preprocess_null_model(X, Y, Z, positive = True, discretized = True, tol = tol)
        return X, Y, sp

    
# the tol parameter should have the same value for both preprocess_null_model and preprocess
def gaussian_null_model(X, Y, Z, num, model=0, normalised=False, n=100, prec=2):
    """
    Compute the Gaussian null model for the mutual conditional information.
    Z-Y-X-Z: 0    Z Y-X-Z: 1    Z-Y X-Z: 2    Z-Y-X Z: 3
    Z Y X-Z: 4    Z-Y X Z: 5    Z Y-X Z: 6    Z Y X Z: 7

    Args:
        X (array-like): Input data for variable X.
        Y (array-like): Input data for variable Y.
        Z (array-like): Input data for variable Z.
        num (int): Number of splits.
        model (int, optional): Null model type. Default is 0.
        normalised (bool, optional): Whether to normalize the result. Default is False.
        n (int, optional): Number of iterations for null model calculations. Default is 100.
        prec (int, optional): Precision for rounding. Default is 2.

    Returns:
        tuple: A tuple containing mean and standard deviation of the null model statistics.
    """

    vec = np.array([X, Y, Z])
    size = len(vec[0])

    if model == 0:
        Corr = np.corrcoef(vec)
        Cov = np.cov(vec)
    elif model == 1:
        Cov = np.cov(vec)
        Cov[1,2] = 0
        Cov[2,1] = Cov[1,2]
    elif model == 2:
        Cov = np.cov(vec)
        Cov[0,1] = 0
        Cov[1,0] = Cov[0,1]
    elif model == 3:
        Cov = np.cov(vec)
        Cov[0,2] = 0
        Cov[2,0] = Cov[0,2]
    elif model == 4:
        Cov = np.cov(vec)
        Cov[1,2] = 0
        Cov[2,1] = Cov[1,2]
        Cov[0,1] = 0
        Cov[1,0] = Cov[0,1]
    elif model == 5:
        Cov = np.cov(vec)
        Cov[0,1] = 0
        Cov[1,0] = Cov[0,1]
        Cov[0,2] = 0
        Cov[2,0] = Cov[0,2]
    elif model == 6:
        Cov = np.cov(vec)
        Cov[1,2] = 0
        Cov[2,1] = Cov[1,2]
        Cov[0,2] = 0
        Cov[2,0] = Cov[0,2]
    elif model == 7:
        Cov = np.cov(vec)
        Cov[1,2] = 0
        Cov[2,1] = Cov[1,2]
        Cov[0,1] = 0
        Cov[1,0] = Cov[0,1]
        Cov[0,2] = 0
        Cov[2,0] = Cov[0,2]
        
    M = np.array([np.mean(i) for i in vec])
    list_std = list()
    list_T = list()
    list_Tn = list()
    for i in range(n) :
        mult_dist = sp.stats.multivariate_normal(mean = M, cov = Cov)
        mul = mult_dist.rvs(size)
        mul.shape
        X = mul[:, 0][0:]
        Y = mul[:, 1][0:]
        Z = mul[:, 2][0:]
        X, Y, Z = preprocess(X, Y, Z, num = num)
        if normalised == False:
            std_temp = std(X, Y, Z)
            T_temp, Tn_temp, MIz, Pz, valz = pele_mele(X, Y, Z)
        else:
            std_temp = std(X, Y, Z, normalised = True)
        list_corr.append(corr_temp)
        list_std.append(std_temp)
        list_T.append(T_temp)
        list_Tn.append(Tn_temp)
    mean_std = np.mean(list_std)
    std_std = np.std(list_std)
    mean_T = np.mean(list_T)
    T_std = np.std(list_T)
    mean_Tn = np.mean(list_Tn)
    Tn_std = np.std(list_Tn)
    
    return np.round(mean_std,prec), std_std, np.round(mean_T,prec), T_std, np.round(mean_Tn,prec), Tn_std