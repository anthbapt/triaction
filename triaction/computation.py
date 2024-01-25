import numpy as np
from scipy.sparse import csc_matrix
from scipy.stats import iqr
import scipy.stats as sps
from scipy.special import kl_div

def create_node_edge_incidence_matrix(edge_list:list):
    """Create a node-edge incidence matrix B from a given edge list.

    Parameters
    ----------
    edge_list : list of tuples (i, j) where i < j
        The list of edges (i, j).

    Returns
    -------
    B : numpy.ndarray of shape (n_nodes, n_edges)
        The node-edge incidence matrix.
    
    """
    # Get the number of edges
    n_edges = len(edge_list)

    # If the number of edges is zero, return None
    if n_edges == 0:
        return None
    
    # Create the node-edge incidence matrix
    b_ij = [-1] * n_edges + [1] * n_edges # the (i,j)-th element of B in a sequence
    row_i = [e[0]-1 for e in edge_list] + [e[1]-1 for e in edge_list] # the row indices (i) of the (i, j)-th element
    col_j = [l for l in range(n_edges)] * 2 # the column indices (i) of the (i, j)-th element
    B = csc_matrix((np.array(b_ij), (np.array(row_i), np.array(col_j))), dtype=np.int8)

    return B.toarray()

def extract_by_std(data:np.ndarray, n_std:float=3.):
    """Extract the data within a given number of standard deviations from its mean.
    
    Parameters
    ----------
    data : numpy.ndarray of shape (n_observations, )
        The data.
    n_std : float, optional
        (default = 3.0)
        The number of standard deviations to extract.
    
    Returns
    -------
    data_min : float
        The minimum value in the extracted range of data .
    data_max : float
        The maximum value in the extracted range of data.
    
    Raises
    ------
    ValueError
        If the data is not of shape (n_observations, ).
    
    """
    # Check the shape of the data
    if data.ndim > 2 or data.ndim == 0:
        raise ValueError('The data must be of shape (n_observations, ).')
    
    # Compute the mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    
    data_min = np.min(data[np.abs(data - mean) <= n_std * std])
    data_max = np.max(data[np.abs(data - mean) <= n_std * std])

    # Compute the core range
    return data_min, data_max

def freedman_diaconis_rule(data:np.ndarray, power:float=1./3., factor:float=2.):
    """Compute the number of bins using the Freedman-Diaconis rule.
    
    Parameters
    ----------
    data : numpy.ndarray of shape (n_variables, n_observations)
        The data matrix.
    power : float, optional
        (default = 1./3.)
        The power of the number of observations in the denominator.
    factor : float, optional
        (default = 2.)
        The factor to multiply the width of bins.
    
    Returns
    -------
    bins_edges : numpy.ndarray of shape (n_bins,) or list of numpy.ndarray of shape (n_bins_{i},) [i = 1, ..., n_variables]
        The bins edges.
    
    Raises
    ------
    ValueError
        If the data is not a one-dimensional or two-dimensional array.
    
    """
    if data.ndim == 0 or data.ndim > 2:
        raise ValueError('The data must be a one-dimensional or two-dimensional array.')
    
    # Convert the data to a two-dimensional array
    data = np.atleast_2d(data)

    # Get the number of observations and variables
    n_variables, n_observations = data.shape

    # Initialise the bins edges
    bin_edges = []

    # For each variable, compute the bins edges and append them to the list
    for i in range(n_variables):
        # Compute the interquartile range
        IQR = iqr(data[i], rng=(25, 75)) 
        
        # Compute the width of bins according to the Freedman-Diaconis rule
        width = (factor * IQR) / np.power(n_observations, power)

        # Check if the width is positive
        assert(width > 0.)

        # Generate the bins
        x_abs_max = np.max(np.abs(data[i]))
        idx_upper = int(x_abs_max / width + 1)
        _bin_edges = np.linspace(-idx_upper * width, idx_upper * width, 2 * idx_upper + 1)

        bin_edges.append(_bin_edges)
    
    if n_variables == 1:
        bin_edges = bin_edges[0]
    
    return bin_edges

def _check_data_shape(data:np.ndarray):
    """Check the shape of the data.
    
    Parameters
    ----------
    data : numpy.ndarray
        The data.
    
    Raises
    ------
    ValueError
        If the data is not of shape (n_observations,) or (1, n_observations).
    
    """
    data = np.squeeze(data)

    if data.ndim > 2 or data.ndim == 0:
        raise ValueError(
            'The data must be of shape (n_observations,) or (1, n_observations).'
        )

def _generate_bins(data:np.ndarray, bins:str or np.ndarray or int or list or tuple):
    """Generate the bins.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_observations,) or (1, n_observations) or (n_variables, n_observations)
        The data.
    bins : str or a sequence of int or int or list or tuple
        The number of bins or the method to compute the number of bins.
            - 'fd' (str) : The number of bins is computed using the Freedman-Diaconis rule.
            - n (int) : The number of bins for the variable.
            - list or tuple (list or tuple of numpy.ndarray) : The bin edges for each variable.
    
    Returns
    -------
    _bins : list of numpy.ndarray of shape (n_bins_{i},) [i = 1, ..., n_variables]
        The bins edges for each variable.
    n_bins : int
        The number of bins for each variable.
    
    Raises
    ------
    ValueError
        If the bin edges are not monotonically increasing.
        If the length of bins is not equal to the number of variables.
        If the bin type is invalid.
    """
    # Check the shape of the data
    if data.ndim == 1:
        data = np.atleast_2d(data)

    # Get the number of variables
    n_variables = data.shape[0]

    if n_variables == 1:
        if isinstance(bins, str) and bins == 'fd':
            bin_edges = freedman_diaconis_rule(data.flatten())
            n_bins = len(bin_edges) - 1
        
        elif isinstance(bins, np.ndarray) and np.squeeze(bins).ndim == 1:
            # Check if the bin edges are sorted
            if np.any(np.diff(bins) <= 0):
                raise ValueError(
                    'The bin edges must be monotonically increasing.'
                )
            bin_edges = bins
            n_bins = len(bin_edges) - 1
        
        elif isinstance(bins, int):
            max_amp = np.max(np.abs(data))
            bin_edges = np.linspace(-max_amp, max_amp, bins+1)
            n_bins = bins
        else:
            raise ValueError('Invalid bins.')
        
    elif n_variables > 1:
        if isinstance(bins, str) and bins == 'fd':
            bin_edges = [
                freedman_diaconis_rule(data[i]) for i in range(n_variables)
            ]
            n_bins = [len(bin_edges[i]) - 1 for i in range(n_variables)]
        
        elif isinstance(bins, list):
            bin_edges = []
            n_bins = []
            if len(bins) != n_variables:
                raise ValueError(
                    'The length of bins must be equal to the number of variables.'
                )
            for idx, item in enumerate(bins):
                if isinstance(item, np.ndarray) and np.squeeze(item).ndim == 1:
                    if np.any(np.diff(item) <= 0):
                        raise ValueError(
                            'The bin edges must be monotonically increasing.'
                        )
                    bin_edges.append(item)
                    n_bins.append(len(item) - 1)
                
                elif isinstance(item, str) and item == 'fd':
                    bin_edges.append(
                        freedman_diaconis_rule(data[idx])
                    )
                    n_bins.append(len(bin_edges[idx]) - 1)
                
                elif isinstance(item, int):
                    bin_edges.append(
                        np.linspace(
                            np.min(data[idx]), 
                            np.max(data[idx]), 
                            num=item
                        )
                    )
                    n_bins.append(item)
                else:
                    raise ValueError("Invalid bin type at index", idx, ".")
            
        elif isinstance(bins, int):
            bin_edges = [
                np.linspace(
                    -np.max(np.abs(data[d])), 
                    np.max(np.abs(data[d])), 
                    num=bins+1
                ) for d in range(n_variables)
            ]
            n_bins = [
                len(bin_edges[d]) - 1 for d in range(n_variables)
            ]
    else:
        raise ValueError('Invalid data.')
    
    return bin_edges, n_bins

def estimate_pdf(data:np.ndarray, bins:str or int or np.ndarray='fd', method:str='kde'):
    """Estimate the probability density function of the data.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_observations,) or (1, n_observations)
        The data.
    bins : str or a sequence of int or int, optional
        (default = None)
        The number of bins or the method to compute the number of bins.
            - 'fd' (str) : The number of bins is computed using the Freedman-Diaconis rule.
            - n (int) : The number of bins for the variable.
            - numpy.ndarray : The bin edges for the variable.
    method : str, optional
        (default = 'kde')
        The method to estimate the probability density function.
            - 'hist' : The probability density function is estimated by the histogram method.
            - 'kde' : The probability density function is estimated by the kernel density estimation.
    
    Returns
    -------
    pdf : numpy.ndarray of shape (n_bins, )
        The estimated probability density function.
    x : numpy.ndarray of shape (n_bins, )
        The x values.
    
    """
    # Check the shape of the data
    _check_data_shape(data)

    # Get the number of bins based on Freedman-Diaconis rule or user input
    bin_edges, _ = _generate_bins(data, bins)

    if method == 'hist':
        # Create histogram
        pdf, bin_edges = np.histogram(
            data, 
            bins=bin_edges,
            density=True
        )

        # Calculate the x values corresponding to each bin
        x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    elif method == 'kde':
        # Create kernel density estimator
        kernel = sps.gaussian_kde(data)
        
        # Calculate the x values corresponding to each bin
        x = bin_edges
        
        # Calculate the probability density function
        pdf = kernel(x)
    
    else:
        raise ValueError(
            "Invalid method: method must be either 'hist' or 'kde'."
        )
    
    return pdf, x

def estimate_pdf_joint(data:np.ndarray, bins:str or np.ndarray or int or list or tuple='fd', method:str='kde'):
    """Estimate the joint probability density function of the data by the histogram method.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_variables, n_observations)
        The data.
    bins : str or a sequence of int or int or list, optional
        (default = None)
        The number of bins or the method to compute the number of bins.
            - 'fd' (str) : The number of bins is computed using the Freedman-Diaconis rule.
            - n_{1}, n_{2}, ..., n_{n_variables} (sequence of int) : The number of bins for each variable.
            - n (int): The number of bins for all variables.
            - list (list of numpy.ndarray) : The bin edges for each variable.
    method : str, optional
        (default = 'kde')
        The method to estimate the probability density function.
            - 'hist' : The probability density function is estimated by the histogram method.
            - 'kde' : The probability density function is estimated by the kernel density estimation.
    
    Returns
    -------
    pdf_joint : numpy.ndarray of shape (n_bins_{1}, ..., n_bins_{n_variables})
        The estimated probability density function.
    x : list of numpy.ndarray of shape (n_bins_{i},) [i = 1, ..., n_variables]
        The x values of the corresponding bins.
        
    """
    # Check the shape of the data
    _check_data_shape(data)

    # Get the number of variables
    if data.ndim == 1:
        n_variables = 1
        return estimate_pdf(data, bins=bins, method=method)
    elif data.ndim == 2:
        n_variables = data.shape[0]
    else:
        raise ValueError(
            'The data must be of shape (n_variables, n_observations).'
        )

    # Generate the bins
    bin_edges, _ = _generate_bins(data, bins)
    
    # Estimation by the histogram method
    if method == 'hist':
        # Create histogram
        pdf_joint, x = np.histogramdd(
            data,
            bins=bin_edges,
            density=True
        )
        
        # Calculate the x values corresponding to each bin
        for d in range(n_variables):
            x[d] = 0.5 * (x[d][1:] + x[d][:-1])

    # Estimation by the kernel density estimation   
    elif method == 'kde':
        # Create kernel density estimator
        kernel = sps.gaussian_kde(data)

        x = bin_edges

        # Generate the grid
        grid = np.meshgrid(*bin_edges)

        # Calculate the probability density function at each point of the grid
        x_eval = np.vstack([d.flatten() for d in grid])
        pdf_joint = kernel(x_eval).reshape(grid[0].shape)
    
    else:
        raise ValueError(
            "Invalid method: method must be either 'hist' or 'kde'."
        )
    
    return pdf_joint, x

def estimate_pmf(data:np.ndarray, bins:str or int or np.ndarray='fd', method:str='kde'):
    """Estimate the probability mass function of the data by the histogram method.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_observations,) or (1, n_observations)
        The data.
    bins : str or a sequence of int or int, optional
        (default = None)
        The number of bins or the method to compute the number of bins.
        - 'fd' : The number of bins is computed using the Freedman-Diaconis rule.
        - n : The number of bins for the variable.
        - np.ndarray : The bin edges for the variable.
    method : str, optional
        (default = 'kde')
        The method to estimate the probability density function.
        - 'hist' : The probability density function is estimated by the histogram method.
        - 'kde' : The probability density function is estimated by the kernel density estimation.
    
    Returns
    -------
    pmf : numpy.ndarray of shape (n_bins, )
        The estimated probability mass function of the data.
    X : numpy.ndarray of shape (n_bins, )
        The bin centers for variables.
    
    """
    # Check the shape of the data
    _check_data_shape(data)

    data = np.atleast_2d(data)
    
    if data.ndim == 2 and data.shape[0] == 1:
        # Get the number of bins based on Freedman-Diaconis rule or user input
        bin_edges, _ = _generate_bins(data, bins)
        
        pmf, x = estimate_pdf(data, bins=bin_edges, method=method)
        pmf = pmf / np.sum(pmf)
        
    else:
        raise ValueError(
                'The data must be of shape (n_observations,) or (1, n_observations). '
            )
    
    return pmf, x

def estimate_pmf_joint(data:np.ndarray, bins:str or np.ndarray or int or list or tuple='fd', method:str='kde'):
    """Estimate the joint probability mass function of the data.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_variables, n_observations)
        The data.
    bins : str or a sequence of int or int or list, optional
        (default = None)
        The number of bins or the method to compute the number of bins.
        - 'fd' : The number of bins is computed using the Freedman-Diaconis rule.
        - n_{1}, n_{2}, ..., n_{n_variable} : The number of bins for each variable.
        - n : The number of bins for all variables.
        - list : The bin edges for each variable.
    method : str, optional
        (default = 'kde')
        The method to estimate the probability density function.
        - 'hist' : The probability density function is estimated by the histogram method.
        - 'kde' : The probability density function is estimated by the kernel density estimation.
    
    Returns
    -------
    pmf_joint : numpy.ndarray of shape (n_bins^{1}, n_bins^{2}, ..., n_bins^{n_variables})
        The estimated probability mass function of the data.
    x : list of numpy.ndarray of shape (n_bins^{i},}) [i = 1, ..., n_variables]
        The x values of the corresponding bins.
    
    """
    # Check the shape of the data
    _check_data_shape(data)

    # Get the number of variables
    if data.ndim == 1:
        n_variables = 1
    else:
        data = np.atleast_2d(data)
        n_variables = data.shape[0]

    # Generate the bins
    bin_edges, _ = _generate_bins(data, bins)
    
    pdf_joint, x = estimate_pdf_joint(data, bins=bin_edges, method=method)
    pmf_joint = pdf_joint / np.sum(pdf_joint)
    
    return pmf_joint, x

def estimate_mutual_information(X:np.ndarray, Y:np.ndarray, bins:str or int='fd', pmf_method:str='kde', method:str='kl-div'):
    """Calculate the mutual information between X and Y.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_observations,) or (1, n_observations)
        The data.
    Y : numpy.ndarray of shape (n_observations,) or (1, n_observations)
        The data.
    bins : str or a sequence of int or int, optional
        (default = None)
        The number of bins or the method to compute the number of bins.
        - 'fd' : The number of bins is computed using the Freedman-Diaconis rule.
        - n : The number of bins for the variable. 
    pmf_method : str, optional
        (default = 'kde')
        The method to estimate the probability mass function.
        - 'hist' : The probability density function is estimated by the histogram method.
        - 'kde' : The probability density function is estimated by the kernel density estimation.
    method : str, optional
        (default = 'kl-div')
        The method to estimate the mutual information.
        - 'kl-div' : The mutual information is calculated by the Kullback-Leibler divergence.
        - 'entropy' : The mutual information is calculated from entropies.
    
    Returns
    -------
    mi : float
        The mutual information between X and Y.
    
    """

    # Check the shape of the data
    _check_data_shape(X)
    _check_data_shape(Y)

    if X.shape != Y.shape:
        raise ValueError(
            'The shape of X and Y must be equal.'
        )
    
    # Shape the data
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    
    # Initialise the mutual information
    mi = 0.

    # Estimate the probability mass functions of X and Y
    pmf_X, _ = estimate_pmf(X, bins=bins, method=pmf_method)
    pmf_Y, _ = estimate_pmf(Y, bins=bins, method=pmf_method)

    # Estimate the joint probability mass function of X and Y
    pmf_XY, _ = estimate_pmf_joint(np.vstack((X, Y)), bins=bins, method=pmf_method)


    if method == 'kl-div':
        # Generate the outer product of the probability mass functions of X and Y
        outer_pmfXY = np.outer(pmf_X, pmf_Y)

        # Calculate the conditional mutual information
        for j in range(pmf_X.shape[0]): # Loop over X
            for k in range(pmf_Y.shape[0]): # Loop over Y
                mi += kl_div(pmf_XY[k, j], outer_pmfXY[j, k])
    
    elif method == 'entropy':
        # Calculate the entropies
        H_X = sps.entropy(pmf_X)
        H_Y = sps.entropy(pmf_Y)
        H_XY = sps.entropy(pmf_XY.flatten())

        # Calculate the mutual information
        mi = H_XY - H_X - H_Y

    else:
        raise ValueError(
            "Invalid method: method must be either 'kl-div' or 'entropy'."
        )

    return mi

def covariance(data:np.ndarray):
    """Calculate the covariance matrix.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_nodes, n_timesteps, n_samples)
        The time series data.

    Returns
    -------
    cov_np : numpy.ndarray of shape (n_samples, n_nodes * n_nodes)
        The covariance matrix.
    
    """
    # Check the shape of the data
    if data.ndim != 3:
        raise ValueError(
            'The data must be of shape (n_nodes, n_timesteps, n_variables).'
        )
    
    # Get the number of samples
    _, _, n_samples = data.shape

    # Calculate the covariance matrix
    cov_np = np.zeros((n_samples, 9))

    # Loop over samples
    for i in range(n_samples):
        X_i = data[:, :, i]
        cov_i = np.cov(X_i)
        cov_np[i] = cov_i.flatten()
    
    return cov_np

def conditional_correlation(X:np.ndarray, Y:np.ndarray, Z:np.ndarray, bins:str or int='fd'):
    """Compute the conditional variance.
    
    Parameter
    ---------
    X : numpy.ndarray of shape (n_observations,) or (1, n_observations)
        The time series data.
    Y : numpy.ndarray of shape (n_observations,) or (1, n_observations)
        The time series data. 
    Z : numpy.ndarray of shape (n_observations,) or (1, n_observations)
        The time series data. (condition)
    bins : int or str, optional
        (default = 'fd')
        The number of bins or the method to compute the number of bins.
        - 'fd' : Freedman-Diaconis rule
        - n (int) : The number of bins for the variable.
    
    Returns
    -------
    corr_cond : numpy.ndarray of shape (n_bins,)
        The conditional correlation.
    z : numpy.ndarray of shape (n_bins,)
        The bin values of the conditional variable.
    corr_cond_err : numpy.ndarray of shape (n_bins,)
        The standard error of the conditional correlation.
    
    """
    # Check the shape of the data
    _check_data_shape(X)
    _check_data_shape(Y)
    _check_data_shape(Z)
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(
            'The shape of X, Y, and Z must be equal.'
        )
    
    # Shape the data
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    Z = np.atleast_2d(Z)

    if Z.ndim == 2 and Z.shape[0] == 1:
        # Generate the bins
        bin_edges_z, n_bins = _generate_bins(Z, bins)

        # Initialise the conditional correlation
        cond_corr = np.zeros(n_bins)
        cond_corr_stderr = np.zeros(n_bins)
        
        # Get the digitised data
        Z_dig = np.digitize(Z, bin_edges_z)

        # Calculate the conditional correlation
        for j in range(n_bins): # Loop over bins
            if np.sum(Z_dig == j) < n_bins:
                cond_corr[j] = np.nan
            else:
                cond_corr[j] = np.corrcoef(X[Z_dig == j], Y[Z_dig == j])[0, 1]
                cond_corr_stderr[j] = np.sqrt((1 - cond_corr[j]**2) / (np.sum(Z_dig == j) - 2))
        
        # Calculate the bin values
        z = 0.5 * (bin_edges_z[1:] + bin_edges_z[:-1])
    
    else:
        raise ValueError(
            'The data must be of shape (n_observations,) or (n_observations, 1). '
        )
    
    return cond_corr, z, cond_corr_stderr

def conditional_mutual_information(X:np.ndarray, Y:np.ndarray, Z:np.ndarray, bins:str or int='fd', pmf_method:str='kde', method:str='kl-div'):
    """Calculate the conditional mutual information between X and Y given Z.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_observations, ) or (1, n_observations)
        The data.
    Y : numpy.ndarray of shape (n_observations, ) or (1, n_observations)
        The data.
    Z : numpy.ndarray of shape (n_observations, ) or (1, n_observations)
        The data to be conditioned.
    bins : str or a sequence of int or int, optional
        (default = None)
        The number of bins or the method to compute the number of bins.
        - 'fd' : The number of bins is computed using the Freedman-Diaconis rule.
        - n_1, n_2, ..., n_n : The number of bins for each variable.
        - n : The number of bins for all variables.
    pmf_method : str, optional
        (default = 'kde')
        The method to estimate the probability mass function.
        - 'hist' : The probability density function is estimated by the histogram method.
        - 'kde' : The probability density function is estimated by the kernel density estimation.
    method : str, optional
        (default = 'kl-div')
        The method to estimate the mutual information.
        - 'kl-div' : The mutual information is calculated by the Kullback-Leibler divergence.
        - 'entropy' : The mutual information is calculated from entropies.

    Returns
    -------
    cmi : numpy.ndarray of shape (n_bins,)
        The conditional mutual information between X and Y given Z=z for each z in Z.
    pmf_Z : numpy.ndarray of shape (n_bins,)
        The probability mass function of Z.
    z : numpy.ndarray of shape (n_bins,)
        The z values of the corresponding bins.
    
    """
    # Check the shape of the data
    _check_data_shape(X)
    _check_data_shape(Y)
    _check_data_shape(Z)
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(
            'The shape of X, Y, and Z must be equal.'
        )
    
    # Shape the data
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    Z = np.atleast_2d(Z)

    if Z.ndim == 2 and Z.shape[0] == 1:
        # Generate the bins
        bin_edges_z, n_bins = _generate_bins(Z, bins)

        # Get the digitised data
        Z_dig = np.digitize(Z, bin_edges_z)
        pmf_Z, _ = estimate_pmf(Z, bins=bin_edges_z, method=pmf_method)

        # Initialise the conditional mutual information        
        cmi = np.zeros(n_bins)

        # Loop over bins
        for i in range(n_bins):
            if np.sum(Z_dig == i) < n_bins:
                cmi[i] = np.nan
            else:
                cmi[i] = estimate_mutual_information(
                    X[Z_dig == i], 
                    Y[Z_dig == i], 
                    bins=bins, 
                    pmf_method=pmf_method, 
                    method=method
                )
            
        # Calculate the bin values
        z = 0.5 * (bin_edges_z[1:] + bin_edges_z[:-1])
    
    else:
        raise ValueError(
            'The data to be conditioned on must be of shape (n_observations,) or (1, n_observations).'
        )
    
    return cmi, pmf_Z, z