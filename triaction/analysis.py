from sklearn.metrics import mean_squared_error
from triaction.seaborn_grid import SeabornFig2Grid
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from itertools import groupby
from sklearn import tree
import triaction.infocore as ifc
import seaborn as sns
import scipy as sp
import numpy as np
import os


def func(x: np.ndarray, a: float, b: float):
    """
    Compute a linear function.

    This function computes the output of a simple linear equation `y = a * x + b`, where `a` is 
    the slope and `b` is the intercept. It applies this linear transformation element-wise to 
    the input array `x`.

    Args:
        x (numpy.ndarray): 
            The input data, a 1D or 2D array of numeric values over which the linear transformation 
            is applied.
        a (float): 
            The slope parameter of the linear equation.
        b (float): 
            The intercept parameter of the linear equation.

    Returns:
        numpy.ndarray: 
            The output values, computed as `y = a * x + b`. This will have the same shape as the 
            input array `x`.

    Notes:
        - This function assumes `x` is a `numpy.ndarray` and supports element-wise operations.
    """
    
    y = a * x + b
    return y



def fit(x: np.ndarray, y: np.ndarray):
    """
    Fit a curve to input data using a linear model.

    This function performs curve fitting to the provided input data `x` and `y` using a linear 
    function (`y = a * x + b`). It fits separate curves for the `x` and `y` values and returns 
    the fitted curves along both axes.

    Args:
        x (numpy.ndarray): 
            A 2D array where `x[0, :]` represents the independent variable values and `x[1, :]` 
            represents the dependent variable values for the x-axis curve fitting.
        y (numpy.ndarray): 
            A 2D array where `y[0, :]` represents the independent variable values and `y[1, :]` 
            represents the dependent variable values for the y-axis curve fitting.

    Returns:
        tuple: 
            A tuple containing:
                - x_fit (numpy.ndarray): A 2D array where the first row represents the fitted 
                  x-values and the second row represents the corresponding fitted y-values.
                - y_fit (numpy.ndarray): A 2D array where the first row represents the fitted 
                  y-values and the second row represents the corresponding fitted y-values.
    
    Notes:
        - The function uses the `curve_fit` function from the `scipy.optimize` module to fit 
          linear functions to the data.
        - The fitted curves are represented by `y = a * x + b` and are computed separately 
          for both `x` and `y` input data.
    """
    
    minima = min(np.min(x[0,:]), np.min(y[0,:]))
    maxima = max(np.max(x[1,:]), np.max(y[1,:]))
    t = np.linspace(minima, maxima, 100)
    popt_x, pcov_x = curve_fit(func, x[0,:], x[1,:])
    popt_y, pcov_y = curve_fit(func, y[0,:], y[1,:])
    
    x_fit = np.array([t, func(t, popt_x[0], popt_x[1])])
    y_fit = np.array([t, func(t, popt_y[0], popt_y[1])])
    
    return x_fit, y_fit


def opimise_kl(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """
    Optimize KL divergence between two conditional distributions based on a third variable.

    This function computes the Kullback-Leibler (KL) divergence between the conditional distributions 
    of variables `X` and `Y` conditioned on variable `Z`. It finds the value of `Z` that maximizes the 
    KL divergence between the two distributions and returns this optimized value.

    Args:
        X (numpy.ndarray): 
            Input data for variable X. The data should be in a 1D array.
        Y (numpy.ndarray): 
            Input data for variable Y. The data should be in a 1D array.
        Z (numpy.ndarray): 
            Input data for variable Z. The data should be in a 1D array, and the function uses this 
            variable to condition the distributions of `X` and `Y`.

    Returns:
        float: 
            The optimized conditional value of `Z` that maximizes the KL divergence between the 
            conditional distributions of `X` and `Y` given `Z`.

    Notes:
        - The function sorts the unique values of `Z` and computes the KL divergence between the 
          conditional distributions of `X` and `Y` at each threshold of `Z`.
        - The KL divergence is calculated using the `scipy.special.kl_div` function for the probability 
          distributions derived from `X` and `Y` conditioned on `Z`.
        - The result is the value of `Z` where the KL divergence is maximized, which is used as the 
          optimized conditional value.
    """
    
    val_Z = sorted(list(set(Z)))
    kl_list = list()
    for k in range(1, len(val_Z) - 1):
        X_sup = X[Z > val_Z[k]]
        Y_sup = Y[Z > val_Z[k]]
        X_inf = X[Z <= val_Z[k]]
        Y_inf = Y[Z <= val_Z[k]]
        inf_cond = np.array([X_inf, Y_inf])
        sup_cond = np.array([X_sup, Y_sup])
        fit_inf_cond, fit_sup_cond = fit(inf_cond, sup_cond)
        
        # Test positivity of the two fits for the KL divergence
        if (np.all(fit_inf_cond[1] >= 0) == False) and (np.all(fit_sup_cond[1] >= 0) == True):
            temp = np.where(fit_inf_cond[1] >= 0)
            fit_sup = fit_sup_cond[1][temp]
            fit_inf = fit_inf_cond[1][temp]
        elif (np.all(fit_sup_cond[1] >= 0) == False) and (np.all(fit_inf_cond[1] >= 0) == True):
            temp = np.where(fit_sup_cond[1] >= 0)
            fit_inf = fit_inf_cond[1][temp]
            fit_sup = fit_sup_cond[1][temp]
        elif (np.all(fit_inf_cond[1] >= 0) == False) and (np.all(fit_sup_cond[1] >= 0) == False):
            temp = max(np.min(np.where(fit_sup_cond[1] >= 0)), np.min(np.where(fit_inf_cond[1] >= 0)))
            fit_inf = fit_inf_cond[1][temp::]
            fit_sup = fit_sup_cond[1][temp::]
        elif (np.all(fit_inf_cond[1] >= 0) == True) and (np.all(fit_sup_cond[1] >= 0) == True):
            fit_inf = fit_inf_cond[1]
            fit_sup = fit_sup_cond[1]
        
        fit_inf = fit_inf / np.sum(fit_inf)
        fit_sup = fit_sup / np.sum(fit_sup)

        kl = sp.special.kl_div(fit_inf, fit_sup)
        kl_list.append(np.sum(kl))        
    
    cond_val = val_Z[np.argmax(kl_list)] + 1  # We start at the second value of val_Z
    
    return cond_val


def visualisation_conditioned(timeseries, I, num, tlen, name: str = None, cond = None):
    """
    Visualize conditioned distributions based on the values of a third variable.

    This function generates joint plots of two variables (X and Y) conditioned on the values of a third 
    variable (Z), where the conditioning can be based on a single value, a range, or the optimized KL 
    divergence value. The function uses seaborn to plot the distributions and allows for saving the result 
    to an image file.

    Args:
        timeseries (array-like): 
            A data structure (e.g., pandas DataFrame, numpy array) that contains the time series data 
            for multiple variables. 
        I (list or array-like): 
            A list of indices or criteria used to filter or select data from the `timeseries`.
        num (int): 
            The number of samples to consider for conditioning the distributions.
        tlen (int): 
            The length of the time series or the data points to consider.
        name (str, optional): 
            Name of the file to save the visualization image. If not provided, the visualization will 
            not be saved. Defaults to None.
        cond (float, int, list, optional): 
            The conditional value(s) for the variable `Z`. This could be:
            - A single float (a threshold value for conditioning),
            - An integer (same as above but with integer values),
            - A list of two values (defining a range for conditioning).
            If `None`, the optimal value of `Z` based on KL divergence is used. Defaults to None.

    Returns:
        None:
            The function will display the joint plots and optionally save them to a file.

    Notes:
        - The function creates joint plots of `X` and `Y` conditioned on the variable `Z`.
        - The data is first sorted based on `Z` and then split according to the given condition `cond`.
        - If no condition is specified (`cond=None`), the KL divergence optimization is performed to find 
          the optimal conditioning threshold for `Z`.
        - Conditional values can be a single value, an integer, or a list of two values.
        - Seaborn's `jointplot` is used for visualizing the distributions, and the function allows for saving 
          the resulting plots to a PNG file.
    """
    
    idx = timeseries[I[2]].argsort()
    Z_sort = timeseries[I[2]][idx]
    
    X_sort, Y_sort, sp = ifc.timeseries_quantile(timeseries, I, num, tlen)
    X = X_sort
    Y = Y_sort
    Z = sp
    
    if cond == None:
        cond_val = opimise_kl(X, Y, Z)
        X_sup = X[Z > cond_val]
        Y_sup = Y[Z > cond_val]
        X_inf = X[Z < cond_val]
        Y_inf = Y[Z < cond_val]
        xmin, xmax = 0, np.max(np.concatenate((X, Y)))
        ymin, ymax = 0, np.max(np.concatenate((X, Y)))
        g0 = sns.jointplot(x = X_sup, y = Y_sup, label = 'Z > '+f"$z_{cond_val}$", xlim = (xmin,xmax), ylim = (ymin,ymax))
        g0.ax_joint.set_xlabel('X', fontsize=15)
        g0.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='15')
        g1 = sns.jointplot(x = X_inf, y = Y_inf, label = 'Z < '+f'$z_{cond_val}$', xlim = (xmin,xmax), ylim = (ymin,ymax))
        g1.ax_joint.set_xlabel('X', fontsize=15)
        g1.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='15')

    if type(cond) is int:
        X_sup = X[Z > cond]
        Y_sup = Y[Z > cond]
        X_inf = X[Z <= cond]
        Y_inf = Y[Z <= cond]
        xmin, xmax = 0, np.max(np.concatenate((X, Y)))
        ymin, ymax = 0, np.max(np.concatenate((X, Y)))
        g0 = sns.jointplot(x = X_inf, y = Y_inf, label = 'Z < '+f'$z_{cond_val}$', xlim = (xmin,xmax), ylim = (ymin,ymax))
        g0.ax_joint.set_xlabel('X', fontsize=15)
        g0.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='15')
        g1 = sns.jointplot(x = X_sup, y = Y_sup, label = 'Z > '+f'$z_{cond_val}$', xlim = (xmin,xmax), ylim = (ymin,ymax))
        g1.ax_joint.set_xlabel('X', fontsize=15)
        g1.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='15')

    if type(cond) is list:
        X_a = X[Z < cond[0]]
        Y_a = Y[Z < cond[0]]
        X_b = X[np.logical_and(Z <= cond[1], Z >= cond[0])]
        Y_b = Y[np.logical_and(Z <= cond[1], Z >= cond[0])]
        X_c = X[Z > cond[1]]
        Y_c = Y[Z > cond[1]]
        xmin, xmax = np.min(np.concatenate((X, Y))), np.max(np.concatenate((X, Y)))
        ymin, ymax = np.min(np.concatenate((X, Y))), np.max(np.concatenate((X, Y)))
        g0 = sns.jointplot(x = X_a, y = Y_a, label = 'Z < '+str(round(max(Z_sort[Z < cond[0]]),2)), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g0.ax_joint.set_xlabel('X', fontsize=20)
        g0.ax_joint.set_ylabel('Y', fontsize=20)
        g0.ax_joint.tick_params(axis='both', which='major', labelsize=15)
        g0.ax_joint.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(fontsize='15')
        g1 = sns.jointplot(x = X_b, y = Y_b, label = str(round(max(Z_sort[Z < cond[0]]),2)) + '≤ Z ≤ '+str(round(min(Z_sort[Z > cond[1]]),2)), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g1.ax_joint.set_xlabel('X', fontsize=20)
        g1.ax_joint.set_ylabel('Y', fontsize=20)
        g1.ax_joint.tick_params(axis='both', which='major', labelsize=15)
        g1.ax_joint.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(fontsize='15')
        g2 = sns.jointplot(x = X_c, y = Y_c, label = str(round(min(Z_sort[Z > cond[1]]),2)) + '< Z', xlim = (xmin,xmax), ylim = (ymin,ymax))
        g2.ax_joint.set_xlabel('X', fontsize=20)
        g2.ax_joint.set_ylabel('Y', fontsize=20)
        g2.ax_joint.tick_params(axis='both', which='major', labelsize=15)
        g2.ax_joint.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(fontsize='15')
    
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Conditional distribution', fontsize=18)
    gs = gridspec.GridSpec(1, 3)

    mg0 = SeabornFig2Grid(g0, fig, gs[0])
    mg1 = SeabornFig2Grid(g1, fig, gs[1])
    if type(cond) is list:
        mg2 = SeabornFig2Grid(g2, fig, gs[2])

    gs.tight_layout(fig)
    
    if name != None:
        plt.savefig(name + '.png', format = 'png')


def visualisation_conditioned_val(timeseries: np.ndarray, I: list, num: int, tlen: int, name: str = None, cond: float = None):
    """
    Visualizes the conditional distribution of two variables (X, Y) based on a third variable (Z) 
    using joint plots for various conditions on Z.

    The function sorts the input time series based on quantile values and then generates joint plots 
    of X and Y conditioned on specific thresholds or ranges of Z. The conditional values of Z are 
    determined based on the provided 'cond' argument, which can be either:
        - `None`: Use the optimized KL divergence to split the data into two groups based on a threshold.
        - `int`: Split the data into two groups where Z is greater than or less than or equal to the provided value.
        - `list`: Split the data into three groups based on a range of values for Z, with conditions `Z < cond[0]`, `cond[0] <= Z <= cond[1]`, and `Z > cond[1]`.

    The function generates and displays joint plots for each of the specified conditions, and optionally 
    saves the resulting figure as a PNG file.

    Args:
        timeseries (np.ndarray): A 2D numpy array where each row represents a time step and each column represents a variable.
        I (list): A list of indices to be used for processing the time series.
        num (int): A numerical value to adjust quantile thresholds in the analysis.
        tlen (int): Length of the time series or number of time steps in the series.
        name (str, optional): The name for the saved image file. If None, no file is saved.
        cond (float or list, optional): The condition(s) on the third variable (Z) used to split the data.
            - If `None`, the function optimizes the KL divergence to determine a threshold for Z.
            - If `int`, the data is split into two groups based on whether Z is greater than or less than or equal to the provided value.
            - If `list`, the data is split into three groups based on the two values in the list.

    Returns:
        None: The function generates plots and optionally saves the figure as a PNG file.

    Notes:
        The function requires the seaborn library for generating the joint plots.
        The optimized KL divergence method (`opimise_kl`) is assumed to be defined elsewhere in the codebase.
        The SeabornFig2Grid function is also assumed to be defined elsewhere for custom grid layout handling.

    """

    X_sort, Y_sort, Z_sort, sp = ifc.timeseries_quantile_val(timeseries, I, num, tlen)
    X = X_sort
    Y = Y_sort
    Z = sp
    if cond == None:
        cond_val = opimise_kl(X, Y, Z)
        X_sup = X[Z > cond_val]
        Y_sup = Y[Z > cond_val]
        X_inf = X[Z < cond_val]
        Y_inf = Y[Z < cond_val]
        val = round(np.quantile(Z_sort,cond_val/num),1)
        xmin, xmax = 0, np.max(np.concatenate((X, Y)))
        ymin, ymax = 0, np.max(np.concatenate((X, Y)))
        g0 = sns.jointplot(x = X_sup, y = Y_sup, label = 'Z > '+str(val), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g0.ax_joint.set_xlabel('X', fontsize=15)
        g0.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='16')
        g1 = sns.jointplot(x = X_inf, y = Y_inf, label = 'Z < '+str(val), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g1.ax_joint.set_xlabel('X', fontsize=15)
        g1.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='16')

    if type(cond) is int:
        X_sup = X[Z > cond]
        Y_sup = Y[Z > cond]
        X_inf = X[Z <= cond]
        Y_inf = Y[Z <= cond]
        val = round(np.quantile(Z_sort,cond_val/num),1)
        xmin, xmax = 0, np.max(np.concatenate((X, Y)))
        ymin, ymax = 0, np.max(np.concatenate((X, Y)))
        g0 = sns.jointplot(x = X_inf, y = Y_inf, label = 'Z < '+str(val), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g0.ax_joint.set_xlabel('X', fontsize=15)
        g0.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='12')
        g1 = sns.jointplot(x = X_sup, y = Y_sup, label = 'Z > '+str(val), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g1.ax_joint.set_xlabel('X', fontsize=15)
        g1.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='12')

    if type(cond) is list:
        X_a = X[Z < cond[0]]
        Y_a = Y[Z < cond[0]]
        X_b = X[np.logical_and(Z <= cond[1], Z >= cond[0])]
        Y_b = Y[np.logical_and(Z <= cond[1], Z >= cond[0])]
        X_c = X[Z > cond[1]]
        Y_c = Y[Z > cond[1]]
        val1 = round(np.quantile(Z_sort,cond[0]/num),1)
        val2 = round(np.quantile(Z_sort,cond[1]/num),1)
        xmin, xmax = 0, np.max(np.concatenate((X, Y)))
        ymin, ymax = 0, np.max(np.concatenate((X, Y)))
        sns.set_context("talk")
        g0 = sns.jointplot(x = X_a, y = Y_a, label = 'Z < '+'$\mathregular{z_{i}}$'.replace('i',str(cond[0])), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g0.ax_joint.set_xlabel('X', fontsize=18)
        g0.ax_joint.set_ylabel('Y', fontsize=18)
        plt.legend(fontsize='16')
        g1 = sns.jointplot(x = X_b, y = Y_b, label ='$\mathregular{z_{i}}$'.replace('i',str(cond[0]))+'$\mathregular{\leq}$'+ 'Z' + '$\mathregular{\leq}$' + '$\mathregular{z_{i}}$'.replace('i',str(cond[1])), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g1.ax_joint.set_xlabel('X', fontsize=18)
        g1.ax_joint.set_ylabel('Y', fontsize=18)
        plt.legend(fontsize='16')
        g2 = sns.jointplot(x = X_c, y = Y_c, label = '$\mathregular{z_{i}}$'.replace('i',str(cond[1])) + '< Z', xlim = (xmin,xmax), ylim = (ymin,ymax))
        g2.ax_joint.set_xlabel('X', fontsize=18)
        g2.ax_joint.set_ylabel('Y', fontsize=18)
        plt.legend(fontsize='16')
        print(str(cond[0]), val1)
        print(str(cond[1]), val2)
    
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Conditional distribution', fontsize=18)
    gs = gridspec.GridSpec(1, 3)

    mg0 = SeabornFig2Grid(g0, fig, gs[0])
    mg1 = SeabornFig2Grid(g1, fig, gs[1])
    if type(cond) is list:
        mg2 = SeabornFig2Grid(g2, fig, gs[2])

    gs.tight_layout(fig)
    
    if name != None:
        plt.savefig(name + '.png', format = 'png')
        
    
def decision_tree(x_1d: np.ndarray, y_1d: np.ndarray, disp_fig: bool = False, disp_txt_rep: bool = False, disp_tree: bool = False, name: str = None):
    """
    Create and visualize a Decision Tree model to predict mutual information (MI) conditioned on quantile values.

    This function fits a Decision Tree Regressor to the provided input and output data, and visualizes the results 
    by displaying scatter plots, prediction plots, text representation of the decision tree, and graphical representation 
    of the decision tree based on user preferences. The function also returns threshold values for splits and output 
    values for the tree leaves.

    Args:
        x_1d (np.ndarray): A 1D numpy array representing the z-th quantile values (input features).
        y_1d (np.ndarray): A 1D numpy array representing the mutual information (MI) conditioned on z (target values).
        disp_fig (bool, optional): If True, display the scatter plot and prediction plot. Default is False.
        disp_txt_rep (bool, optional): If True, display the text representation of the decision tree. Default is False.
        disp_tree (bool, optional): If True, display the graphical representation of the decision tree. Default is False.
        name (str, optional): The name for saving output figures. If None, no figure will be saved. Default is None.

    Returns:
        tuple:
            split1 (int): The threshold value for the first split in the decision tree.
            split2 (int): The threshold value for the second split in the decision tree.
            output (list): A list containing the output values for the leaves of the decision tree.

    Notes:
        - The function requires the `sklearn.tree.DecisionTreeRegressor` and `matplotlib.pyplot` libraries.
        - The `output` list contains values associated with the leaves of the decision tree after training.
        - The `name` parameter, if provided, will save the figure as `name_tree1.png` for the scatter plot and prediction plot, and `name_tree2.png` for the decision tree plot.

    """

    if name != None:
        save_folder = "output"
        try : 
            os.mkdir(save_folder)
        except OSError : 
            pass

    size = int(np.max(x_1d))
    x = np.zeros((size,1))
    y = np.zeros((size,1))
    x[0:size,0] = x_1d
    y[0:size,0] = y_1d
    max_leaf_nodes = 3
    regr = tree.DecisionTreeRegressor(max_depth = 2, max_leaf_nodes = max_leaf_nodes)
    regr.fit(x, y)
    x_test = np.arange(0.0, size, 0.01)[:, np.newaxis]
    y_pred = regr.predict(x_test)
    val = sorted(regr.tree_.threshold[regr.tree_.threshold>0])
    values = list(regr.tree_.threshold)
    num_times, occurrence = max((len(list(item)), key) for key, item in groupby(values))
    index_leaves = [i for i, x in enumerate(values) if x == min(values)]
    if num_times == max_leaf_nodes:
        temp = list(regr.tree_.value[index_leaves,0,0])
        output = temp[1::]
        output.append(temp[0])
    else:
        output = list(regr.tree_.value[index_leaves,0,0])

    if disp_fig == True or name is not None:
        plt.figure()
        plt.scatter(x, y, s = 50, edgecolor = "black", c = "darkorange", label = "data")
        plt.plot(x_test, y_pred, color = "cornflowerblue", label = "max_depth=1", linewidth = 2)
        plt.title('Decision tree', fontsize = 15)
        plt.xlabel("$\mathregular{z^{th}}$ quantile", fontsize = 12)
        plt.ylabel("MI(XY|Z = z)", fontsize = 12)
        if name is not None:
            plt.savefig(name + '_tree1',bbox_inches="tight", dpi = 600)
    
    if disp_txt_rep == True:
        text_representation = tree.export_text(regr)
    
    if disp_tree == True or name is not None:
        fig = plt.figure(figsize = (8,4), dpi = 600)
        _ = tree.plot_tree(regr, filled = True, feature_names = ['z', 'z'], impurity = False, fontsize = 10)
        if name is not None:
            fig.savefig(name + '_tree2.png', dpi = 600)
    
    split1 = int(val[0])+1
    split2 = int(val[1])

    return split1, split2, output

    
def decision_tree_val(x_1d: np.ndarray, y_1d: np.ndarray, z: np.ndarray, disp_fig: bool = False, disp_txt_rep: bool = False, disp_tree: bool = False, name: str = None):
    """
    Create and visualize a Decision Tree model for predicting mutual information (MI) conditioned on a third variable (z).

    This function fits a Decision Tree Regressor to the provided input data (x_1d), output data (y_1d), 
    and conditional variable (z). The function visualizes the results through scatter plots, prediction plots, 
    and decision tree representations (both textual and graphical). It also returns threshold values for splits 
    and output values for the leaves of the decision tree.

    Args:
        x_1d (np.ndarray): A 1D numpy array representing the z-th quantile values (input features).
        y_1d (np.ndarray): A 1D numpy array representing the mutual information (MI) conditioned on z (target values).
        z (np.ndarray): A 1D numpy array representing the conditional variable.
        disp_fig (bool, optional): If True, display the scatter plot and prediction plot. Default is False.
        disp_txt_rep (bool, optional): If True, display the text representation of the decision tree. Default is False.
        disp_tree (bool, optional): If True, display the graphical representation of the decision tree. Default is False.
        name (str, optional): The name for saving output figures. If None, no figure will be saved. Default is None.

    Returns:
        tuple:
            split1 (int): The threshold value for the first split in the decision tree.
            split2 (int): The threshold value for the second split in the decision tree.
            output (list): A list containing the output values for the leaves of the decision tree.

    Notes:
        - The function requires the `sklearn.tree.DecisionTreeRegressor` and `matplotlib.pyplot` libraries.
        - The function computes quantiles of the conditional variable `z` and uses them for splitting the data.
        - The `output` list contains values associated with the leaves of the decision tree after training.
        - If `name` is provided, the function saves figures as `name_tree1.png` for the scatter plot and prediction plot, 
          and `name_tree2.png` for the decision tree plot.
        - The `z` values are sorted and used to determine thresholds for decision tree splits.

    """

    if name != None:
        save_folder = "output"
        try : 
            os.mkdir(save_folder)
        except OSError : 
            pass

    idx = z.argsort()
    z_sort = z[idx]
    z_val = np.round(np.array([np.quantile(z_sort,i) for i in np.arange(1,len(y_1d)+1)/len(y_1d)]),1)
    size = int(np.max(x_1d))
    x = np.zeros((size,1))
    y = np.zeros((size,1))
    x[0:size,0] = x_1d
    y[0:size,0] = y_1d
    max_leaf_nodes = 3
    regr = tree.DecisionTreeRegressor(max_depth = 2, max_leaf_nodes = max_leaf_nodes)
    regr.fit(x, y)
    x_test = np.arange(0.0, size, 0.01)[:, np.newaxis]
    y_pred = regr.predict(x_test)
    val = sorted(regr.tree_.threshold[regr.tree_.threshold>0])
    values = list(regr.tree_.threshold)
    num_times, occurrence = max((len(list(item)), key) for key, item in groupby(values))
    index_leaves = [i for i, x in enumerate(values) if x == min(values)]
    if num_times == max_leaf_nodes:
        temp = list(regr.tree_.value[index_leaves,0,0])
        output = temp[1::]
        output.append(temp[0])
    else:
        output = list(regr.tree_.value[index_leaves,0,0])

    if disp_fig == True or name is not None:
        plt.figure()
        plt.scatter(x, y, s = 50, edgecolor = "black", c = "darkorange", label = "data")
        plt.plot(x_test, y_pred, color = "cornflowerblue", label = "max_depth=1", linewidth = 2)
        plt.title('Decision tree', fontsize = 15)
        plt.xlabel("$\mathregular{z^{th}}$ quantile", fontsize = 12)
        plt.ylabel("MI(XY|Z = z)", fontsize = 12)
        plt.xticks(ticks=x_1d, labels=z_val)
        if name is not None:
            plt.savefig(name + '_tree1',bbox_inches="tight", dpi = 600)
    
    if disp_txt_rep == True:
        text_representation = tree.export_text(regr)
    
    if disp_tree == True or name is not None:
        fig = plt.figure(figsize = (8,4), dpi = 600)
        _ = tree.plot_tree(regr, filled = True, feature_names = ['z', 'z'], impurity = False, fontsize = 10)
        if name is not None:
            fig.savefig(name + '_tree2.png', dpi = 600)
        
    split1 = int(val[0])+1
    split2 = int(val[1])

    return split1, split2, output