from sklearn.metrics import mean_squared_error
from triaction.seaborn_grid import SeabornFig2Grid
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


def func(x:np.ndarray, a:float, b:float):
    """
    Compute a linear function.

    Parameters
    ----------
    x : numpy.ndarray
        Input data.
    a : float
        Slope parameter.
    b : float
        Intercept parameter.

    Returns
    ----------
    y : numpy.ndarray
        Output values computed using the linear function.
    """
    
    y = a * x + b
    return y


def fit(x:np.ndarray, y:np.ndarray):
    """
    Fit a curve to input data.

    Parameters
    ----------
    x : numpy.ndarray
        Input x-values.
    y : numpy.ndarray
        Input y-values.

    Returns
    ----------
    x_fit : numpy.ndarray
        Fitted curve along X-axis
    y_fit : numpy.ndarray
        Fitted curve along Y-axis
    """
    
    minima = min(np.min(x[0,:]),np.min(y[0,:]))
    maxima = max(np.max(x[1,:]),np.max(y[1,:]))
    t = np.linspace(minima, maxima, 100)
    popt_x, pcov_x = curve_fit(func, x[0,:], x[1,:])
    popt_y, pcov_y = curve_fit(func, y[0,:], y[1,:])
    
    x_fit = np.array([t, func(t,popt_x[0], popt_x[1])])
    y_fit = np.array([t, func(t,popt_y[0], popt_y[1])])
    
    return x_fit, y_fit


def opimise_kl(X:np.ndarray, Y:np.ndarray, Z:np.ndarray):
    """
    Optimize KL divergence between two conditional distributions.

    Parameters
    ----------
    X : numpy.ndarray
        Input data for variable X.
    Y : numpy.ndarray
        Input data for variable Y.
    Z : numpy.ndarray
        Input data for variable Z.

    Returns
    ----------
    cond_val : float
        The optimized conditional value.
    """

    val_Z = sorted(list(set(Z)))
    kl_list = list()
    for k in range(1, len(val_Z)-1):
        X_sup = X[Z > val_Z[k]]
        Y_sup = Y[Z > val_Z[k]]
        X_inf = X[Z <= val_Z[k]]
        Y_inf = Y[Z <= val_Z[k]]
        inf_cond = np.array([X_inf, Y_inf])
        sup_cond = np.array([X_sup, Y_sup])
        fit_inf_cond, fit_sup_cond = fit(inf_cond, sup_cond)
        
        # test positivity of the two fit for the kl div np.all(a) > 0
        if (np.all(fit_inf_cond[1] >= 0) == False) and (np.all(fit_sup_cond[1] >= 0) == True):
            temp = np.where(fit_inf_cond[1]>=0)
            fit_sup = fit_sup_cond[1][temp]
            fit_inf = fit_inf_cond[1][temp]
        elif (np.all(fit_sup_cond[1] >= 0) == False) and (np.all(fit_inf_cond[1] >= 0) == True):
            temp = np.where(fit_sup_cond[1]>=0)
            fit_inf = fit_inf_cond[1][temp]
            fit_sup = fit_sup_cond[1][temp]
        elif (np.all(fit_inf_cond[1] >= 0) == False) and (np.all(fit_sup_cond[1] >= 0) == False):
            temp = max(np.min(np.where(fit_sup_cond[1]>=0)), np.min(np.where(fit_inf_cond[1]>=0)))
            fit_inf = fit_inf_cond[1][temp::]
            fit_sup = fit_sup_cond[1][temp::]
        elif (np.all(fit_inf_cond[1] >= 0) == True) and (np.all(fit_sup_cond[1] >= 0) == True):
            fit_inf = fit_inf_cond[1]
            fit_sup = fit_sup_cond[1]
        
        fit_inf = fit_inf/np.sum(fit_inf)
        fit_sup = fit_sup/np.sum(fit_sup)

        kl = sp.special.kl_div(fit_inf, fit_sup)
        kl_list.append(np.sum(kl))        
    cond_val = val_Z[np.argmax(kl_list)] + 1 # we start at the second value of val_Z
    
    return cond_val


def visualisation_conditioned(timeseries:np.ndarray, I:list, num:int, tlen:int, name:str = None, cond:float = None):
    """
    Visualize conditioned distributions.

    Parameters
    ----------
    timeseries : numpy.ndarray
        2D array representing the time series data.
    I : list
        List of indices specifying rows of interest in the time series.
    num : int
        Number of quantiles to compute.
    tlen : int
        Length of the time window considered for quantile computation.
    name : str
        Name for saving the visualization image.
    cond : float
        Conditional value for visualization.

    Returns
    ----------
    None
    """

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
        g0 = sns.jointplot(x = X_sup, y = Y_sup, label = 'Z > '+str(cond_val), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g0.ax_joint.set_xlabel('X', fontsize=15)
        g0.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='15')
        g1 = sns.jointplot(x = X_inf, y = Y_inf, label = 'Z < '+str(cond_val), xlim = (xmin,xmax), ylim = (ymin,ymax))
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
        g0 = sns.jointplot(x = X_inf, y = Y_inf, label = 'Z < '+str(cond), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g0.ax_joint.set_xlabel('X', fontsize=15)
        g0.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='15')
        g1 = sns.jointplot(x = X_sup, y = Y_sup, label = 'Z > '+str(cond), xlim = (xmin,xmax), ylim = (ymin,ymax))
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
        xmin, xmax = 0, np.max(np.concatenate((X, Y)))
        ymin, ymax = 0, np.max(np.concatenate((X, Y)))
        g0 = sns.jointplot(x = X_a, y = Y_a, label = 'Z < '+str(cond[0]), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g0.ax_joint.set_xlabel('X', fontsize=15)
        g0.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='15')
        g1 = sns.jointplot(x = X_b, y = Y_b, label = str(cond[0]) + '<= Z <= '+str(cond[1]), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g1.ax_joint.set_xlabel('X', fontsize=15)
        g1.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='15')
        g2 = sns.jointplot(x = X_c, y = Y_c, label = str(cond[1]) + '< Z', xlim = (xmin,xmax), ylim = (ymin,ymax))
        g2.ax_joint.set_xlabel('X', fontsize=15)
        g2.ax_joint.set_ylabel('Y', fontsize=15)
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


def visualisation_conditioned_val(timeseries:np.ndarray, I:list, num:int, tlen:int, name:str = None, cond:float = None):
    """
    Visualize conditioned distributions.

    Parameters
    ----------
    timeseries : numpy.ndarray
        2D array representing the time series data.
    I : list
        List of indices specifying rows of interest in the time series.
    num : int
        Number of quantiles to compute.
    tlen : int
        Length of the time window considered for quantile computation.
    name : str
        Name for saving the visualization image.
    cond : float
        Conditional value for visualization.

    Returns
    ----------
    None
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
        #g0 = sns.jointplot(x = X_a, y = Y_a, label = 'Z < '+str(val1), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g0 = sns.jointplot(x = X_a, y = Y_a, label = 'Z < '+'$\mathregular{z_{i}}$'.replace('i',str(cond[0])), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g0.ax_joint.set_xlabel('X', fontsize=18)
        g0.ax_joint.set_ylabel('Y', fontsize=18)
        plt.legend(fontsize='16')
        #g1 = sns.jointplot(x = X_b, y = Y_b, label = str(val1) + '<= Z <= '+str(val2), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g1 = sns.jointplot(x = X_b, y = Y_b, label ='$\mathregular{z_{i}}$'.replace('i',str(cond[0]))+'$\mathregular{\leq}$'+ 'Z' + '$\mathregular{\leq}$' + '$\mathregular{z_{i}}$'.replace('i',str(cond[1])), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g1.ax_joint.set_xlabel('X', fontsize=18)
        g1.ax_joint.set_ylabel('Y', fontsize=18)
        plt.legend(fontsize='16')
        #g2 = sns.jointplot(x = X_c, y = Y_c, label = str(val2) + '< Z', xlim = (xmin,xmax), ylim = (ymin,ymax))
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
        
    
def decision_tree(x_1d:np.ndarray, y_1d:np.ndarray, disp_fig:bool = False, disp_txt_rep:bool = False, disp_tree:bool = False, name:str = None):
    """
    Create and visualize a Decision Tree model for predicting mutual information.

    Parameters
    ----------
    x_1d : numpy.ndarray
        Input data representing the z-th quantile.
    y_1d : numpy.ndarray 
        Output data representing mutual information (MI) conditioned on z.
    disp_fig : bool
        Display scatter and prediction plot. Default is False.
    disp_txt_rep : bool
        Display text representation of the decision tree. Default is False.
    disp_tree : bool
        Display graphical representation of the decision tree. Default is False.
    name : str
        Name for saving output figures. Default is None.

    Returns
    ----------
    split1 : int
        Threshold value for the first split.
    split2 : int
        Threshold value for the second split.
    output : list
        Output values for the decision tree leaves.
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

    
def decision_tree_val(x_1d:np.ndarray, y_1d:np.ndarray, z:np.ndarray, disp_fig:bool = False, disp_txt_rep:bool = False, disp_tree:bool = False, name:str = None):
    """
    Create and visualize a Decision Tree model for predicting mutual information.

    Parameters
    ----------
    Parameters
    ----------
    x_1d : numpy.ndarray
        Input data representing the z-th quantile.
    y_1d : numpy.ndarray 
        Output data representing mutual information (MI) conditioned on z.
    z : numpy.ndarray
        Conditional variable
    disp_fig : bool
        Display scatter and prediction plot. Default is False.
    disp_txt_rep : bool
        Display text representation of the decision tree. Default is False.
    disp_tree : bool
        Display graphical representation of the decision tree. Default is False.
    name : str
        Name for saving output figures. Default is None.


    Returns
    ----------
    split1 : int
        Threshold value for the first split.
    split2 : int
        Threshold value for the second split.
    output : list
        Output values for the decision tree leaves.
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
