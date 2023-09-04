from pyitlib.discrete_random_variable import information_mutual, information_mutual_conditional
from sklearn.metrics import mean_squared_error
from scipy.spatial import cKDTree as KDTree
from seaborn_grid import SeabornFig2Grid
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn import tree
import pyitlibnew as drv
import matplotlib as mpl
import seaborn as sns
import scipy as sp
import numpy as np


def func(x, a, b):
    """
    Compute a linear function.

    Args:
        x (array-like): Input data.
        a (float): Slope parameter.
        b (float): Intercept parameter.

    Returns:
        array-like: Output values computed using the linear function.
    """
    
    y = a * x + b
    return y


def fit(x, y):
    """
    Fit a curve to input data.

    Args:
        x (array-like): Input x-values.
        y (array-like): Input y-values.

    Returns:
        tuple: A tuple containing x_fit and y_fit, the fitted curve.
    """
    
    minima = min(np.min(x[0,:]),np.min(y[0,:]))
    maxima = max(np.max(x[1,:]),np.max(y[1,:]))
    t = np.linspace(minima, maxima, 100)
    popt_x, pcov_x = curve_fit(func, x[0,:], x[1,:])
    popt_y, pcov_y = curve_fit(func, y[0,:], y[1,:])
    
    x_fit = np.array([t, func(t,popt_x[0], popt_x[1])])
    y_fit = np.array([t, func(t,popt_y[0], popt_y[1])])
    
    return x_fit, y_fit


def opimise_kl(X, Y, Z):
    """
    Optimize KL divergence between two conditional distributions.

    Args:
        X (array-like): Input data for variable X.
        Y (array-like): Input data for variable Y.
        Z (array-like): Input data for variable Z.

    Returns:
        float: The optimized conditional value.
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


def visualisation_conditioned(X, Y, Z, name:str = None, cond = None):
    """
    Visualize conditioned distributions.

    Args:
        X (array-like): Input data for variable X.
        Y (array-like): Input data for variable Y.
        Z (array-like): Input data for variable Z.
        name (str, optional): Name for saving the visualization image.
        cond (float, optional): Conditional value for visualization.

    Returns:
        None
    """

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
    else:
        X_sup = X[Z > cond]
        Y_sup = Y[Z > cond]
        X_inf = X[Z < cond]
        Y_inf = Y[Z < cond]
        xmin, xmax = 0, np.max(np.concatenate((X, Y)))
        ymin, ymax = 0, np.max(np.concatenate((X, Y)))
        g0 = sns.jointplot(x = X_sup, y = Y_sup, label = 'Z > '+str(cond), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g0.ax_joint.set_xlabel('X', fontsize=15)
        g0.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='15')
        g1 = sns.jointplot(x = X_inf, y = Y_inf, label = 'Z < '+str(cond), xlim = (xmin,xmax), ylim = (ymin,ymax))
        g1.ax_joint.set_xlabel('X', fontsize=15)
        g1.ax_joint.set_ylabel('Y', fontsize=15)
        plt.legend(fontsize='15')
    
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Conditional distribution', fontsize=18)
    gs = gridspec.GridSpec(1, 2)

    mg0 = SeabornFig2Grid(g0, fig, gs[0])
    mg1 = SeabornFig2Grid(g1, fig, gs[1])

    gs.tight_layout(fig)
    
    if name != None:
        plt.savefig(name + '.png', format = 'png')
    
    
def decision_tree(x_1d, y_1d, disp_fig = False, disp_txt_rep = False, disp_tree = False):
    """
    Perform decision tree analysis.

    Args:
        x_1d (array-like): Input x-values.
        y_1d (array-like): Input y-values.
        disp_fig (bool, optional): Display figures. Default is False.
        disp_txt_rep (bool, optional): Display text representation. Default is False.
        disp_tree (bool, optional): Display decision tree. Default is False.

    Returns:
        float: Split value obtained from the decision tree analysis.
    """

    size = len(x_1d)
    x = np.zeros((size,1))
    y = np.zeros((size,1))
    x[0:size,0] = x_1d
    y[0:size,0] = y_1d
    regr = tree.DecisionTreeRegressor(max_depth=2, max_leaf_nodes=3)
    regr.fit(x, y)
    x_test = np.arange(0.0, size, 0.01)[:, np.newaxis]
    y_pred = regr.predict(x_test)
    val = regr.tree_.threshold[regr.tree_.threshold>0]
    min_arg = np.where(x == int(min(val)))[0]
    max_arg = np.where(x == int(max(val))+1)[0]
    split_val = x[np.where(y == max(y[min_arg], y[max_arg]))[0]-1][0]
    
    if disp_fig == True:
        plt.figure()
        plt.scatter(x, y, s=50, edgecolor="black", c="darkorange", label="data")
        plt.plot(x_test, y_pred, color="cornflowerblue", label="max_depth=1", linewidth=2)
        plt.title('Decision tree', fontsize=15)
        plt.xlabel("$\mathregular{z^{th}}$ quantile", fontsize=12)
        plt.ylabel("MI(XY|Z = z) - MI(XY|Z)", fontsize=12)
        plt.savefig('output/tree.png', format = 'png', dpi = 600)
    
    if disp_txt_rep == True:
        text_representation = tree.export_text(regr)
        print(text_representation)
    
    if disp_tree == True:
        fig = plt.figure(figsize=(8,4), dpi=600)
        _ = tree.plot_tree(regr, filled=True, feature_names=['z', 'z'], impurity=False, fontsize=10)
        fig.savefig('output/tree2.png', format = 'png')
        
    return split_val[0]


def triadic_analysis(X, Y, Z, bins, tol = 0, th = None, save_folder = None):
    """
    Perform triadic analysis.

    Args:
        X (array-like): Input data for variable X.
        Y (array-like): Input data for variable Y.
        Z (array-like): Input data for variable Z.
        bins (int): Number of bins for analysis.
        tol (int, optional): Tolerance for preprocessing. Default is 0.
        th (float, optional): Threshold value for decision tree analysis.
        save_folder (str, optional): Folder for saving visualizations.

    Returns:
        None
    """

    X, Y, Z, X_sort, Y_sort = drv.preprocess(X, Y, Z, num = bins, tol = tol, XY_sort = True)
    std, std_null, corr, zscore, T, T_null, T_zscore, Tn, Tn_null, Tn_zscore, MI_XY, MIC = stats(X, Y, Z, num = bins)
    T, Tn, MIz, Pz, valz = drv.pele_mele(X, Y, Z)
    print('Sigma: ', std)
    print('Sigma_null: ', std_null)
    print('corr: ', corr)
    print('z_score: ', zscore)
    print('MIC:', MIC)
    print('MI_XY:', MI_XY)
    sig_dif_abs_dis = abs(MIz-np.ones(len(MIz))*np.sum(MIz*Pz))
    if th == None:
        th = decision_tree(valz, sig_dif_abs_dis, disp_fig = True, disp_txt_rep = False, disp_tree = True)
        if save_folder == None:
            visualisation_conditioned(X_sort, Y_sort, Z, name = 'distribution_conditioned')
        else: 
            visualisation_conditioned(X_sort, Y_sort, Z, name = save_folder + '/distribution_conditioned')
    else:
        if save_folder == None:
            visualisation_conditioned(X_sort, Y_sort, Z, name = 'distribution_conditioned')
        else: 
            visualisation_conditioned(X_sort, Y_sort, Z, name = save_folder + '/distribution_conditioned')
            
            
def stats(X, Y, Z, num = 5, tol = 0):
    """
    Calculate some measures.

    Args:
        X (array-like): Input data for variable X.
        Y (array-like): Input data for variable Y.
        Z (array-like): Input data for variable Z.
        num (int, optional): Number of bins for analysis. Default is 5.
        tol (int, optional): Tolerance for preprocessing. Default is 0.

    Returns:
        tuple: A tuple containing some measures.
    """    

    X, Y, Z = drv.preprocess(X, Y, Z, num = num, tol = tol)
    corr = drv.corr(X, Y, Z)
    std = drv.std(X, Y, Z)
    T, Tn, MIz, Pz, valz = drv.pele_mele(X, Y, Z)
    std_null, std_std, T_null, T_std, Tn_null, Tn_std = drv.gaussian_null_model(X, Y, Z, num = num)
    zscore = np.abs((std - std_null)/std_std)
    T_zscore = np.abs((T - T_null)/T_std)
    Tn_zscore = np.abs((Tn - Tn_null)/Tn_std)
    MI_XY = information_mutual(X, Y)    
    MIC = information_mutual_conditional(X, Y, Z)
    
    return std, std_null, corr, zscore, T, T_null, T_zscore, Tn, Tn_null, Tn_zscore, MI_XY, MIC


def stats_dis(triadic, data, num = 5):
    """
    Calculate some measures for a dataset.

    Args:
        triadic (DataFrame): DataFrame containing triadic data.
        data (DataFrame): DataFrame containing input data.
        num (int, optional): Number of bins for analysis. Default is 5.

    Returns:
        tuple: A tuple containing some measures for a dataset.
    """

    std_array = np.array([])
    std_null_array = np.array([])
    corr_array = np.array([])
    zscore_array = np.array([])
    MI_XY_array = np.array([])
    MIC_array = np.array([])
    T_array = np.array([])
    T_null_array = np.array([])
    T_zscore_array = np.array([])
    Tn_array = np.array([])
    Tn_null_array = np.array([])
    Tn_zscore_array = np.array([])
    for k in range(len(triadic)):
        X = triadic.iloc[k]['node1']
        Y = triadic.iloc[k]['node2']
        Z = triadic.iloc[k]['reg']
        X = np.array(data[X])
        Y = np.array(data[Y])
        Z = np.array(data[Z])
        std, std_null, corr, zscore, T, T_null, T_zscore, Tn, Tn_null, Tn_zscore, MI_XY, MIC = stats(X, Y, Z)
        std_array = np.append(std_array, std)
        std_null_array = np.append(std_null_array, std_null)
        corr_array = np.append(corr_array, corr)
        zscore_array = np.append(zscore_array, zscore)
        T_array = np.append(T_array, T)
        T_null_array = np.append(T_null_array, T_null)
        T_zscore_array = np.append(T_zscore_array, T_zscore)
        Tn_array = np.append(Tn_array, Tn)
        Tn_null_array = np.append(Tn_null_array, Tn_null)
        Tn_zscore_array = np.append(Tn_zscore_array, Tn_zscore)
        MI_XY_array = np.append(MI_XY_array, MI_XY)
        MIC_array = np.append(MIC_array, MIC)
        if k%50 == 0:
            print(str(k/len(triadic)*100)+'%')
    
    return std_array, std_null_array, corr_array, zscore_array, T_array, T_null_array, T_zscore_array, Tn_array, Tn_null_array, Tn_zscore_array, MI_XY_array, MIC_array
