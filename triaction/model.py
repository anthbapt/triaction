import numpy as np
from scipy.integrate import odeint
import sdeint

# Define the Node Dynamics with Triadic Interactions class
class NDwTIs:
    """Node Dynamics with Triadic Interactions.

    Parameters
    ----------
    B : numpy.ndarray of shape (n_nodes, n_edges)
        the boundary operator of the structural network
    K : numpy.ndarray of shape (n_edges, n_nodes)
        the regulator network (structure of triadic interactions)
    w_pos : float
        the weight of positive regulator
    w_neg : float
        the weight of negative regulator
    threshold : float
        the threshold parameter
    alpha : float
        the coefficient of the triadic Laplacian
    noise_std : float
        the standard deviation of the Gaussian noise
    external_force : function, default = None
        the external force as a function of time
    x_init : numpy.ndarray, default = None
        the initial states of nodes
    dt : float, default = 0.01
        the time step size of the evolution
    t_max : float, default = 1.
        the time duration of the evolution

    Returns
    -------

    Attributes
    ----------
    n_nodes : int
        the number of nodes in the structural network
    
    n_edges : int
        the number of edges in the structural network
    
    n_hyperedges : int
        the number of triadic interactions
    
    n_pos_regulators : int
        the number of positive regulators
    
    n_neg_regulators : int
        the number of negative regulators
    
    n_timesteps : int
        the number of timesteps
        
    """
    
    def __init__(self, B:np.ndarray, K:np.ndarray, w_pos:float, w_neg:float, threshold:float, alpha:float, noise_std:float, external_force=None, x_init:np.ndarray=None, dt:float=0.01, t_max:float=1.):
        """Initialise the triadic interaction null model.
        
        Parameters
        ----------
        B : numpy.ndarray of shape (n_nodes, n_edges)
            The boundary operator of the structural network.
        K : numpy.ndarray of shape (n_edges, n_nodes)
            The regulator network.
        w_pos : float
            The weight of positive regulator.
        w_neg : float
            The weight of negative regulator.
        threshold : float
            The threshold parameter.
        alpha : float
            The coefficient of the triadic Laplacian.
        noise_std : float
            The standard deviation of the Gaussian noise.
        external_force : function, optional (default = None)
            The external force as a function of time.
        x_init : numpy.ndarray, optional (default = None)
            The initial states of nodes.
        dt : float, optional (default = 0.01)
            The time step size of the evolution.
        t_max : float, optional (default = 1)
            The time duration of the evolution.
            
        """
        # Structural network
        self.B = B
        self.n_nodes = B.shape[0]
        self.n_edges = B.shape[1]
        
        # Regulator network
        self.K = K
        self.w_pos = w_pos
        self.w_neg = w_neg
        self.n_hyperedges = int(np.sum(np.abs(self.K)))
        self.n_pos_regulators = int(np.sum(self.K == 1))
        self.n_reg_regulators = int(np.sum(self.K == -1))
        
        # Model parameters
        self.alpha = alpha
        self.threshold = threshold
        self.noise_std = noise_std
        
        # External force
        if external_force is not None:
            self.external_force = external_force
        else:
            self.external_force = None
        
        # Time evolution parameters
        self.dt = dt
        self.t_max = t_max
        self.n_timesteps = int(t_max / dt) + 1
        
        # Initial states
        if x_init is None:
            self.x_init = np.random.rand(self.n_nodes)
        else:
            self.x_init = x_init
    
    def getLaplacian(self, x:np.ndarray)->np.ndarray:
        """Compute the Laplacian of the states.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_nodes,)
            The states of nodes.

        Returns
        -------
        L : numpy.ndarray of shape (n_nodes, n_nodes)
            The Laplacian of the states.
        
        """
        # Triadic Laplacian
        Kx = np.dot(self.K, x)
        J = self.w_pos * (Kx > self.threshold) + self.w_neg * (Kx <= self.threshold)
        W = np.diag(J)
        L = np.dot(self.B, np.dot(W, self.B.T))
        return L
    
    def derivative(self, x:np.ndarray, t:float)->np.ndarray:
        """The time-derivatives of the states.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_nodes,)
            The states of nodes.
        t : float
            The time.

        Returns
        -------
        dxdt : numpy.ndarray of shape (n_nodes,)
            The time-derivatives of the states.
        
        """
        L = self.getLaplacian(x)
        if self.external_force is not None:
            return - np.dot(L, x) - self.alpha * x + self.external_force(t)
        return - np.dot(L, x) - self.alpha * x
    
    def noise(self, x:np.ndarray, t:float)->np.ndarray:
        """The coeficients of the noise term.

        Parameters
        ----------
        x : numpy.ndarray of shape (n_nodes,)
            The states of nodes.
        t : float
            The time.

        Returns
        -------
        noise : numpy.ndarray of shape (n_nodes, n_nodes)
            The coeficients of the noise term.
        
        """
        return self.noise_std * np.diag(np.ones(self.n_nodes)) # This results in an uncorrelated noise
    
    def integrate(self, deterministic:bool=False)->np.ndarray:
        """Evolve the system.

        Parameters
        ----------
        deterministic : bool, optional (default = False)
            If True, the integration is deterministic. (Default value = False)

        Returns
        -------
        timeseries : numpy.ndarray of shape (n_nodes, n_timesteps)
            The time series of the states.
        
        """
        t = np.linspace(0, self.t_max, num=self.n_timesteps)
        if deterministic:
            timeseries = odeint(self.derivative, self.x_init, t)
        else:
            timeseries = sdeint.itoint(self.derivative, self.noise, self.x_init, t)
        return timeseries.T

    def run(self, deterministic:bool=False)->np.ndarray:
        """Run the system.

        Parameters
        ----------
        deterministic : bool, optional (default = False)
            If True, the model runs deterministically. (Default value = False)

        Returns
        -------
        timeseries : numpy.ndarray of shape (n_nodes, n_timesteps)
            The time series of the states.
        
        """
        return self.integrate(deterministic=deterministic)
