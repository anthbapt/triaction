import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SeabornFig2Grid:
    """A class for arranging Seaborn plots within a grid.

    This class is designed to arrange Seaborn plots created using FacetGrid,
    PairGrid, or JointGrid within a specified subplot specification of a Matplotlib
    figure.

    Parameters
    ----------
    seaborngrid : sns.axisgrid.FacetGrid or sns.axisgrid.PairGrid or sns.axisgrid.JointGrid
        The Seaborn grid to be arranged within the specified subplot.
    fig : matplotlib.figure.Figure
        The Matplotlib figure to contain the Seaborn grid.
    subplot_spec : matplotlib.gridspec.SubplotSpec
        The subplot specification defining the position and size of the grid.

    
    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure containing the Seaborn grid.
    sg : sns.axisgrid.FacetGrid or sns.axisgrid.PairGrid or sns.axisgrid.JointGrid
        The Seaborn grid being arranged.
    subplot : matplotlib.gridspec.SubplotSpec
        The subplot specification defining the position and size of the grid.
    subgrid : matplotlib.gridspec.GridSpec
        The grid specification created for the Seaborn grid.

    Methods
    ----------
    - movegrid():
        Move and resize a PairGrid or FacetGrid within the specified subplot.
    - movejointgrid():
        Move and resize a JointGrid within the specified subplot.
    - moveaxes(ax, gs):
        Move and resize a Matplotlib axes within the specified grid specification.
    - finalize():
        Finalize the arrangement by closing the original Seaborn figure and
        connecting the Matplotlib figure to resize events.
    - resize(evt = None):
        Resize the Seaborn figure to match the size of the Matplotlib figure.

    Example
    ----------
        To use this class to arrange a Seaborn grid within a Matplotlib figure,
        create an instance of the class and call its constructor with the
        appropriate Seaborn grid, figure, and subplot specification:

        ```python
        grid = SeabornFig2Grid(seaborn_grid, matplotlib_figure, subplot_spec)
        ```
    """


    def __init__(self, seaborngrid, fig, subplot_spec):
        """Initialize the SeabornFig2Grid instance.

        Parameters
        ----------
        seaborngrid : sns.axisgrid.FacetGrid or sns.axisgrid.PairGrid or sns.axisgrid.JointGrid
            The Seaborn grid to be arranged within the specified subplot.
        fig : matplotlib.figure.Figure
            The Matplotlib figure to contain the Seaborn grid.
        subplot_spec : matplotlib.gridspec.SubplotSpec
            The subplot specification defining the position and size of the grid.
        """
        
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()


    def _movegrid(self):
        """Move PairGrid or Facetgrid to the specified subplot."""

        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec = self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])


    def _movejointgrid(self):
        """Move Jointgrid to the specified subplot."""

        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r + 1, r + 1, subplot_spec = self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])


    def _moveaxes(self, ax, gs):
        """Move and resize a Matplotlib axes within the specified grid specification.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Matplotlib axes to be moved and resized.
        gs : matplotlib.gridspec.GridSpec
            The grid specification defining the position and size.
        """

        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)


    def _finalize(self):
        """Finalize the arrangement by closing the original Seaborn figure and connecting to resize events."""

        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)


    def _resize(self, evt = None):
        """Resize the Seaborn figure to match the size of the Matplotlib figure."""

        self.sg.fig.set_size_inches(self.fig.get_size_inches())
