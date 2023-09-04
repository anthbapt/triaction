import multiprocessing as mp
import pyitlibnew as drv
import pandas as pd
import numpy as np
import glob
import os


class FromDataToTriadic:
    """
    This class provides functionality for processing triadic data.

    Args:
        input_test (str): Path to the input triadic data file.
        input_graph (str): Path to the input graph data file.
        input_exp (str): Path to the input experimental data file.
        output (str): Path to the output data file.
        num_cpu (int): Number of CPU cores to use for processing.

    Attributes:
        input_graph (str): Path to the input graph data file.
        input_exp (str): Path to the input experimental data file.
        output (str): Path to the output data file.
        cpu (int): Number of CPU cores to use for processing.
        exp (pandas.DataFrame): DataFrame containing experimental data.
        triadic_list (pandas.DataFrame): DataFrame containing triadic data.
        out (pandas.DataFrame): Empty DataFrame for storing the output.

    Methods:
        sub_run(self, sub_triadic, i):
            Process a subset of triadic data and save the results to a file.

        run(self):
            Split the triadic data and run processing in parallel using multiple CPU cores.

        stitch(self):
            Combine processed data from multiple files into a single DataFrame.

        construct_triadic_dataframe(self, num=None):
            Process triadic data and return a DataFrame sorted by 'std'.

    """
    

    def __init__(self, input_test = '', input_graph = '', \
                 input_exp = '', output = '', num_cpu = 1):
        self.input_graph = input_graph
        self.input_exp = input_exp
        self.output = output
        self.cpu = num_cpu
        self.exp = pd.read_csv(input_exp, sep='\t', index_col=0)
        self.triadic_list = pd.read_csv(input_test, sep='\t', header=None)
        self.triadic_list['str'] = [0 for i in range(len(self.triadic_list))]
        self.triadic_list['corr'] = [0 for i in range(len(self.triadic_list))]
        self.out = pd.DataFrame()


    def sub_run(self, sub_triadic, i):
        """
        Process a subset of triadic data and save the results to a file.

        Args:
            sub_triadic (pandas.DataFrame): Subset of triadic data to process.
            i (int): Index of the subset.

        Returns:
            None
        """

        for k in range(len(sub_triadic)):
            Z = np.array((self.exp.loc[sub_triadic.iloc[k][0]]))
            X = np.array((self.exp.loc[sub_triadic.iloc[k][1]]))
            Y = np.array((self.exp.loc[sub_triadic.iloc[k][2]]))
            sub_triadic['str'].iloc[k] = drv.std(X, Y, Z)
            sub_triadic['corr'].iloc[k] = drv.corr(X, Y, Z)
        sub_triadic.to_csv('/data_' + str(i) + '.tsv', sep='\t', header=None, index=False)


    def run(self):
        """
        Split the triadic data and run processing in parallel using multiple CPU cores.

        Returns:
            None
        """

        df_split = np.array_split(self.triadic_list, self.cpu)
        p = mp.Pool(processes=self.cpu)
        p.starmap(self.sub_run, [(df_split[i], i) for i in range(self.cpu)])


    def stitch(self):
        """
        Combine processed data from multiple files into a single DataFrame.

        Returns:
            None
        """

        data_raw = sorted(glob.glob('data_*'))
        data_list = list()
        for k in data_raw:
            temp = pd.read_csv(k, sep='\t', header=None)
            data_list.append(temp)
        self.out = pd.concat([data_list[i] for i in range(len(data_list))], ignore_index=True)


    def construct_triadic_dataframe(self, num=None):
        """
        Process triadic data and return a DataFrame sorted by 'std'.

        Args:
            num (int, optional): Number of top records to return.

        Returns:
            pandas.DataFrame: Processed triadic data sorted by 'std'.
        """

        self.run()
        self.stitch()
        self.data_df = self.data_df.sort_values(by=['std'])
        if num is None:
            return self.data_df
        else:
            return self.data_df.iloc[0:num]
