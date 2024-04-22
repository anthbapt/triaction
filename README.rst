=============================================================================================
Triaction : Python code for triadic interactions
=============================================================================================

.. image:: logo.png
   :width: 200

This repository contains the Python package called triaction, which enables the detection of triadic interactions. It also provides visualization capabilities to visualise triadic interaction network.

The package has been created by Anthony Baptista and Ginestra Bianconi.

# Add link to arXiv link

-----------------
 Installation
-----------------

.. code-block:: bash    

  $ python setup.py install

-----------------
Data
-----------------

``Continuous data``:
 * Gene expression

-----------------
Usage
-----------------

Scatter plot of the results of the Triaction algorithm on AML gene expression data. Each data point shows the information-theoretic measures for a triple of nodes $X$, $Y$ and $Z$, namely MI and CMI, the mutual information and conditional mutual information between $X$ and $Y$, respectively. The colour of each point corresponds to the value of $\Sigma$, which characterises the strength of the triadic interaction between gene $Z$ and the edge between $X$ and $Y$.

.. image:: output/MI_merge.png
   :width: 600

.. code-block:: Python  

   name_X = 'GATA1'
   name_Y = 'TAL1'
   name_Z = 'KLF5'
   X = np.array(gene_expression.T[name_X])
   Y = np.array(gene_expression.T[name_Y])
   Z = np.array(gene_expression.T[name_Z])

   timeseries = np.zeros((3,len(X)))
   timeseries[0,:] = X
   timeseries[1,:] = Y
   timeseries[2,:] = Z
   I = [1,2,3]
   num = 5
   tlen = len(X)
   nrunmax = 1000
   MI, MIz, MIz_null, MIC, Theta_S, Theta2_T, Theta2_Tn, Sigma, Sigma_null_list, P, P_T, P_Tn = ifc.Theta_score_null_model(timeseries, I, num, tlen, nrunmax, True, True)
   MI, MIz, MIC, Corr_Sigma, Sigma, T, Tn, MINDY, MI1, MI2 = ifc.mutual_information_analysis_continuous_extended(timeseries, [0,1,2], num, tlen)
   x = range(1, num+1)
   th1,th2,c = decision_tree(x, MIz, disp_fig=True, disp_txt_rep=True,
                 disp_tree=True, name=save_folder + '/' + 'good_decision_tree')
   I = [0,1,2]
   visualisation_conditioned_val(timeseries, I, num, tlen, name = save_folder + '/' + 'good', cond = [th1,th2])

.. image:: output/good.png
   :width: 600

.. code-block:: Python  

   from triaction.triadic_vis import triadic_network_vis_from_data_and_graph

   sub_triadic = pd.concat([short_range[short_range['P']<0.001], long_range[long_range['P']<0.001]]).reset_index()
   sub_sub_triadic = sub_triadic.sort_values(by='Theta', ascending=False).reset_index()
   sub_sub_triadic = sub_sub_triadic[0:10]
   
   set_nodes = set(sub_sub_triadic['reg']).union(set(sub_sub_triadic['node1'])).union(set(sub_sub_triadic['node2']))
   graph_ppi.remove_edges_from(list(nx.selfloop_edges(graph_ppi)))
   sub_graph = nx.Graph(graph_ppi.subgraph(list(set_nodes)))
   
   sub_final = sub_triadic[sub_triadic['node1'].isin(set_nodes)]
   sub_final = sub_final[sub_final['node2'].isin(set_nodes)]
   sub_final = sub_final[sub_final['reg'].isin(set_nodes)]
   sub_final = sub_final.reset_index(drop=True)
   top = len(sub_final)
   triadic_network_vis_from_data_and_graph(sub_graph, sub_final, top)

.. image:: output/triadic_vis_from_data.png
   :width: 600
