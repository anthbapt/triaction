import matplotlib.pyplot as plt
from matplotlib import pylab
import networkx as nx
import pandas as pd
import numpy as np
import os


def triadic_network_vis_from_scratch(structural_nodes:list, structural_edges:list, 
                                    pos_regulatory_nodes:list, pos_regulatory_edges:list, 
                                    neg_regulatory_nodes:list, neg_regulatory_edges:list, 
                                    connected_edges:list):
    """
    Visualize a triadic network from scratch.

    Parameters
    ----------
    structural_nodes : list
        List of nodes for the structural graph.
    structural_edges : list
        List of edges for the structural graph.
    pos_regulatory_nodes : list
        List of nodes with positive regulatory edges.
    pos_regulatory_edges : list
        List of positive regulatory edges.
    neg_regulatory_nodes : list
        List of nodes with negative regulatory edges.
    neg_regulatory_edges : list
        List of negative regulatory edges.
    connected_edges : list
        List of edges connecting nodes.

    Keyword Args:
    ----------
    save_folder : str
        Folder path to save the visualization. Default is None.

    Returns
    ----------
    None
    """

    save_folder = "output"
    try : 
        os.mkdir(save_folder)
    except OSError : 
        pass

    regulatory_nodes = pos_regulatory_nodes + neg_regulatory_nodes
    regulatory_edges = pos_regulatory_edges + neg_regulatory_edges
    structural_graph = nx.Graph()
    structural_graph.add_nodes_from(structural_nodes)
    structural_graph.add_edges_from(structural_edges)
    pos = nx.spring_layout(structural_graph)

    for k in range(int(len(connected_edges)/2)):
        reg = connected_edges[2*k][1]
        pos_node1 = pos[connected_edges[2*k][0]]
        pos_node2 = pos[connected_edges[(2*k)+1][0]]
        pos_reg = np.array([(pos_node1[0]+pos_node2[0])/2, (pos_node1[1]+pos_node2[1])/2])
        pos[reg] = pos_reg
    structural_graph.add_nodes_from(regulatory_nodes)
    structural_graph.add_edges_from(regulatory_edges)

    # nodes
    options = {"node_size": 150}
    nx.draw_networkx_nodes(structural_graph, pos, nodelist = neg_regulatory_nodes, node_shape = 'h', node_color = "red", **options)
    nx.draw_networkx_nodes(structural_graph, pos, nodelist = pos_regulatory_nodes, node_shape = 'h', node_color = "green", **options)
    nx.draw_networkx_nodes(structural_graph, pos, nodelist = structural_nodes, node_color = "blue", alpha = 0.5, **options)

    # edges
    nx.draw_networkx_edges(structural_graph,
        pos,
        edgelist = neg_regulatory_edges,
        width = 0.5,
        alpha = 0.5,
        style = 'dashed',
        edge_color = "red")
    nx.draw_networkx_edges(structural_graph,
        pos,
        edgelist = pos_regulatory_edges,
        width = 0.5,
        alpha = 0.5,
        style = 'dashed',
        edge_color = "green")
    nx.draw_networkx_edges(
        structural_graph,
        pos,
        edgelist = structural_edges,
        width = 0.5,
        alpha = 0.5,
        edge_color = "blue")
    
    # labels
    node_labels = {i:str(i) for i in structural_nodes}
    pos_labels = {i:pos[i] for i in structural_nodes}
    nx.draw_networkx_labels(structural_graph, pos_labels, node_labels, font_size = 8, font_color = "black", verticalalignment = 'center')
    plt.savefig(save_folder + '/test_vis_from_scratch.png', format = 'png', dpi = 600)
    
    
def triadic_network_vis_from_data(triadic:pd.DataFrame, top:int):
    """
    Visualize a triadic network from given data.

    Parameters
    ----------
    triadic : pandas.DataFrame
        DataFrame containing triadic data.
    top : int
        Number of top records to visualize.

    Keyword Args
    ----------
    save_folder : str
        Folder path to save the visualization. Default is None.

    Returns
    ----------
    None
    """

    save_folder = "output"
    try : 
        os.mkdir(save_folder)
    except OSError : 
        pass
    
    plt.figure(num = None, figsize = (top, top), dpi = 600)
    plt.axis('off')
    fig = plt.figure(1)
    cut = 1.00
    
    triadic = triadic.iloc[0:top]
    triadic_nodes = (set(list(triadic[0])).union(set(list(triadic[1])))).union(set(list(triadic[2])))
    pos_regulatory_nodes = list()
    pos_regulatory_edges = list()
    neg_regulatory_nodes = list()
    neg_regulatory_edges = list()
    connected_edges = list()
    for k in range(len(triadic)):
        reg = triadic.iloc[k][0]
        node1 = triadic.iloc[k][1]
        node2 = triadic.iloc[k][2]
        sign = triadic.iloc[k][5]
        if sign == 1:
            regulatory_node = str(node1)+'_'+str(node2)
            pos_regulatory_nodes.append(regulatory_node)
            pos_regulatory_edges.append((reg,regulatory_node))
        else:
            regulatory_node = str(node1)+'_'+str(node2)
            neg_regulatory_nodes.append(regulatory_node)
            neg_regulatory_edges.append((reg,regulatory_node))
        connected_edges.append((node1,regulatory_node))
        connected_edges.append((node2,regulatory_node))

    regulatory_nodes = pos_regulatory_nodes + neg_regulatory_nodes
    regulatory_edges = pos_regulatory_edges + neg_regulatory_edges
    graph = nx.Graph(regulatory_nodes)
    pos = nx.spring_layout(graph)
    for k in range(int(len(connected_edges)/2)):
        reg = connected_edges[2*k][1]
        pos_node1 = pos[connected_edges[2*k][0]]
        pos_node2 = pos[connected_edges[(2*k)+1][0]]
        max_x = max(pos_node1[0],pos_node2[0])
        min_x = min(pos_node1[0],pos_node2[0])
        max_y = max(pos_node1[1],pos_node2[1])
        min_y = min(pos_node1[1],pos_node2[1])
        pos_reg = np.array([(min_x + (max_x-min_x)/2), (min_y + (max_y-min_y)/2)])
        pos[reg] = pos_reg
    graph.add_edges_from(regulatory_edges)

    # nodes
    nx.draw_networkx_nodes(graph, pos, nodelist = neg_regulatory_nodes, \
                           node_shape = 'h', node_color="red", node_size = 2)
    nx.draw_networkx_nodes(graph, pos, nodelist = pos_regulatory_nodes, \
                           node_shape = 'h', node_color="green", node_size = 2)

    # edges
    nx.draw_networkx_edges(graph,
        pos,
        edgelist = neg_regulatory_edges,
        width = 0.2,
        alpha = 0.5,
        style = 'dashed',
        edge_color = "red")
    nx.draw_networkx_edges(graph,
        pos,
        edgelist = pos_regulatory_edges,
        width = 0.2,
        alpha = 0.5,
        style = 'dashed',
        edge_color = "green")
    
    # labels
    node_labels = {i:str(i) for i in regulatory_nodes}
    pos_labels = {i:pos[i] for i in regulatory_nodes}
    nx.draw_networkx_labels(graph, pos_labels, node_labels, font_size = 2, font_color = "black")
    plt.savefig(save_folder + '/triadic_vis_from_data.png', format = 'png', dpi = 600)
    
    
def triadic_network_vis_from_data_and_graph(graph:nx.Graph, triadic:pd.DataFrame, top:int):
    """
    Visualize a triadic network from given data and an existing graph.

    Parameters
    ----------
    graph : networkx.Graph
        Existing graph.
    triadic : pandas.DataFrame
        DataFrame containing triadic data.
    top : int
        Number of top records to visualize.

    Keyword Args
    ----------
    save_folder : str
        Folder path to save the visualization. Default is None.

    Returns
    ----------
    None
    """
    
    save_folder = "output"
    try : 
        os.mkdir(save_folder)
    except OSError : 
        pass

    plt.figure(num = None, figsize = (10, 10), dpi = 600)
    plt.axis('off')
    fig = plt.figure(1)
    cut = 1.00
    
    triadic = triadic.iloc[0:top]
    triadic_nodes = (set(list(triadic['reg'])).union(set(list(triadic['node1'])))).union(set(list(triadic['node2'])))
    graph.remove_edges_from(nx.selfloop_edges(graph))
    sub_graph = graph.subgraph(triadic_nodes)
    pos_regulatory_nodes = list()
    pos_regulatory_edges = list()
    neg_regulatory_nodes = list()
    neg_regulatory_edges = list()
    connected_edges = list()
    for k in range(len(triadic)):
        reg = triadic.iloc[k]['reg']
        node1 = triadic.iloc[k]['node1']
        node2 = triadic.iloc[k]['node2']
        sign = np.sign(triadic.iloc[k]['corr'])
        if sign == 1:
            regulatory_node = str(node1)+'_'+str(node2)
            pos_regulatory_nodes.append(regulatory_node)
            pos_regulatory_edges.append((reg,regulatory_node))
        else:
            regulatory_node = str(node1)+'_'+str(node2)
            neg_regulatory_nodes.append(regulatory_node)
            neg_regulatory_edges.append((reg,regulatory_node))
        connected_edges.append((node1,regulatory_node))
        connected_edges.append((node2,regulatory_node))

    structural_nodes = sub_graph.nodes()
    structural_edges = sub_graph.edges()
    regulatory_nodes = pos_regulatory_nodes + neg_regulatory_nodes
    regulatory_edges = pos_regulatory_edges + neg_regulatory_edges

    structural_graph = nx.Graph()
    structural_graph.add_nodes_from(structural_nodes)
    structural_graph.add_edges_from(structural_edges)
    pos = nx.spring_layout(structural_graph)
    for k in range(int(len(connected_edges)/2)):
        reg = connected_edges[2*k][1]
        pos_node1 = pos[connected_edges[2*k][0]]
        pos_node2 = pos[connected_edges[(2*k)+1][0]]
        max_x = max(pos_node1[0],pos_node2[0])
        min_x = min(pos_node1[0],pos_node2[0])
        max_y = max(pos_node1[1],pos_node2[1])
        min_y = min(pos_node1[1],pos_node2[1])
        pos_reg = np.array([(min_x + (max_x-min_x)/2), (min_y + (max_y-min_y)/2)])
        pos[reg] = pos_reg
    structural_graph.add_nodes_from(regulatory_nodes)
    structural_graph.add_edges_from(regulatory_edges)

    # nodes
    nx.draw_networkx_nodes(structural_graph, pos, nodelist = neg_regulatory_nodes, \
                           node_shape = 'D', node_color = "black", node_size = 40)
    nx.draw_networkx_nodes(structural_graph, pos, nodelist = pos_regulatory_nodes, \
                           node_shape = 'D', node_color = "black", node_size = 40)
    nx.draw_networkx_nodes(structural_graph, pos, nodelist = structural_nodes, \
                           node_color = "blue", alpha = 0.5, node_size = 120)

    # edges
    nx.draw_networkx_edges(structural_graph,
        pos,
        edgelist = neg_regulatory_edges,
        width = 0.5,
        alpha = 0.5,
        style = 'dashed',
        edge_color = "red")
    nx.draw_networkx_edges(structural_graph,
        pos,
        edgelist = pos_regulatory_edges,
        width = 0.5,
        alpha = 0.5,
        style = 'dashed',
        edge_color = "green")
    nx.draw_networkx_edges(
        structural_graph,
        pos,
        edgelist = structural_edges,
        width = 0.5,
        alpha = 0.5,
        edge_color = "blue")
    
    # labels
    node_labels = {i:str(i) for i in structural_nodes}
    pos_labels = {i:pos[i] for i in structural_nodes}
    nx.draw_networkx_labels(structural_graph, pos_labels, node_labels, font_size = 8, font_color = "black", verticalalignment = 'bottom')
    plt.savefig(save_folder + '/triadic_vis_from_data.png', format = 'png', dpi = 600)