import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# from scipy import interpolate
# from scipy.signal import savgol_filter
# from sklearn.preprocessing import MinMaxScaler







def vis_graph(node_pos, node_w, edge_idx, edge_w, f_name, gt_id=None, img_path=None):
    """
    input:  
        node_pos:[Num_node, 2]
        node_w: [Num_node]
        edge_idx: [2, Num_edge]
        edge_w: [Num_edge]

    """
    # node_pos, node_w, edge_idx, edge_w = graph.pos, graph.y, graph.edge_index, graph.edge_attr


    # Create a graph object
    G = nx.Graph()
    N = node_w.shape[0]
    # Add nodes to the graph with their positions and weights

    for i in range(N):
        G.add_node(i, pos=(node_pos[i,0].item(), node_pos[i,1].item()), weight=node_w[i].item())

    # Add edges to the graph with their weights
    N_edge = edge_idx.shape[1]
    for i in range(N_edge):
        G.add_edge(edge_idx[0, i].item(), edge_idx[1, i].item(), weight=edge_w[i].item())

    # Draw the graph with node colors and edge widths
    pos = nx.get_node_attributes(G, 'pos')
    weights = nx.get_node_attributes(G, 'weight')
    # edge_widths = nx.get_edge_attributes(G, 'weight').values()
    edge_weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    # scaler = MinMaxScaler(feature_range=(1.5, 2.5))
    # scaler.fit_transform([[w] for w in edge_weights])
    # edge_weights = scaler.transform([[w] for w in edge_weights]).ravel()

    
    plt.figure(figsize=(8, 8))
    if img_path is not None:
        img = plt.imread(img_path)
        plt.imshow(img)
    
    nx.draw_networkx_nodes(G, pos, node_color=list(weights.values()), cmap=plt.cm.Reds)
    nx.draw_networkx_nodes(G, pos, nodelist=[1], linewidths=4, edgecolors='orange') #coarse search result
    
    if gt_id is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[gt_id], linewidths=4, edgecolors='green') #refine result

    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_weights, edge_cmap=plt.cm.Blues)
    # nx.draw_networkx_labels(G, pos)
    # nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes()}, font_size=12, font_color='black', label_pos=0.5, ax=plt.gca())
    
    
    plt.axis('off')
    plt.savefig(f_name)
    plt.close()

def vis_single_grapg(x, pos, edge_idx, edge_attr, x_batch, f_name, img_path):
    nod_n = len(x_batch[x_batch==0])
    this_node, this_pos = x[:nod_n], pos[:nod_n]
    this_y = np.arange(nod_n)
    select_edge = edge_idx[0]< nod_n
    this_edge = edge_idx[:, select_edge]
    this_edge_attr = edge_attr[select_edge].sum(-1)


    vis_graph(this_pos, this_y, this_edge, this_edge_attr, f_name, gt_id=None, img_path=img_path)


def statstic_edge_weigt(graphs_list, gt_ids):
    """
        statistic the edge weight of every nodes, show the assumption that inliers' edge weight are higher than the outlier's.
        graph:[list], element: torch_geometric graph Data 
        gt_ids:[list], elemeny: int, index of the inlier, 'gt_ids' has the same length with the graph_list.
        
    """
    all_inlier = []
    all_outlier = []
    for graph, gt_id in zip(graphs_list, gt_ids):
        node_pos, node_w, edge_idx, edge_w = graph.pos, graph.y, graph.edge_index, graph.edge_attr
        G = nx.Graph()
        N = node_w.shape[0]
        # Add nodes to the graph with their positions and weights

        for i in range(N):
            G.add_node(i, pos=(node_pos[i,0].item(), node_pos[i,1].item()), weight=node_w[i].item())

        # Add edges to   the graph with their weights
        N_edge = edge_idx.shape[1]
        for i in range(N_edge):
            G.add_edge(edge_idx[0, i].item(), edge_idx[1, i].item(), weight=edge_w[i].item())

        #statstic
        for i, node in enumerate(G.nodes()):
            weight_list = [edge[2]['weight'] for edge in G.edges(node, data=True)]
            weight = sum(weight_list) #/ len(weight_list)
            if i == gt_id:
                all_inlier.append(weight)
            else:
                all_outlier.append(weight)

    
        
    n1, bins1, _ = plt.hist(all_inlier, bins=50, density=True, histtype='step', color='green', alpha=0)
    n2, bins2, _ = plt.hist(all_outlier, bins=50, density=True, histtype='step', color='red', alpha=0)
    # plt.close()
    n1 = n1 / np.sum(n1)
    n2 = n2 / np.sum(n2)

    fig, axs = plt.subplots(2,2)
    axs = axs.flatten()

    ####################smooth and interp#############3

    # window_size = 7
    # order = 3
    # n1_smoothed = savgol_filter(n1, window_size, order)
    # n2_smoothed = savgol_filter(n2, window_size, order)

    # interp_func1 = interpolate.interp1d(bins1[:-1], n1_smoothed, kind='cubic')
    # interp_func2 = interpolate.interp1d(bins2[:-1], n2_smoothed, kind='cubic')

    # # create a new range of values for the x-axis to plot the smooth curves
    # xnew = np.linspace(max(bins1[0], bins2[0]), min(bins1[-2], bins2[-2]), num=100, endpoint=True)

    # ynew1 = interp_func1(xnew)
    # ynew2 = interp_func2(xnew)

    # different = ynew1 - ynew2
    # # plot the smoothed curves
    
    # axs[0].plot(xnew, ynew1, color='green', label='Inliers')
    # axs[0].plot(xnew, ynew2, color='red', label='Outliers')
    # axs[1].plot(xnew, different, color='black')
    ####################################################

    # axs[0].set_aspect('equal', 'box')
    # axs[1].set_aspect('equal', 'box')

    # plot original 
    axs[0].plot(bins1[:-1], n1, color='green', label='Inliers')
    axs[0].plot(bins2[:-1], n2, color='red', label='Outliers')
    # axs[1].plot(xnew, different, color='black')

    # set the x-axis and y-axis labels
    # axs[0].set_xlabel('Sum of edges')
    # axs[0].set_ylabel('Frequency')

    # add a legend to the plot
    # axs[0].legend(['Inliers', 'Outliers'])
    plt.savefig('edge_weight_statstic.png')
    plt.close()
    

    

    
