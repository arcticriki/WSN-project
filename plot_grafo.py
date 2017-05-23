import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_grafo(node_list, n , k , sensors_indexes):
    ii=-1
    x1 = np.zeros(n)
    y1 = np.zeros(n)

    plt.figure(figsize=(10, 10))

    for i in xrange(n):
        ii += 1
        x1[ii] = node_list[i].X
        y1[ii] = node_list[i].Y
        for iii in xrange(node_list[i].node_degree):
            #color = '#eeefff'
            xx = [node_list[i].X, node_list[i].neighbor_list[iii].X]
            yy = [node_list[i].Y, node_list[i].neighbor_list[iii].Y]
            plt.plot(xx,yy,color='grey')




    x2 = np.zeros(len(sensors_indexes))
    y2 = np.zeros(len(sensors_indexes))
    ii = -1
    for i in sensors_indexes:
        ii += 1
        x2[ii] = node_list[i].X
        y2[ii] = node_list[i].Y
        for iii in xrange(node_list[i].node_degree):
            xx = [node_list[i].X, node_list[i].neighbor_list[iii].X]
            yy = [node_list[i].Y, node_list[i].neighbor_list[iii].Y]
            plt.plot(xx,yy,color='red')

    plt.plot(x1, y1, color='grey', linestyle='', marker='o',markersize=10.0, markeredgewidth=1.0)
    plt.plot(x2, y2, color='red', linestyle='', marker ='o',markersize=10.0, markeredgewidth=1.0,
    markerfacecolor='red', markeredgecolor='black')
    plt.axis('off')
    plt.xlim(-0.05, 5.05)
    plt.ylim(-0.05,5.05)
    #plt.show()
    plt.savefig('Immagini/Grafo_n=' + str(n) + '_k=' + str(k) + '.pdf', dpi=150, transparent=True)
    print 'Immagine Grafo salvata!'


















    # G=nx.random_geometric_graph(200,0.125)
    # # position is stored as node attribute data for random_geometric_graph
    # pos=nx.get_node_attributes(G,'pos')
    #
    # # find node near center (0.5,0.5)
    # dmin=1
    # ncenter=0
    # for n in pos:
    #     x,y=pos[n]
    #     d=(x-0.5)**2+(y-0.5)**2
    #     if d<dmin:
    #         ncenter=n
    #         dmin=d
    #
    # # color by path length from node near center
    # p=nx.single_source_shortest_path_length(G,ncenter)
    #
    # plt.figure(figsize=(8,8))
    # nx.draw_networkx_edges(G,pos,nodelist=[ncenter],alpha=0.4)
    # nx.draw_networkx_nodes(G,pos,nodelist=p.keys(),
    #                        node_size=80,
    #                        node_color=p.values(),
    #                        cmap=plt.cm.Reds_r)
    #
    # plt.xlim(-0.05,1.05)
    # plt.ylim(-0.05,1.05)
    # plt.axis('off')
    # plt.savefig('random_geometric_graph.png')
    # plt.show()