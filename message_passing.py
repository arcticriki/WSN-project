import numpy as np              # import of package numpy for mathematical tools
import random as rnd            # import of package random for homonym tools
import copy

payload = 1
def message_passing(node_list,n, k, h):
    decoding_indices = rnd.sample(range(0, n), h)  # selecting h random nodes in the graph
    degrees = [0] * h           # Initialize ancillary variables
    IDs     = [0] * h
    XORs    = [0] * h

    for node in range(h):       # Retrieve the h nodes
        degree, ID    = node_list[decoding_indices[node]].storage_info()
        degrees[node] = copy.deepcopy(degree)               # deepcopy of the information
        IDs[node]     = copy.deepcopy(ID)

    # -- MP. Naive approach --------------------------------
    ripple_IDs = []             # auxialiary vectors
    hashmap = np.zeros((n, 2))  # vector nx2: pos[ID-1,0]-> "1" pkt of (ID-1) is decoded, "0" otherwise; pos[ID-1,1]->num_hashmap
    num_hashmap = 0             # key counter: indicates the number of decoded packets
    empty_ripple = False

    while (empty_ripple == False):
        empty_ripple = True
        position = 0            # linear search of degree one nodes
        while position < len(degrees):

            if degrees[position] == 1:                      # if degree 1 is found, the packet is resolved
                if hashmap[IDs[position][0] - 1, 0] == 0:   # We save the ID and increase the number of decoded packets
                    hashmap[IDs[position][0] - 1, 0] = 1
                    hashmap[IDs[position][0] - 1, 1] = num_hashmap
                    num_hashmap += 1
                empty_ripple = False                        # if degree 1 pkt are found, they are saved in the ripple
                del degrees[position]                       # decrease degree
                ripple_IDs.append(IDs[position])            # update ripples
                del IDs[position]                           # update IDs
            else:
                position = position + 1

        # scanning the ripple
        for each_element in ripple_IDs:                     # consider each element in the ripple
            for each_node in IDs:                           # consider each element in IDs
                u = 0
                while u < len(each_node):
                    if each_element[0] == each_node[u]:     # remove from each node the ID which are in the ripple, decreasing the degree
                        indice_ID = IDs.index(each_node)
                        degrees[indice_ID] -= 1             # decrease the degree of a node
                        indice_ripple = ripple_IDs.index(each_element)
                        temp = each_node
                        del temp[u]
                        IDs[indice_ID] = temp
                        each_node = temp
                    else:
                        u += 1
        i = 0
        while i < len(IDs):         # remove each node with degree 0
            if degrees[i] == 0:
                IDs.remove([])
                degrees.remove(0)
            else:
                i += 1

    if num_hashmap < k:
        return 1    # if we do not decode the k pkts than we make an error
    else:
        return 0    # if we decoded all k packets, no errors must be considered
