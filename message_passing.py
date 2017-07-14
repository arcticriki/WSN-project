import numpy as np  # import of package numpy for mathematical tools
import random as rnd  # import of package random for homonym tools
import copy

payload = 10
def message_passing(node_list,n, k, h):
    decoding_indices = rnd.sample(range(0, n), h)  # selecting h random nodes in the graph
    degrees = [0] * h
    IDs     = [0] * h
    XORs    = [0] * h

    for node in range(h):
        degree, ID, XOR = node_list[decoding_indices[node]].storage_info()

        degrees[node] = copy.deepcopy(degree)
        IDs[node] = copy.deepcopy(ID)
        XORs[node] = copy.deepcopy(XOR)

    # -- MP. Naive approach --------------------------------

    ripple_payload = []  # auxialiary vectors
    ripple_IDs = []
    hashmap = np.zeros((n, 2))  # vector nx2: pos[ID-1,0]-> "1" pkt of (ID-1) is decoded, "0" otherwise; pos[ID-1,1]->num_hashmap
    num_hashmap = 0  # key counter: indicates the index of the next free row in decoded matrix

    empty_ripple = False

    while (empty_ripple == False):

        empty_ripple = True

        position = 0  # linear search of degree one nodes

        while position < len(degrees):

            if degrees[position] == 1:  # if degree 1 is found
                if hashmap[IDs[position][0] - 1, 0] == 0:
                    hashmap[IDs[position][0] - 1, 0] = 1
                    hashmap[IDs[position][0] - 1, 1] = num_hashmap
                    num_hashmap += 1
                empty_ripple = False
                del degrees[position]  # decrease degree
                ripple_IDs.append(IDs[position])  # update ripples
                del IDs[position]
                ripple_payload.append(XORs[position])
                del XORs[position]  # update vector XORs
            else:
                position = position + 1

        # scanning the ripple
        for each_element in ripple_IDs:  # prendi ogni elemento del ripple...
            for each_node in IDs:  # ...e ogni elemento del vettore degli ID...
                u = 0
                while u < len(each_node):
                    if each_element[0] == each_node[u]:
                        indice_ID = IDs.index(each_node)
                        degrees[indice_ID] -= 1
                        indice_ripple = ripple_IDs.index(each_element)
                        XORs[indice_ID] = XORs[indice_ID] ^ ripple_payload[indice_ripple]
                        temp = each_node
                        del temp[u]
                        IDs[indice_ID] = temp
                        each_node = temp

                    else:
                        u += 1

        i = 0
        while i < len(IDs):
            if degrees[i] == 0:
                IDs.remove([])
                # XORs.remove(XORs[i])
                del XORs[i]
                degrees.remove(0)
            else:
                i += 1

    if num_hashmap < k:
        return 1  # if we do not decode the k pkts that we make an error
    else:
        return 0
