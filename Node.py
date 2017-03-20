class Node(object):

    def __init__(self, ID, X, Y):
        self.ID = ID
        self.X = X
        self.Y = Y
        self.neighbor_list = []

    def node_write(self, ID, X, Y, neighbor): #change ID and position coordinates
        self.ID = ID
        self.X = X
        self.Y = Y
        #self.neighbor_list.append(neighbor)

    def spec(self):
        print 'Node ID is %d and its position is (x=%d, y=%d)' % (self.ID, self.X, self.Y)
        print self.neighbor_list

    def get_pos(self):
        return self.X, self.Y

    def neighbor_write(self, neighbor):
        self.neighbor_list.append(neighbor)


