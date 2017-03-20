class Storage(object):

    def __init__(self, ID, X, Y):
        self.ID = ID
        self.X = X
        self.Y = Y
        self.neighbor_list = []

    def node_write(self, ID, X, Y): #change ID and position coordinates
        self.ID = ID
        self.X = X
        self.Y = Y

    def spec(self):
        print 'Storage ID is %d and its position is (x=%d, y=%d)' % (self.ID, self.X, self.Y)

    def get_pos(self):
        return self.X, self.Y

    def neighbor_write(self, neighbor):
        self.neighbor_list.append(neighbor)


class Sensor(Storage):

    def __init__(self, ID, X, Y):
        self.ID = ID
        self.X = X
        self.Y = Y
        self.neighbor_list = []
    def spec(self):
        print 'Sensor ID is %d and its position is (x=%d, y=%d)' % (self.ID, self.X, self.Y)

