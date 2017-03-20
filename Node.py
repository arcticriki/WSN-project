class Node(object):

    def __init__(self, ID, X, Y):
        self.ID = ID
        self.X = X
        self.Y = Y

    def node_write(self, ID, X, Y):
        self.ID = ID
        self.X = X
        self.Y = Y

    def spec(self):
        print 'Node ID is %d and its position is (x=%d, y=%d)' % (self.ID, self.X, self.Y)
