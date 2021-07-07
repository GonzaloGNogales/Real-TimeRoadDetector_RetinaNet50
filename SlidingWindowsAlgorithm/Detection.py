class Detection:
    def __init__(self, x1, y1, x2, y2, p, c):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.probability = p
        self.det_class = c

    def unpack(self):
        return self.x1, self.y1, self.x2, self.y2, self.probability, self.det_class

    def get_box_tensor(self):
        return [[self.x1, self.y1, self.x2, self.y2]]
