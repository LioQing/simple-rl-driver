import math


class Transformable:
    """
    A class to represent a Transformable object.
    """

    x: float
    y: float
    rot: float

    def __init__(self, x: float = 0, y: float = 0, rot: float = 0):
        """
        Constructs all the necessary attributes for the Transformable object.
        :param x: The x coordinate
        :param y: The y coordinate
        :param rot: The rotation
        """
        self.x = x
        self.y = y
        self.rot = rot

    def translate(self, x: float, y: float):
        self.x += x
        self.y += y

    def translate_forward(self, dist: float):
        self.translate(dist * math.sin(self.rot), dist * math.cos(self.rot))

    def rotate(self, rad: float):
        self.rot = (self.rot + rad) % (2 * math.pi)