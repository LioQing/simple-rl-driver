import math

class Transformable:
    """
    A class to represent a Transformable object.
    """

    x: float
    y: float
    rot: float

    def __init__(self, x: float = 0, y: float = 0, rot: float = 0):
        self.x = x
        self.y = y
        self.rot = rot

    def translate(self, x: float, y: float):
        """
        Translate the object.
        :param x: The x translation
        :param y: The y translation
        :return: None
        """
        self.x += x
        self.y += y

    def translate_forward(self, dist: float):
        """
        Translate the object forward.
        :param dist: The distance to translate
        :return: None
        """
        self.translate(dist * math.sin(self.rot), dist * math.cos(self.rot))

    def rotate(self, rad: float):
        """
        Rotate the object.
        :param rad: The rotation in radians
        :return: None
        """
        self.rot = (self.rot + rad) % (2 * math.pi)