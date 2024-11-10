def lerp(a, b, t):
    """
    Linear interpolate from a to b by t
    :param a: The start value
    :param b: The end value
    :param t: The interpolation value
    :return: The interpolated value
    """
    return a + (b - a) * t
