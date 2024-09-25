from typing import Tuple, List, Iterable

import numpy as np
import pygame


class BezierCurvePoint:
    """
    A class to represent a point in a Bézier curve.
    """

    x: int
    y: int
    control_x: int
    control_y: int

    def __init__(self, x: int, y: int, control_x: int, control_y: int):
        self.x = x
        self.y = y
        self.control_x = control_x
        self.control_y = control_y

    def relative_control_point(self) -> Tuple[int, int]:
        """
        Get the relative control point.
        :return: The relative control point
        """
        return self.control_x - self.x, self.control_y - self.y

    def opposite_control_point(self) -> Tuple[int, int]:
        """
        Get the opposite control point.
        :return: The opposite control point
        """
        return (
            self.x - self.relative_control_point()[0],
            self.y - self.relative_control_point()[1],
        )

    def translated(self, x: int, y: int) -> "BezierCurvePoint":
        """
        Get the translated point.
        :param x: The x translation
        :param y: The y translation
        :return: The translated point
        """
        return BezierCurvePoint(
            self.x + x, self.y + y, self.control_x + x, self.control_y + y
        )

    def translate(self, x: int, y: int):
        """
        Translate the point.
        :param x: The x translation
        :param y: The y translation
        :return: None
        """
        self.x += x
        self.y += y
        self.control_x += x
        self.control_y += y

    def scaled(self, scale: int) -> "BezierCurvePoint":
        """
        Get the scaled point.
        :param scale: The scale factor
        :return: The scaled point
        """
        return BezierCurvePoint(
            self.x * scale,
            self.y * scale,
            self.control_x * scale,
            self.control_y * scale,
        )

    def scale(self, scale: int):
        """
        Scale the point.
        :param scale: The scale factor
        :return: None
        """
        self.x *= scale
        self.y *= scale
        self.control_x *= scale
        self.control_y *= scale


class BezierCurve:
    """
    A class to represent a Bézier curve.
    """

    pts: List[BezierCurvePoint]
    debug_point_size: int

    def __init__(self, pts: Iterable[BezierCurvePoint] = (), debug_point_size: int = 6):
        self.pts = list(pts)
        self.debug_point_size = debug_point_size

    def get_polyline(self, steps: float = 0.01) -> Iterable[Tuple[int, int]]:
        """
        Get the polyline of the Bézier curve.
        :param steps: The number of steps to take between each pair of points
        :return: The polyline of the Bézier curve
        """
        return (
            (
                int(
                    (1 - i) ** 3 * p1.x
                    + 3 * (1 - i) ** 2 * i * p1.control_x
                    + 3 * (1 - i) * i**2 * p2.opposite_control_point()[0]
                    + i**3 * p2.x
                ),
                int(
                    (1 - i) ** 3 * p1.y
                    + 3 * (1 - i) ** 2 * i * p1.control_y
                    + 3 * (1 - i) * i**2 * p2.opposite_control_point()[1]
                    + i**3 * p2.y
                ),
            )
            for p1, p2 in zip(self.pts, self.pts[1:])
            for i in np.arange(0, 1, steps)
        )

    def draw_edit(self, surface: pygame.Surface, color: pygame.Color, width: int = 1):
        """
        Draw the Bézier curve in edit mode.
        :param surface: The surface to draw on
        :param color: The color to draw with
        :param width: The width of the lines
        :return: None
        """
        for p in self.pts:
            pygame.draw.circle(surface, color, (p.x, p.y), self.debug_point_size, 0)
            pygame.draw.circle(
                surface, color, (p.control_x, p.control_y), self.debug_point_size, 0
            )
            pygame.draw.circle(
                surface, color, p.opposite_control_point(), self.debug_point_size, 1
            )
            pygame.draw.line(
                surface,
                color,
                p.opposite_control_point(),
                (p.control_x, p.control_y),
                width,
            )

    def draw(
        self,
        surface: pygame.Surface,
        color: pygame.Color,
        width: int = 1,
        steps: float = 0.01,
    ):
        """
        Draw the Bézier curve.
        :param surface: The surface to draw on
        :param color: The color to draw with
        :param width: The width of the lines
        :param steps: The number of steps to take between each pair of points
        :return: None
        """
        polyline = list(self.get_polyline(steps))

        for p1, p2 in zip(polyline, polyline[1:]):
            pygame.draw.line(surface, color, p1, p2, width)

    def serialize(self) -> str:
        """
        Serialize the Bézier curve.
        :return: The serialized Bézier curve string
        """
        return "\n".join(f"{p.x} {p.y} {p.control_x} {p.control_y}" for p in self.pts)

    @staticmethod
    def deserialize(data: str) -> "BezierCurve":
        """
        Deserialize the Bézier curve.
        :param data: The serialized Bézier curve string
        :return: The deserialized Bézier curve
        """
        return BezierCurve(
            BezierCurvePoint(*map(int, line.strip().split()))
            for line in data.split("\n")
            if line.strip()
        )
