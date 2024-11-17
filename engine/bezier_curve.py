from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import numpy.typing as npt
import pygame

from engine.utils import vec


@dataclass
class BezierCurvePoint:
    """
    A class to represent a point in a Bézier curve.
    """

    pos: npt.NDArray[np.int32]
    control: npt.NDArray[np.int32]

    @property
    def local_control(self) -> npt.NDArray[np.int32]:
        """
        Get the local control point.

        :return: The local control point
        """
        return self.control - self.pos

    @property
    def opp_control(self) -> npt.NDArray[np.int32]:
        """
        Get the opposite control point.

        :return: The opposite control point
        """
        return self.pos - self.local_control

    def move_to(self, pos: npt.NDArray[np.int32]):
        """
        Move the point to a new position.

        :param pos: The new position
        :return: None
        """
        self.control = pos + self.local_control
        self.pos = pos

    def moved_to(self, pos: npt.NDArray[np.int32]) -> "BezierCurvePoint":
        """
        Get a new point moved to a new position.

        :param pos: The new position
        :return: The new point
        """
        return BezierCurvePoint(pos, pos + self.local_control)

    def translate(self, delta: npt.NDArray[np.int32]):
        """
        Translate the point.

        :param delta: The translation vector
        :return: None
        """
        self.pos += delta
        self.control += delta

    def translated(self, delta: npt.NDArray[np.int32]) -> "BezierCurvePoint":
        """
        Get a new point translated.

        :param delta: The translation vector
        :return: The new point
        """
        return BezierCurvePoint(self.pos + delta, self.control + delta)

    def serialize(self) -> str:
        """
        Serialize the point.

        :return: The serialized point string
        """
        return (
            f"{self.pos[0]} {self.pos[1]} {self.control[0]} {self.control[1]}"
        )

    @classmethod
    def deserialize(cls, data: str) -> "BezierCurvePoint":
        """
        Deserialize the point.

        :param data: The serialized point string
        :return: The deserialized point
        """
        x, y, cx, cy = map(int, data.strip().split())
        return BezierCurvePoint(
            vec(x, y, dtype=np.int32),
            vec(cx, cy, dtype=np.int32),
        )


class BezierCurve:
    """
    A class to represent a Bézier curve.
    """

    pts: List[BezierCurvePoint]
    debug_point_size: int

    def __init__(
        self, pts: Iterable[BezierCurvePoint] = (), debug_point_size: int = 6
    ):
        self.pts = list(pts)
        self.debug_point_size = debug_point_size

    def get_polyline(
        self, steps: float = 0.01
    ) -> Iterable[npt.NDArray[np.float32]]:
        """
        Get the polyline of the Bézier curve.

        :param steps: The number of steps to take between each pair of points
        :return: The polyline of the Bézier curve
        """
        return (
            (1 - i) ** 3 * p1.pos
            + 3 * (1 - i) ** 2 * i * p1.control
            + 3 * (1 - i) * i**2 * p2.opp_control
            + i**3 * p2.pos
            for p1, p2 in zip(self.pts, self.pts[1:])
            for i in np.arange(0, 1, steps)
        )

    def draw_edit(
        self, surface: pygame.Surface, color: pygame.Color, width: int = 1
    ):
        """
        Draw the Bézier curve in edit mode.

        :param surface: The surface to draw on
        :param color: The color to draw with
        :param width: The width of the lines
        :return: None
        """
        for p in self.pts:
            pygame.draw.circle(surface, color, p.pos, self.debug_point_size, 0)
            pygame.draw.circle(
                surface,
                color,
                p.control,
                self.debug_point_size,
                0,
            )
            pygame.draw.circle(
                surface,
                color,
                p.opp_control,
                self.debug_point_size,
                1,
            )
            pygame.draw.line(
                surface,
                color,
                p.opp_control,
                p.control,
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
        return "\n".join(p.serialize() for p in self.pts)

    @classmethod
    def deserialize(cls, data: str) -> "BezierCurve":
        """
        Deserialize the Bézier curve.

        :param data: The serialized Bézier curve string
        :return: The deserialized Bézier curve
        """
        return BezierCurve(
            BezierCurvePoint.deserialize(line)
            for line in data.split("\n")
            if line.strip()
        )
