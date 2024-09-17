from typing import Sequence, Tuple, List, Optional

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
        return self.control_x - self.x, self.control_y - self.y

    def opposite_control_point(self) -> Tuple[int, int]:
        return (
            self.x - self.relative_control_point()[0],
            self.y - self.relative_control_point()[1],
        )

    def translated(self, x: int, y: int) -> "BezierCurvePoint":
        return BezierCurvePoint(
            self.x + x, self.y + y, self.control_x + x, self.control_y + y
        )

    def scaled(self, scale: int) -> "BezierCurvePoint":
        return BezierCurvePoint(
            self.x * scale,
            self.y * scale,
            self.control_x * scale,
            self.control_y * scale,
        )


class BezierCurve:
    """
    A class to represent a Bézier curve.
    """

    pts: List[BezierCurvePoint]
    debug_point_size: int

    def __init__(self, pts: Sequence[BezierCurvePoint], debug_point_size: int = 6):
        self.pts = list(pts)
        self.debug_point_size = debug_point_size

    def get_polyline(self, steps: float = 0.01) -> List[Tuple[int, int]]:
        polyline = []
        for p1, p2 in zip(self.pts, self.pts[1:]):
            for i in np.arange(0, 1, steps):
                p = (
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
                polyline.append(p)
        return polyline

    def draw_debug(self, surface: pygame.Surface, color: pygame.Color, width: int = 1):
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
        polyline = self.get_polyline(steps)

        for p1, p2 in zip(polyline, polyline[1:]):
            pygame.draw.line(surface, color, p1, p2, width)

    def serialize(self) -> str:
        return "\n".join(
            f"{p.x} {p.y} {p.control_x} {p.control_y}"
            for p in self.pts
        )

    @staticmethod
    def deserialize(data: str) -> Optional["BezierCurve"]:
        pts = [
            BezierCurvePoint(*map(int, line.strip().split()))
            for line in data.split("\n")
            if line.strip()
        ]

        if not pts:
            return None

        return BezierCurve(pts)
