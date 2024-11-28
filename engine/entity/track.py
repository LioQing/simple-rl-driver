from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt
import pyclipper
import pygame
import shapely

import engine.bezier_curve as bc
from engine.entity.camera import Camera
from engine.utils import vec


class Track:
    """
    A class representing the track in game.
    """

    curve: bc.BezierCurve
    width: int
    polyline_factor: float
    polyline: List[npt.NDArray[np.float32]]
    polygon: List[npt.NDArray[np.float32]]

    # Shapely objects for memoization
    shapely_linear_ring: shapely.LinearRing
    shapely_polygon: shapely.Polygon

    WIDTH = 100
    SCALE = 5

    def __init__(
        self,
        curve: bc.BezierCurve,
        width: int,
        polyline_factor: float = 0.05,
    ):
        self.curve = curve
        self.width = width
        self.polyline_factor = polyline_factor

        self.polyline = list(self.curve.get_polyline(self.polyline_factor))

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(self.polyline, pyclipper.JT_ROUND, pyclipper.ET_OPENROUND)
        self.polygon = [vec(*p) for p in pco.Execute(self.width)[0]]

        self.shapely_linear_ring = shapely.LinearRing(self.polygon)
        self.shapely_polygon = shapely.Polygon(self.polygon)

    def get_start_dir(self) -> npt.NDArray[np.float32]:
        """
        Get the starting direction of the Bézier curve.

        :return: The starting direction
        """
        return (
            0.5**3 * self.curve.pts[0].pos
            + 3 * 0.5**2 * 0.5 * self.curve.pts[0].control
            + 3 * 0.5 * 0.5**2 * self.curve.pts[1].opp_control
            + 0.5**3 * self.curve.pts[1].pos
        )

    @staticmethod
    def load(
        name: str,
        directory: Path = Path("data/tracks"),
        polyline_factor: float = 0.05,
    ) -> "Track":
        """
        Load the track from file system.

        :param name: The name of the track to load
        :param directory: The directory to load the track from
        :param width: The width of the track
        :param scale: The scale of the track
        :param polyline_factor: The polyline factor
        :return: The track
        """
        track_file = directory / f"{name}.txt"

        if not track_file.exists():
            raise FileNotFoundError(f"Track file {track_file} does not exist")

        curve = bc.BezierCurve.deserialize(track_file.read_text())
        curve.pts = [
            bc.BezierCurvePoint(
                p.pos * Track.SCALE,
                p.control * Track.SCALE,
            )
            for p in (p.translated(-curve.pts[0].pos) for p in curve.pts)
        ]

        return Track(curve, Track.WIDTH, polyline_factor)

    def draw(
        self,
        screen: pygame.Surface,
        camera: Camera,
        width: int = 1,
    ):
        """
        Draw the track on the screen.

        :param screen: The screen to draw on
        :param camera: The camera to use
        :param width: The width of the track edges
        :return: None
        """
        pygame.draw.polygon(
            screen,
            pygame.Color(0, 0, 0),
            [camera.get_coord(p) for p in self.polygon],
            width,
        )
