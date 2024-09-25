from pathlib import Path
from typing import List, Tuple

import engine.bezier_curve as bc
from engine.entity.camera import Camera
import pygame
import pyclipper


class Track:
    curve: bc.BezierCurve
    width: int
    polyline_factor: float
    polyline: List[Tuple[float, float]]
    polygon: List[Tuple[float, float]]

    DEFAULT_POLYLINE_FACTOR = 0.05
    DEFAULT_DIRECTORY = Path("data/tracks")
    DEFAULT_WIDTH = 100
    DEFAULT_SCALE = 5

    def __init__(
        self,
        curve: bc.BezierCurve,
        width: int,
        polyline_factor: float = DEFAULT_POLYLINE_FACTOR,
    ):
        self.curve = curve
        self.width = width
        self.polyline_factor = polyline_factor

        self.polyline = list(self.curve.get_polyline(self.polyline_factor))

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(self.polyline, pyclipper.JT_ROUND, pyclipper.ET_OPENROUND)
        self.polygon = pco.Execute(self.width)[0]

    @staticmethod
    def load(
        name: str,
        directory: Path = DEFAULT_DIRECTORY,
        width: int = DEFAULT_WIDTH,
        scale: int = DEFAULT_SCALE,
        polyline_factor: float = DEFAULT_POLYLINE_FACTOR,
    ) -> "Track":
        track_file = directory / f"{name}.txt"

        if not track_file.exists():
            raise FileNotFoundError(f"Track file {track_file} does not exist")

        curve = bc.BezierCurve.deserialize(track_file.read_text())
        curve.pts = [
            p.translated(-curve.pts[0].x, -curve.pts[0].y).scaled(scale)
            for p in curve.pts
        ]

        return Track(curve, width, polyline_factor)

    def draw(
        self,
        screen: pygame.Surface,
        color: pygame.Color,
        camera: Camera,
        width: int = 1,
    ):
        pygame.draw.polygon(
            screen, color, [camera.get_coord(x, y) for x, y in self.polygon], width
        )
