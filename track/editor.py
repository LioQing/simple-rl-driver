from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pygame

import engine.bezier_curve as bc
from engine.bezier_curve import BezierCurve


class TrackEditor:
    """
    A class to represent the track editor.
    """

    @dataclass
    class PointState:
        """
        Editing a point.
        """

        index: int
        is_new: bool = False

    @dataclass
    class ControlState:
        """
        Editing a control point.
        """

        index: int
        is_new: bool = False

    @dataclass
    class OppositeControlState:
        """
        Editing an opposite control point.
        """

        index: int

    curve: bc.BezierCurve
    edit: Optional[Union[PointState, ControlState, OppositeControlState]]
    point_size: int

    DEFAULT_POINT_SIZE = 6
    DEFAULT_DIRECTORY = Path("data/tracks")

    def __init__(self, point_size: int = DEFAULT_POINT_SIZE):
        self.curve = bc.BezierCurve()
        self.edit = None
        self.point_size = point_size

    @staticmethod
    def load(name: str, directory: Path = DEFAULT_DIRECTORY) -> "TrackEditor":
        """
        Load the track editor for the given track from file system,
        creates a new file if it does not exist.
        :param name: The name of the track to load
        :param directory: The directory to load the track from
        :return: The track editor
        """
        editor = TrackEditor()

        track_file = directory / f"{name}.txt"

        if not track_file.exists():
            return editor

        editor.curve = BezierCurve.deserialize(track_file.read_text())

        return editor

    def save(self, name: str, directory: Path = DEFAULT_DIRECTORY):
        """
        Save the track editor to file system.
        :param name: The name of the track to save
        :param directory: The directory to save the track to
        :return: None
        """
        track_file = directory / f"{name}.txt"

        if not track_file.exists():
            track_file.parent.mkdir(parents=True, exist_ok=True)
            track_file.touch()

        track_file.write_text(self.curve.serialize())

    def on_mouse_up(self, button: int):
        """
        Handle the mouse up event.
        :param button: The button that was pressed
        :return: None
        """
        if button != 1:
            return

        x, y = pygame.mouse.get_pos()

        if (
            isinstance(self.edit, TrackEditor.ControlState)
            and self.edit.is_new
        ):
            self.curve.pts.append(bc.BezierCurvePoint(x, y, x, y))
            self.edit = TrackEditor.PointState(
                len(self.curve.pts) - 1, is_new=True
            )
        else:
            self.edit = None

    def on_mouse_down(self, button: int):
        """
        Handle the mouse down event.
        :param button: The button that was pressed
        :return: None
        """
        if button != 1:
            return

        x, y = pygame.mouse.get_pos()

        if self.edit is None:
            for i, pt in enumerate(self.curve.pts):
                dist = (pt.x - x) ** 2 + (pt.y - y) ** 2
                control_dist = (pt.control_x - x) ** 2 + (
                    pt.control_y - y
                ) ** 2
                opposite_control_dist = (
                    pt.opposite_control_point()[0] - x
                ) ** 2 + (pt.opposite_control_point()[1] - y) ** 2

                if control_dist < self.point_size**2:
                    self.edit = TrackEditor.ControlState(i)
                    break
                elif opposite_control_dist < self.point_size**2:
                    self.edit = TrackEditor.OppositeControlState(i)
                    break
                elif dist < self.point_size**2:
                    self.edit = TrackEditor.PointState(i)
                    break
            else:
                self.curve.pts.append(bc.BezierCurvePoint(x, y, x, y))
                self.edit = TrackEditor.ControlState(
                    len(self.curve.pts) - 1, is_new=True
                )
        elif isinstance(self.edit, TrackEditor.PointState):
            self.edit = TrackEditor.ControlState(
                self.edit.index, is_new=self.edit.is_new
            )

    def on_mouse_moved(self, screen: pygame.Surface):
        """
        Handle the mouse moved event.
        :param screen: The screen
        :return: None
        """
        x, y = pygame.mouse.get_pos()

        if isinstance(self.edit, TrackEditor.PointState):
            bounded_x = max(0, min(x, screen.get_width()))
            bounded_y = max(0, min(y, screen.get_height()))
            self.curve.pts[self.edit.index].translate(
                bounded_x - self.curve.pts[self.edit.index].x,
                bounded_y - self.curve.pts[self.edit.index].y,
            )
        elif isinstance(self.edit, TrackEditor.ControlState):
            self.curve.pts[self.edit.index].control_x = x
            self.curve.pts[self.edit.index].control_y = y
        elif isinstance(self.edit, TrackEditor.OppositeControlState):
            self.curve.pts[self.edit.index].control_x = self.curve.pts[
                self.edit.index
            ].x - (x - self.curve.pts[self.edit.index].x)
            self.curve.pts[self.edit.index].control_y = self.curve.pts[
                self.edit.index
            ].y - (y - self.curve.pts[self.edit.index].y)

    def on_key_pressed(self, key: int):
        """
        Handle the key pressed event.
        :param key: The key that was pressed
        :return: None
        """
        if self.curve.pts and (
            key in (pygame.K_DELETE, pygame.K_BACKSPACE, pygame.K_ESCAPE)
            or key == pygame.K_z
            and pygame.key.get_mods() & pygame.KMOD_CTRL
        ):
            self.edit = None
            self.curve.pts.pop(len(self.curve.pts) - 1)

    def draw(
        self,
        screen: pygame.Surface,
        line_color: pygame.Color = pygame.Color(0, 0, 0),
        line_width: int = 3,
        edit_color: pygame.Color = pygame.Color(255, 0, 0),
        edit_width: int = 1,
    ):
        """
        Draw the editor.
        :param screen: The screen to draw on
        :param line_color: The color of the line
        :param line_width: The width of the line
        :param edit_color: The color of the edit
        :param edit_width: The width of the edit
        :return: None
        """
        self.curve.draw(screen, line_color, line_width)
        self.curve.draw_edit(screen, edit_color, edit_width)
