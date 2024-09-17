from pathlib import Path
from typing import Optional, Tuple

import pygame

import engine.bezier_curve as bc
from engine.bezier_curve import BezierCurve


class TrackEditor:
    """
    A class to represent the track editor.
    """

    curve: Optional[bc.BezierCurve]
    selected_point: Tuple[Optional[int], int]
    is_creating: bool
    debug_point_size: int

    DEFAULT_DIRECTORY = Path("data/tracks")

    def __init__(self, debug_point_size: int = 6):
        self.curve = None
        self.selected_point = (None, 0)
        self.is_creating = False
        self.debug_point_size = debug_point_size

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

        track_file.write_text(self.curve.serialize() if self.curve is not None else "")

    def on_mouse_up(self, button: int):
        """
        Handle the mouse up event.
        :param button: The button that was pressed
        :return: None
        """
        if button != 1:
            return

        x, y = pygame.mouse.get_pos()

        if self.curve is None:
            self.is_creating = True
            self.curve = bc.BezierCurve([bc.BezierCurvePoint(x, y, x, y)])
            self.curve.pts.append(bc.BezierCurvePoint(x, y, x, y))
            self.selected_point = (1, 0)
        elif self.is_creating:
            self.curve.pts.append(bc.BezierCurvePoint(x, y, x, y))
            self.selected_point = (len(self.curve.pts) - 1, 0)
        else:
            self.selected_point = (None, 0)

    def on_mouse_down(self, button: int):
        """
        Handle the mouse down event.
        :param button: The button that was pressed
        :return: None
        """
        if button != 1 or self.curve is None:
            return

        x, y = pygame.mouse.get_pos()

        if self.is_creating:
            self.selected_point = (self.selected_point[0], 1)
        elif self.selected_point == (None, 0):
            for i in range(len(self.curve.pts)):
                dist = (self.curve.pts[i].x - x) ** 2 + (self.curve.pts[i].y - y) ** 2
                control_dist = (self.curve.pts[i].control_x - x) ** 2 + (
                    self.curve.pts[i].control_y - y
                ) ** 2
                opposite_control_dist = (
                    self.curve.pts[i].opposite_control_point()[0] - x
                ) ** 2 + (self.curve.pts[i].opposite_control_point()[1] - y) ** 2

                if control_dist < self.debug_point_size**2:
                    self.selected_point = (i, 1)
                    break
                elif opposite_control_dist < self.debug_point_size**2:
                    self.selected_point = (i, 2)
                    break
                elif dist < self.debug_point_size**2:
                    self.selected_point = (i, 0)
                    break
            else:
                self.is_creating = True
                self.curve.pts.append(bc.BezierCurvePoint(x, y, x, y))
                self.selected_point = (len(self.curve.pts) - 1, 1)

    def on_mouse_moved(self, screen: pygame.Surface):
        """
        Handle the mouse moved event.
        :return: None
        """
        if self.curve is None:
            return

        x, y = pygame.mouse.get_pos()

        if self.selected_point != (None, 0):
            if self.selected_point[1] == 0:
                bounded_x = max(0, min(x, screen.get_width()))
                bounded_y = max(0, min(y, screen.get_height()))
                self.curve.pts[self.selected_point[0]] = self.curve.pts[
                    self.selected_point[0]
                ].translated(
                    bounded_x - self.curve.pts[self.selected_point[0]].x,
                    bounded_y - self.curve.pts[self.selected_point[0]].y,
                )
            elif self.selected_point[1] == 1:
                self.curve.pts[self.selected_point[0]].control_x = x
                self.curve.pts[self.selected_point[0]].control_y = y
            elif self.selected_point[1] == 2:
                self.curve.pts[self.selected_point[0]].control_x = self.curve.pts[
                    self.selected_point[0]
                ].x - (x - self.curve.pts[self.selected_point[0]].x)
                self.curve.pts[self.selected_point[0]].control_y = self.curve.pts[
                    self.selected_point[0]
                ].y - (y - self.curve.pts[self.selected_point[0]].y)

    def on_key_pressed(self, key: int):
        """
        Handle the key pressed event.
        :param key: The key that was pressed
        :return: None
        """
        if self.curve is None:
            return

        if (
            key in (pygame.K_DELETE, pygame.K_BACKSPACE, pygame.K_ESCAPE) or
            key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL
        ):
            self.selected_point = (None, 0)
            self.is_creating = False
            self.curve.pts.pop(len(self.curve.pts) - 1)

        if not self.curve.pts:
            self.curve = None

    def draw_editing(
        self,
        screen: pygame.Surface,
        line_color: pygame.Color = pygame.Color(0, 0, 0),
        line_width: int = 3,
        edit_color: pygame.Color = pygame.Color(255, 0, 0),
        edit_width: int = 1,
    ):
        """
        Draw the editing.
        :param screen: The screen to draw on
        :param line_color: The color of the line
        :param line_width: The width of the line
        :param edit_color: The color of the edit
        :param edit_width: The width of the edit
        :return: None
        """
        if self.curve is not None:
            self.curve.draw(screen, line_color, line_width)
            self.curve.draw_debug(screen, edit_color, edit_width)
