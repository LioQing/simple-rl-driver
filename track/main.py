import argparse

import pygame

from engine.entity.track import Track
from track.editor import TrackEditor

DESCRIPTION = (
    "Track editor mode.\n"
    "\n"
    "controls:\n"
    "  press left mouse button         add a point\n"
    "  hold and drag left mouse button edit the curve\n"
    "  release left mouse button       start editing next point\n"
    "  ctrl + s                        save the track\n"
    "  ctrl + z, del, backspace, esc   undo\n"
    "  ctrl + h                        flip the track horizontally\n"
    "  ctrl + v                        flip the track vertically\n"
    "  ctrl + q                        quit the program\n"
)


def main_scene(args: argparse.Namespace):
    """
    Main scene for editing the track

    :param args: The arguments
    :return: None
    """
    # Initialize pygame by calling `init`
    pygame.init()

    # Set the window title with `set_caption`
    pygame.display.set_caption("Simple RL Driver - Track Editor")

    # Create a clock object to help control the frame rate
    clock = pygame.time.Clock()

    # Create a screen with `set_mode`
    #
    # `args.resolution` is a tuple[int, int] in the form of (width, height)
    #
    # `args.fullscreen` is a boolean indicating whether to run in fullscreen
    # mode
    screen = pygame.display.set_mode(
        args.resolution,
        (pygame.FULLSCREEN if args.fullscreen else 0) | pygame.RESIZABLE,
    )

    # Setup the track editor by loading the track
    #
    # `args.track` is a string representing the name of the track to edit
    editor = TrackEditor.load(args.track)

    # Main loop forever while `running` is True
    running = True
    while running:
        # Handle events from `pygame.event.get()`
        #
        # It returns a list of all the events that have happened since the last
        # time we called the function, i.e. since the last frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # If it is a quit event, we stop the loop
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # If the mouse is pressed, we call the `on_mouse_down` method
                # of the editor
                editor.on_mouse_down(event.button)
            elif event.type == pygame.MOUSEBUTTONUP:
                # If the mouse is released, we call the `on_mouse_up` method
                # of the editor
                editor.on_mouse_up(event.button)
            elif event.type == pygame.MOUSEMOTION:
                # If the mouse is moved, we call the `on_mouse_moved` method
                # of the editor
                editor.on_mouse_moved(screen)
            elif event.type == pygame.KEYDOWN:
                # If a key is pressed, we call the `on_key_pressed` method of
                # the editor
                editor.on_key_pressed(event.key, screen)

                if (
                    pygame.key.get_mods() & pygame.KMOD_CTRL
                    and event.key == pygame.K_s
                ):
                    # If control + s is pressed, we save the track to the file
                    # same as the one we loaded, i.e. `args.track`
                    editor.save(args.track)

                if (
                    pygame.key.get_mods() & pygame.KMOD_CTRL
                    and event.key == pygame.K_q
                ):
                    # If control + q is pressed, we quit the program, same as
                    # when quit event is received
                    running = False

        # Clear the screen with white color by using `fill` method on the
        # screen object
        screen.fill((255, 255, 255))

        # Draw the editor on the screen by calling the `draw` method of the
        # editor with the screen, the color to draw the track with, and the
        # width of the track
        editor.draw(
            screen,
            pygame.Color(0, 0, 0),
            int(Track.WIDTH / Track.SCALE),
        )

        # Update the display with `update`
        pygame.display.update()

        # Tick the clock to control the frame rate
        clock.tick(60)

    # Quit pygame with `quit`
    pygame.quit()


def configure_parser(parser: argparse.ArgumentParser):
    """
    Configure the parser for the program

    :param parser: The parser to configure
    :return: None
    """
    parser.add_argument(
        "--track",
        "-t",
        dest="track",
        type=str,
        help="The name of the track to edit",
        required=True,
    )
    parser.add_argument(
        "--resolution",
        dest="resolution",
        type=int,
        nargs=2,
        help="The resolution of the track",
        default=(800, 640),
    )
    parser.add_argument(
        "--fullscreen",
        dest="fullscreen",
        action="store_true",
        help="Whether to run in fullscreen mode",
    )


def main():
    """
    Entry point for the program in track mode

    :return: None
    """
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    configure_parser(parser)
    args = parser.parse_args()
    main_scene(args)


if __name__ == "__main__":
    main()
