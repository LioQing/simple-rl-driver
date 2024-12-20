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
    pygame.init()
    pygame.display.set_caption("Simple RL Driver - Track Editor")
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(
        args.resolution,
        (pygame.FULLSCREEN if args.fullscreen else 0) | pygame.RESIZABLE,
    )

    # Setup
    editor = TrackEditor.load(args.track)

    # Main loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                editor.on_mouse_down(event.button)
            elif event.type == pygame.MOUSEBUTTONUP:
                editor.on_mouse_up(event.button)
            elif event.type == pygame.MOUSEMOTION:
                editor.on_mouse_moved(screen)
            elif event.type == pygame.KEYDOWN:
                editor.on_key_pressed(event.key, screen)

                if (
                    pygame.key.get_mods() & pygame.KMOD_CTRL
                    and event.key == pygame.K_s
                ):
                    editor.save(args.track)
                if (
                    pygame.key.get_mods() & pygame.KMOD_CTRL
                    and event.key == pygame.K_q
                ):
                    running = False

        # Render
        screen.fill((255, 255, 255))
        editor.draw(
            screen,
            pygame.Color(0, 0, 0),
            int(Track.WIDTH / Track.SCALE),
        )

        pygame.display.update()

        clock.tick(60)

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
