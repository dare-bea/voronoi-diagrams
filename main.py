from random import random, randrange
from PIL import Image, ImageDraw, ImageFont
from os import makedirs
from shutil import rmtree
import math
import numpy as np
import tkinter as tk
from tkinter import ttk
from sys import argv
import argparse
from pathlib import Path
from dataclasses import dataclass

# ----------------------------
# CONFIG
# ----------------------------

def render_frame(r: float,
                 /,
                 *,
                 height: int | None = None,
                 width: int | None = None,
                 grid: np.ndarray | None = None,
                 points: np.ndarray,
                 colors: np.ndarray,
                 font=ImageFont.load_default(),
                 draw=ImageDraw.Draw,
                 font_size: int = 0) -> Image.Image:
    if grid is None:
        if height is None or width is None:
            raise ValueError("Must provide either grid or height and width")
        yy, xx = np.indices((height, width))
        grid = np.stack((xx, yy), axis=-1)
        grid = grid[..., None, :]  # shape (H, W, 1, 2)

    HEIGHT = grid.shape[0]
    WIDTH = grid.shape[1]
    diff = np.abs(grid - points)  # (H, W, N, 2)

    if r is None or not np.isfinite(r) or r == math.inf:
        dists = np.max(diff, axis=-1)
    elif abs(r - 1.0) < 1e-9:
        dists = np.sum(diff, axis=-1)
    else:
        dists = np.power(np.sum(np.power(diff, r), axis=-1), 1 / r)
        bad = ~np.isfinite(dists)
        if np.any(bad):
            dists[bad] = np.max(diff[bad], axis=-1)

    nearest = np.argmin(dists, axis=-1)
    frame_rgb = (colors[nearest] * 255).astype(np.uint8)
    img = Image.fromarray(frame_rgb, "RGB")

    # Borders
    # border_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    # border_mask[:-1, :] |= nearest[:-1, :] != nearest[1:, :]
    # border_mask[:, :-1] |= nearest[:, :-1] != nearest[:, 1:]
    # frame_rgb[border_mask] = 0
    img = Image.fromarray(frame_rgb, "RGB")

    # Label
    if font_size:
        nimg = Image.new('RGB', (WIDTH, HEIGHT + font_size + 10))
        nimg.paste(img, (0, 0))
        draw = draw(nimg)
        draw.text((5, HEIGHT + 2), f'{r = :.3e}', fill='white', font=font)
        img = nimg

    return img

@dataclass
class GifGUI:
    root: tk.Tk
    status: ttk.Label
    progress: ttk.Progressbar

def rand_gif_main(cl_args: list[str]):
    DEFAULT_PATH = Path(__file__).parent / 'results' / 'voronoi.gif'
    FRAME_PATH = Path(__file__).parent / 'tmp'
    
    if DEFAULT_PATH.is_relative_to(Path.cwd()):
        display_path = DEFAULT_PATH.relative_to(Path.cwd())
    else:
        display_path = DEFAULT_PATH.absolute()
    parser = argparse.ArgumentParser(
        description='Generate a Voronoi diagram animation.'
    )
    parser.add_argument('--output', type=Path, help=f'Output file path. (default: {display_path})', default=Path(DEFAULT_PATH))
    parser.add_argument('--width', type=int, help='Width of the image. (default: 500)', default=500)
    parser.add_argument('--height', type=int, help='Height of the image. (default: 500)', default=500)
    parser.add_argument('--points', type=int, help='Number of points. (default: 100)', default=100)
    parser.add_argument('--fps', type=int, help='Frames per second. (default: 30)', default=30)
    parser.add_argument('--duration', type=float, help='Duration of the animation in seconds. (default: 5.0)', default=5.0)
    parser.add_argument('--hold-duration', type=float, help='Additional duration to hold the first and last frame. (default: 1.0)', default=1.0)
    parser.add_argument('--font-size', type=int, help='Font size. (default: 12)', default=12)
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI.')
    parser.add_argument('--save-frames', action='store_true', help='Save individual frames.')

    # Parse arguments
    args = parser.parse_args(cl_args)
    
    WIDTH = args.width
    HEIGHT = args.height
    POINTS = args.points
    FPS = args.fps
    FRAMES = int(args.duration * FPS)
    HOLD_FRAMES = int(args.hold_duration * FPS)
    DURATION = 1000 / FPS
    FONT_SIZE = args.font_size
    OUTPUT = args.output
    USE_GUI = not args.no_gui
    SAVE_FRAMES = args.save_frames

    print(
        f'Generating {FRAMES} frames of {WIDTH}x{HEIGHT} Voronoi with {POINTS} points...'
    )

    # ----------------------------
    # GUI SETUP
    # ----------------------------
    gui: GifGUI | None = None

    if USE_GUI:
        root = tk.Tk()
        status = ttk.Label(root, text='Generating points...')
        progress = ttk.Progressbar(root,
                                   orient='horizontal',
                                   maximum=FRAMES,
                                   mode='determinate')
        status.pack()
        progress.pack()
        root.geometry('300x50')
        root.update()
        gui = GifGUI(root, status, progress)
    else:
        print('Generating points...')

    # ----------------------------
    # POINT GENERATION
    # ----------------------------
    points = np.array([(randrange(WIDTH), randrange(HEIGHT))
                       for _ in range(POINTS)],
                      dtype=float)
    colors = np.array([(random(), random(), random()) for _ in range(POINTS)],
                      dtype=float)

    if SAVE_FRAMES:
        makedirs(FRAME_PATH, exist_ok=True)

    # ----------------------------
    # PRECOMPUTED GRID
    # ----------------------------
    yy, xx = np.indices((HEIGHT, WIDTH))
    grid = np.stack((xx, yy), axis=-1)  # shape (H, W, 2)
    grid = grid[..., None, :]  # shape (H, W, 1, 2)
    points = points[None, None, :, :]  # shape (1, 1, N, 2)

    # ----------------------------
    # PREP RENDER
    # ----------------------------
    imgs = []

    for f in range(FRAMES):
        try:
            r = np.exp(f / (FRAMES - f - 1))
        except (OverflowError, ZeroDivisionError):
            r = np.inf

        if gui is not None:
            gui.status.config(text=f'Computing frame {f+1}/{FRAMES}...')
            gui.progress['value'] = f
            gui.root.update()
        else:
            print(f'Computing frame {f+1}/{FRAMES}...', end='\r') # End with carriage return to overwrite the line when rendering is complete later

        img = render_frame(r, grid=grid, points=points, colors=colors, font_size=FONT_SIZE)

        if SAVE_FRAMES:
            img.save(FRAME_PATH / f'voronoi_{f:03}.png')
        imgs.append(img)
        print(f'Frame {f+1}/{FRAMES} complete ({r = :.3e})')

    imgs[0].save(OUTPUT,
                 save_all=True,
                 append_images=[imgs[0]] * HOLD_FRAMES + imgs[1:] +
                 [imgs[-1]] * HOLD_FRAMES,
                 duration=DURATION,
                 loop=0)

    if gui is not None:
        gui.status.config(text='Completed! See results/voronoi.gif')
        gui.progress['value'] = FRAMES
    print("Saved results/voronoi.gif")

    with Image.open('results/voronoi.gif') as img:
        img.show()

    if gui is not None:
        gui.root.mainloop()

def from_image_main(cl_args: list[str]):
    parser = argparse.ArgumentParser(
        description='Converts an image to a Voronoi diagram',
    )
    parser.add_argument('image', type=Path, help='The image to convert')
    parser.add_argument('output', type=Path, help='The output file')
    parser.add_argument('--rval', '-r', type=float, default=2.0, help='The r value (1.0: Manhattan, 2.0: Euclidean, inf: Chebyshev) (default: 2.0)')
    parser.add_argument('--resize', type=int, default=None, nargs=2, help='Resize the image to a maximum width and height (default: no resize)')
    parser.add_argument("--points", '-p', type=int, default=100, help='The number of points to use (default: 100)')
    parser.add_argument('--font-size', type=int, default=0, help='Include a label with the given font size detailing the r value. (0 = no label) (default: 0)')
    parser.add_argument('--no-warning', action='store_true', help='Disable the warning for large images. Warning: large images may take a while to render. (default: off)')

    args = parser.parse_args(cl_args)
    print(f'Loading image {args.image}...')

    src_img = Image.open(args.image).convert('RGB')
    print("Loaded image successfully!")
    if args.resize is not None:
        src_img.thumbnail(args.resize)

    width = src_img.width
    height = src_img.height

    if width * height > 256 * 512 and not args.no_warning:
        warning_response = input(f"Warning: Image is large ({width}x{height}). Are you sure you want to continue? (Y/n)")
        if warning_response.casefold() != 'y':
            print("Aborted.")
            exit(1)

    points = np.array([(randrange(width), randrange(height)) for _ in range(args.points)], dtype=float)
    # Colors is the RGB values of the points
    colors = np.array([src_img.getpixel((int(p[0]), int(p[1]))) for p in points], dtype=float) / 255.0
    points = points[None, None, :, :]  # shape (1, 1, N, 2)
    print("Generated points successfully!")

    del src_img

    print(f"Rendering the image ({width}x{height})... This may take a while.")
    out_img = render_frame(args.rval, width=width, height=height, points=points, colors=colors, font_size=args.font_size)
    print("Generated image successfully!")

    out_img.save(args.output)
    print(f'See {args.output}')

def help_main(cl_args):
    print(f"Usage: python {display_path} <mode> [args]")
    print()
    print("Modes:")
    print("  rand-gif [args] - Generate a random Voronoi diagram animation")
    print("  from-image [args] - Convert an image to a Voronoi diagram")

if __name__ == '__main__':
    FILE_PATH = Path(__file__)
    if FILE_PATH.is_relative_to(Path.cwd()):
        display_path = FILE_PATH.relative_to(Path.cwd())
    else:
        display_path = FILE_PATH.absolute()
    
    if len(argv) < 2:
        help_main(argv[1:])
        exit(2)

    mode = argv[1]
    if mode == 'rand-gif':
        rand_gif_main(argv[2:])
    elif mode == 'from-image':
        from_image_main(argv[2:])
    elif mode in ["help", '-h', '--help', 'h']:
        help_main(argv[2:])
    else:
        print(f"Unknown mode: {mode}")
        exit(2)
    