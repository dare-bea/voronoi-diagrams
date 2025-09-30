from random import random, randrange
from PIL import Image, ImageDraw, ImageFont
from os import makedirs
from shutil import rmtree
import math
import numpy as np
import tkinter as tk
from tkinter import ttk

# ----------------------------
# CONFIG
# ----------------------------
WIDTH, HEIGHT = 500, 500
POINTS = 100
FPS = 30
FRAMES = 5 * FPS
DURATION = 1000 / FPS
HOLD_FRAMES = int(round(1000 / DURATION, 0))
FONT_SIZE = 12

print(f'Generating {FRAMES} frames of {WIDTH}x{HEIGHT} Voronoi with {POINTS} points...')

# ----------------------------
# GUI SETUP
# ----------------------------
root = tk.Tk()
status = ttk.Label(root, text='Generating points...')
progress = ttk.Progressbar(root, orient='horizontal', maximum=FRAMES, mode='determinate')
status.pack()
progress.pack()
root.geometry('300x50')
root.update()

# ----------------------------
# POINT GENERATION
# ----------------------------
points = np.array([(randrange(WIDTH), randrange(HEIGHT)) for _ in range(POINTS)], dtype=float)
colors = np.array([(random(), random(), random()) for _ in range(POINTS)], dtype=float)

rmtree("results", ignore_errors=True)
makedirs("results", exist_ok=True)

# ----------------------------
# PRECOMPUTED GRID
# ----------------------------
yy, xx = np.indices((HEIGHT, WIDTH))
grid = np.stack((xx, yy), axis=-1)  # shape (H, W, 2)
grid = grid[..., None, :]            # shape (H, W, 1, 2)
points = points[None, None, :, :]    # shape (1, 1, N, 2)

# ----------------------------
# PREP RENDER
# ----------------------------
imgs = []
font = ImageFont.load_default()
base_draw = ImageDraw.Draw

for f in range(FRAMES):
    try:
        r = np.exp(f / (FRAMES - f - 1))
    except (OverflowError, ZeroDivisionError):
        r = np.inf

    status.config(text=f'Computing frame {f+1}/{FRAMES}...')
    progress['value'] = f
    root.update()

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
    border_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    border_mask[:-1, :] |= nearest[:-1, :] != nearest[1:, :]
    border_mask[:, :-1] |= nearest[:, :-1] != nearest[:, 1:]
    frame_rgb[border_mask] = 0
    img = Image.fromarray(frame_rgb, "RGB")

    # Label
    if FONT_SIZE:
        nimg = Image.new('RGB', (WIDTH, HEIGHT + FONT_SIZE + 10))
        nimg.paste(img, (0, 0))
        draw = base_draw(nimg)
        draw.text((5, HEIGHT + 2), f'{r = :.3e}', fill='white', font=font)
        img = nimg

    img.save(f'results/voronoi_{f:03}.png')
    imgs.append(img)
    print(f'Frame {f+1}/{FRAMES} complete ({r = :.3e})')

imgs[0].save(
    'results/voronoi.gif',
    save_all=True,
    append_images=[imgs[0]] * HOLD_FRAMES + imgs[1:] + [imgs[-1]] * HOLD_FRAMES,
    duration=DURATION,
    loop=0
)

status.config(text='Completed! See results/voronoi.gif')
progress['value'] = FRAMES
print("Saved results/voronoi.gif")

with Image.open('results/voronoi.gif') as img:
    img.show()

root.mainloop()
