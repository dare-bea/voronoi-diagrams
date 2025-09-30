from random import random, randrange
from PIL import Image, ImageTk
from PIL.ImageDraw import ImageDraw
from PIL import ImageFont
from os import makedirs
from shutil import rmtree
import math
import numpy as np
from numpy import float64 as npfloat

import tkinter as tk
from tkinter import ttk

def manhattan_dist(p, q) -> float:
    return np.sum([np.abs(a - b) for a, b in zip(p, q)])

def chebyshev_dist(p, q) -> float:
    return np.max([np.abs(a - b) for a, b in zip(p, q)])

def minkowski_dist(p, q, r) -> float:
    if r is None or r == math.inf:
        return chebyshev_dist(p, q)
    if r == 1:
        return manhattan_dist(p, q)
    result = np.power(np.sum([np.power(abs(a - b), r) for a, b in zip(p, q)]), (1 / r))
    if not np.isfinite(result):
        return chebyshev_dist(p, q)
    return result


WIDTH, HEIGHT = 50, 50
POINTS = 100

FPS = 30
FRAMES = 5 * FPS  # 5 seconds
DURATION = 1000 / FPS  # ms
HOLD_FRAMES = int(round(1000 / DURATION, 0))
FONT_SIZE = 12

print(
    f'Generating {FRAMES} frames of {WIDTH}x{HEIGHT} voronoi diagram with {POINTS} points...'
)
print(f'Duration: {DURATION:.3f} ms per frame')
print(f'Hold frames: {HOLD_FRAMES} ({DURATION * HOLD_FRAMES:.3f} ms)')
print(f'Total duration: {DURATION * (FRAMES + 2 * HOLD_FRAMES):.3f} ms')
print()

points: dict[tuple[float, float, float], tuple[int, int]] = {}
imgs: list[Image.Image] = []

root = tk.Tk()
status = ttk.Label(root, text='Generating points...')
progress = ttk.Progressbar(root, orient='horizontal', maximum=FRAMES, mode='determinate')
status.pack()
progress.pack()
root.geometry('300x50')

root.update()

for i in range(POINTS):
    color = (random(), random(), random())
    points[color] = (randrange(WIDTH), randrange(HEIGHT))

rmtree('results/', ignore_errors=True)
makedirs("results/", exist_ok=True)

for f in range(FRAMES):
    img = Image.new('RGB', (WIDTH, HEIGHT))
    try:
        r = np.exp(npfloat(f)/npfloat(FRAMES - f - 1))
    except (OverflowError, ZeroDivisionError):
        r = None
    for y in range(HEIGHT):
        status.config(text=f'Drawing areas in frame {f+1}/{FRAMES}... ({y/HEIGHT:.2%})')
        progress['value'] = f + y / HEIGHT / 2
        root.update()
        for x in range(WIDTH):
            pxl = (x, y)
            color = min(points,
                        key=lambda c: minkowski_dist(pxl, points[c], r))
            img.putpixel(pxl, (int(color[0] * 255), int(
                color[1] * 255), int(color[2] * 255)))

    # Draw black borders between different colors
    borders = Image.new('RGBA', (WIDTH, HEIGHT))
    for y in range(HEIGHT):
        status.config(text=f'Drawing borders in frame {f+1}/{FRAMES}... ({y/HEIGHT:.2%})')
        progress['value'] = f + y / HEIGHT / 2 + 0.5
        root.update()
        for x in range(WIDTH):
            pxl = (x, y)
            color = img.getpixel(pxl)
            matches = []
            for dx, dy in (1, 0), (0, 1):
                nx, ny = x + dx, y + dy
                if nx < WIDTH and ny < HEIGHT and img.getpixel((nx, ny)) != color:
                    borders.putpixel(pxl, (0, 0, 0))
                    break
    img = Image.alpha_composite(img.convert('RGBA'), borders).convert("RGB")
    
    if FONT_SIZE:
        nimg = Image.new('RGB', (WIDTH, HEIGHT + FONT_SIZE + 10))
        nimg.paste(img, (0, 0))
        img = nimg

        ImageDraw(img).text(
            (5, HEIGHT + 5),
            f'{r = :.3e}',
            fill='white',
            font=ImageFont.load_default(size=FONT_SIZE)
        )
    img.save(f'results/voronoi_{f:03}.png')
    imgs.append(img)
    print(f'Frame {f+1}/{FRAMES} complete ({r = :.3e})')

imgs[0].save('results/voronoi.gif',
             save_all=True,
             append_images=[imgs[0]] * HOLD_FRAMES + imgs[1:] +
             [imgs[-1]] * HOLD_FRAMES,
             duration=DURATION,
             loop=0)
with Image.open('results/voronoi.gif') as img:
    img.show()

status.config(text='Completed! See results/voronoi.gif')
progress['value'] = FRAMES
root.mainloop()