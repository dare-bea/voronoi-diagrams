import tkinter as tk
from PIL import Image, ImageDraw
from random import randrange, choice
from math import dist as euclidean_dist
from itertools import product
from math import prod
from os import makedirs
from shutil import rmtree


def manhattan_dist(p, q) -> float:
    return sum(abs(a - b) for a, b in zip(p, q))


def chebyshev_dist(p, q) -> float:
    return max(abs(a - b) for a, b in zip(p, q))


def minkowski_dist(r):

    def inner(p, q) -> float:
        return sum(abs(a - b)**r for a, b in zip(p, q))**(1 / r)
    inner.__name__ = f'minkowski_dist({r})'

    return inner


WIDTH, HEIGHT = 500, 500
POINTS = 100
DISTANCE_FUNCTIONS = euclidean_dist, manhattan_dist, chebyshev_dist, minkowski_dist(
    1.5), minkowski_dist(3)
MARKERS = True

points = {}

for i in range(POINTS):
    color = f'#{randrange(16**6):06x}'
    points[color] = (randrange(WIDTH), randrange(HEIGHT))

root = tk.Tk()
canvases = {
    fn: (tk.Canvas(root, width=WIDTH,
                   height=HEIGHT), Image.new('RGB', (WIDTH, HEIGHT)))
    for fn in DISTANCE_FUNCTIONS
}
pixels = {fn: {} for fn in DISTANCE_FUNCTIONS}

for canvas, img in canvases.values():
    canvas.pack(side=tk.LEFT, padx=5, pady=5)

    for color, point in points.items():
        canvas.create_oval(point[0] - 5,
                           point[1] - 5,
                           point[0] + 5,
                           point[1] + 5,
                           fill='white',
                           outline='black')

    canvas.update()

for y in range(HEIGHT):
    for distance, (canvas, img) in canvases.items():
        for x in range(WIDTH):
            pxl = (x, y)
            color = min(points, key=lambda c: distance(pxl, points[c]))
            pixels[distance][pxl] = color
            canvas.create_rectangle(pxl, pxl, fill=color, outline='')
            img.putpixel(pxl, (int(color[1:3], 16), int(
                color[3:5], 16), int(color[5:7], 16)))
        canvas.update()

# Check if any neighboring pixels are of a different color
if MARKERS:
    for y in range(HEIGHT):
        for distance, (canvas, img) in canvases.items():
            for x in range(WIDTH):
                pxl = (x, y)
                color = pixels[distance][pxl]
                for dx, dy in (1, 0), (0, 1), (1, 1):
                    nx, ny = x + dx, y + dy
                    if pixels[distance].get((nx, ny), color) != color:
                        canvas.create_rectangle(pxl,
                                                pxl,
                                                fill='black',
                                                outline='')
                        img.putpixel(pxl, (0, 0, 0))
                        break
            canvas.update()

    for canvas, img in canvases.values():
        for color, (x, y) in points.items():
            canvas.create_oval(x - 2,
                               y - 2,
                               x + 2,
                               y + 2,
                               fill='white',
                               outline='black')
            ImageDraw.ImageDraw(img).ellipse((x - 2, y - 2, x + 2, y + 2),
                                 fill='white',
                                 outline="black")

rmtree('results/', ignore_errors=True)
makedirs("results/", exist_ok=True)

for distance, (canvas, img) in canvases.items():
    img.save(f'results/voronoi_{distance.__name__}.png')

tk.mainloop()
