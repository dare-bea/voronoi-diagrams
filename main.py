from gif import render_frame
from sys import argv
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
from random import randrange

def main(cl_args: list[str]):
    parser = argparse.ArgumentParser(
        description='Converts an image to a Voronoi diagram',
    )
    parser.add_argument('image', type=Path, help='The image to convert')
    parser.add_argument('output', type=Path, help='The output file')
    parser.add_argument('--rval', type=float, default=2.0, help='The r value (1.0: Manhattan, 2.0: Euclidean, inf: Chebyshev) (default: 2.0)')
    parser.add_argument("--points", type=int, default=100, help='The number of points to use (default: 100)')
    parser.add_argument('--font-size', type=int, default=0, help='Include a label with the given font size detailing the r value. (0 = no label) (default: 0)')
    
    args = parser.parse_args(cl_args)
    print(f'Loading image {args.image}...')

    src_img = Image.open(args.image).convert('RGB')
    print("Loaded image successfully!")

    width = src_img.width
    height = src_img.height

    points = np.array([(randrange(width), randrange(height)) for _ in range(args.points)], dtype=float)
    # Colors is the RGB values of the points
    colors = np.array([src_img.getpixel((int(p[0]), int(p[1]))) for p in points], dtype=float) / 255.0
    points = points[None, None, :, :]  # shape (1, 1, N, 2)
    print("Generated points successfully!")

    del src_img

    out_img = render_frame(args.rval, width=width, height=height, points=points, colors=colors, font_size=args.font_size)
    print("Generated image successfully!")
    
    out_img.save(args.output)
    print(f'See {args.output}')
    
if __name__ == '__main__':
    main(argv[1:])