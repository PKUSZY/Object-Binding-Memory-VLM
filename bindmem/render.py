"""Pillow 2D and CPU triangle-rasterized textured 3D extrusions.

3D is genuine orthographic mesh projection with a z buffer. The front-face
texture lives in object coordinates. Shapes refer to the extrusion silhouette,
not to naturalistic semantic categories such as balls or cones.
"""
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import COLORS, digest

CELL = 224
BG = (247, 248, 250)


def vertices(shape):
    n = {'circle': 64, 'triangle': 3, 'square': 4, 'star': 10,
         'pentagon': 5, 'hexagon': 6}[shape]
    phase = math.pi / 4 if shape == 'square' else -math.pi / 2
    return [(math.cos(phase + 2*math.pi*i/n) * (.44 if shape == 'star' and i % 2 else 1),
             math.sin(phase + 2*math.pi*i/n) * (.44 if shape == 'star' and i % 2 else 1)) for i in range(n)]


def texture_rgb(obj, u, v):
    base = np.asarray(COLORS[obj[0]], dtype=float)
    if obj[1] == 'solid':
        ink = np.zeros(np.broadcast(u, v).shape, dtype=bool)
    elif obj[1] == 'stripes':
        ink = np.mod(v + 2, .27) < .065
    elif obj[1] == 'grid':
        ink = (np.mod(u + 2, .29) < .045) | (np.mod(v + 2, .29) < .045)
    else:
        ink = (np.mod(u + 2, .29) - .145)**2 + (np.mod(v + 2, .29) - .145)**2 < .055**2
    return np.where(ink[..., None], base * .38, base)


def render2d(obj, variant=0):
    n = CELL * 2
    yy, xx = np.mgrid[:n, :n]
    rng = random.Random(variant)
    dx, dy = rng.uniform(-8, 8), rng.uniform(-8, 8)
    scale = rng.uniform(75, 81) * 2
    u, v = (xx-n/2-dx)/scale, (yy-n/2-dy)/scale
    rgb = Image.fromarray(texture_rgb(obj, u, v).astype('uint8'))
    mask = Image.new('L', (n, n))
    poly = [(n/2+dx+x*scale, n/2+dy+y*scale) for x, y in vertices(obj[2])]
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    canvas = Image.new('RGB', (n, n), BG)
    canvas.paste(rgb, (0, 0), mask)
    ImageDraw.Draw(canvas).line(poly+[poly[0]], fill=(40, 45, 48), width=2)
    return canvas.resize((CELL, CELL), Image.Resampling.LANCZOS)


def render3d_scene(objects, centers=None, yaw=25, occluded=False, variant=0):
    """centers are world (x,y,z); larger z is closer to camera after rotation."""
    n = CELL * 2
    canvas = np.full((n, n, 3), BG, dtype=np.uint8)
    depth = np.full((n, n), -np.inf)
    angle = math.radians(yaw)
    rot = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                    [-math.sin(angle), 0, math.cos(angle)]])
    centers = centers or [(0, 0, 0)] * len(objects)
    scale = (65 if len(objects) == 1 else 52) * 2 * random.Random(variant).uniform(.97, 1.03)

    def triangle(points, uv, obj, shade):
        screen = points[:, :2]*scale + n/2
        lo = np.maximum(np.floor(screen.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(screen.max(0)).astype(int)+1, n)
        if np.any(hi <= lo):
            return
        yy, xx = np.mgrid[lo[1]:hi[1], lo[0]:hi[0]]
        a, b, c = screen
        den = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-8:
            return
        w0 = ((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1])) / den
        w1 = ((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1])) / den
        w2 = 1-w0-w1
        weights = np.stack([w0, w1, w2], -1)
        z = weights @ points[:, 2]
        target = depth[lo[1]:hi[1], lo[0]:hi[0]]
        keep = (weights.min(-1) >= -1e-6) & (z > target)
        coords = weights @ uv
        rgb = np.clip(texture_rgb(obj, coords[..., 0], coords[..., 1]) * shade, 0, 255).astype('uint8')
        canvas[lo[1]:hi[1], lo[0]:hi[0]][keep] = rgb[keep]
        target[keep] = z[keep]

    for obj, center in zip(objects, centers):
        poly = np.array(vertices(obj[2]))
        m = len(poly)
        front = np.column_stack([poly, np.full(m, .22)])
        back = np.column_stack([poly, np.full(m, -.22)])
        for zface, face, shade in [(.22, front, 1.0), (-.22, back, .72)]:
            for i in range(m):
                j = (i+1) % m
                pts = np.array([[0, 0, zface], face[i], face[j]])
                # Rotate object, then translate in camera/world coordinates.
                triangle(pts @ rot.T + center, np.array([[0, 0], poly[i], poly[j]]), obj, shade)
        for i in range(m):
            j = (i+1) % m
            for pts in [[front[i], back[i], back[j]], [front[i], back[j], front[j]]]:
                pts = np.array(pts)
                triangle(pts @ rot.T + center, pts[:, :2], obj, .62)
    image = Image.fromarray(canvas)
    if occluded:
        # Neutral panel covers the rightmost ~20% of the object's projected width.
        ImageDraw.Draw(image).rectangle((n/2+scale*.52, n*.25, n*.9, n*.82), fill=(183, 187, 193))
    return image.resize((CELL, CELL), Image.Resampling.LANCZOS)


class Renderer:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / 'images').mkdir(parents=True, exist_ok=True)
        self.specs = {}

    def image(self, objects, mode='2d', variant=0, labels=False, relation=None):
        spec = dict(objects=objects, mode=mode, variant=variant, labels=labels, relation=relation)
        rel = f'images/{digest(spec)[:32]}.png'
        self.specs[rel] = spec
        path = self.root / rel
        if path.exists():
            return rel
        if relation == 'front':
            # Both full identities remain mostly visible; z buffer determines overlap.
            ordered = sorted(tuple(o) for o in objects)
            centers = [((-.48 if tuple(o) == ordered[0] else .48),
                        (-.15 if tuple(o) == ordered[0] else .15), .8 if i == 0 else -.8)
                       for i, o in enumerate(objects)]
            # Identity-specific screen positions stay fixed across worlds; only depth swaps.
            im = render3d_scene(objects, centers, yaw=15, variant=variant)
        else:
            cols = {1: 1, 2: 2, 4: 2, 8: 4}.get(len(objects), math.ceil(math.sqrt(len(objects))))
            rows = math.ceil(len(objects)/cols)
            if relation == 'above':
                cols, rows = 1, len(objects)
            im = Image.new('RGB', (cols*CELL, rows*CELL), BG)
            for i, obj in enumerate(objects):
                # Keep each object's pixels constant when regrouping images or candidates.
                object_variant = int(digest([obj, variant])[:8], 16)
                if mode == '2d':
                    tile = render2d(obj, object_variant)
                else:
                    yaw = -30 if mode == '3d_cross_query' else 25
                    tile = render3d_scene([obj], yaw=yaw, occluded=mode == '3d_occluded_query', variant=object_variant)
                if labels:
                    ImageDraw.Draw(tile).text((10, 8), chr(65+i), fill=(0, 0, 0), font_size=20)
                im.paste(tile, ((i % cols)*CELL, (i // cols)*CELL))
        im.save(path)
        return rel
