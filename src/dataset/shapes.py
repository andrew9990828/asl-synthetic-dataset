# ======================================================================
#  shapes.py
#  Synthetic Dataset Generator – Shape Rendering Module
#
#  Author: Andrew Bieber
#  Description:
#      Defines parametric abstract shape generators used to create
#      synthetic images for ML training. Each function renders a 
#      different geometric/abstract pattern with randomness built in
#      to encourage dataset variability.
#
#  This module is intentionally simple, modular, and extensible.
# ======================================================================

import random
import math
from PIL import ImageDraw


# ----------------------------------------------------------------------
# draw_cluster_lines
# ----------------------------------------------------------------------
"""
Draws a cluster of short, randomly oriented line segments centered 
around a focal point. Produces a chaotic "starburst" or "scribble" pattern.

Args:
    draw (ImageDraw): PIL drawing context.
    center (tuple[int,int]): (x, y) coordinates of the pattern center.
    scale (float): Overall size multiplier for the pattern.
    color (str): Line color (default: "black").

Behavior:
    - Creates ~25 lines.
    - Each line begins at a jittered offset near the center.
    - Each line extends in a random direction with random length.
"""
def draw_cluster_lines(draw: ImageDraw, center, scale, color="black"):
    cx, cy = center
    for _ in range(25):
        # Random line length and direction
        length = random.uniform(20, 60) * scale
        angle = random.uniform(0, 2 * math.pi)

        dx = math.cos(angle) * length
        dy = math.sin(angle) * length

        # Random starting point near the center
        x1 = cx + random.uniform(-20, 20) * scale
        y1 = cy + random.uniform(-20, 20) * scale

        x2 = x1 + dx
        y2 = y1 + dy

        draw.line((x1, y1, x2, y2),
                  fill=color,
                  width=max(1, int(4 * scale)))


# ----------------------------------------------------------------------
# draw_arc_shape
# ----------------------------------------------------------------------
"""
Draws a large partial arc to create a curved abstract pattern.

Args:
    draw (ImageDraw): PIL drawing context.
    center (tuple[int,int]): Center of the arc's bounding circle.
    scale (float): Size multiplier.
    color (str): Arc stroke color.

Behavior:
    - Radius varies randomly (40–80 px, scaled).
    - Random start & end angles create asymmetry.
"""
def draw_arc_shape(draw, center, scale, color="black"):
    cx, cy = center
    r = random.uniform(40, 80) * scale

    bbox = (cx - r, cy - r, cx + r, cy + r)
    draw.arc(bbox,
             start=random.uniform(10, 80),
             end=random.uniform(260, 350),
             fill=color,
             width=max(1, int(5 * scale)))


# ----------------------------------------------------------------------
# draw_rect_pillars
# ----------------------------------------------------------------------
"""
Draws a tall rectangular outline to simulate a "pillar" structure.

Args:
    draw (ImageDraw): PIL drawing context.
    center (tuple[int,int]): Rectangle midpoint.
    scale (float): Rectangle scaling factor.
    color (str): Outline color.

Behavior:
    - Width and height vary randomly.
    - Drawn as an outline to preserve visual contrast.
"""
def draw_rect_pillars(draw, center, scale, color="black"):
    cx, cy = center
    w = random.uniform(40, 60) * scale
    h = random.uniform(80, 120) * scale

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    draw.rectangle((x1, y1, x2, y2),
                   outline=color,
                   width=max(1, int(4 * scale)))


# ----------------------------------------------------------------------
# draw_radial_spokes
# ----------------------------------------------------------------------
"""
Draws radial spokes like a wheel or starburst.

Args:
    draw (ImageDraw): PIL drawing context.
    center (tuple[int,int]): Spoke origin.
    scale (float): Size multiplier.
    color (str): Stroke color.

Behavior:
    - Random number of spokes (6–12).
    - Each spoke length varies.
    - Small angle jitter adds variability.
"""
def draw_radial_spokes(draw, center, scale, color="black"):
    cx, cy = center
    spokes = random.randint(6, 12)
    max_len = random.uniform(40, 80) * scale

    for i in range(spokes):
        angle = 2 * math.pi * i / spokes + random.uniform(-0.2, 0.2)
        length = max_len * random.uniform(0.5, 1.0)

        x2 = cx + math.cos(angle) * length
        y2 = cy + math.sin(angle) * length

        draw.line((cx, cy, x2, y2),
                  fill=color,
                  width=max(1, int(4 * scale)))


# ----------------------------------------------------------------------
# draw_abstract_for_letter
# ----------------------------------------------------------------------
"""
Given a letter (A–Z), deterministically selects one of the four
abstract pattern generators. Ensures each class maps to a visual style.

Args:
    letter (str): Alphabet character determining pattern type.
    draw (ImageDraw): PIL drawing context.
    center (tuple[int,int]): Pattern center point.
    scale (float): Scaling factor.

Mapping:
    - (ord(letter) - ord("A")) % 4 defines the style:
        0 → cluster lines
        1 → arc shape
        2 → rectangle pillars
        3 → radial spokes
"""
def draw_abstract_for_letter(letter, draw, center, scale):
    idx = ord(letter) - ord("A")
    style = idx % 4

    if style == 0:
        draw_cluster_lines(draw, center, scale)
    elif style == 1:
        draw_arc_shape(draw, center, scale)
    elif style == 2:
        draw_rect_pillars(draw, center, scale)
    else:
        draw_radial_spokes(draw, center, scale)
