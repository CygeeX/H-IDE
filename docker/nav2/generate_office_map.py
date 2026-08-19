#!/usr/bin/env python3
"""Generate the Nav2 occupancy map matching the H-IDE MuJoCo office scene."""

from __future__ import annotations

from math import ceil, floor
from pathlib import Path


RESOLUTION = 0.05
ORIGIN_X = -6.0
ORIGIN_Y = -4.0
WIDTH = 240
HEIGHT = 160
FREE = 254
OCCUPIED = 0


def _column(x: float) -> int:
    return max(0, min(WIDTH - 1, floor((x - ORIGIN_X) / RESOLUTION)))


def _row(y: float) -> int:
    # PGM rows run top-to-bottom while map coordinates run bottom-to-top.
    return max(0, min(HEIGHT - 1, floor((ORIGIN_Y + HEIGHT * RESOLUTION - y) / RESOLUTION)))


def _box(grid: list[list[int]], cx: float, cy: float, sx: float, sy: float) -> None:
    left, right = _column(cx - sx / 2), _column(cx + sx / 2)
    top, bottom = _row(cy + sy / 2), _row(cy - sy / 2)
    for row in range(min(top, bottom), max(top, bottom) + 1):
        for column in range(min(left, right), max(left, right) + 1):
            grid[row][column] = OCCUPIED


def _circle(grid: list[list[int]], cx: float, cy: float, radius: float) -> None:
    cells = ceil(radius / RESOLUTION)
    center_column, center_row = _column(cx), _row(cy)
    for row in range(max(0, center_row - cells), min(HEIGHT, center_row + cells + 1)):
        for column in range(max(0, center_column - cells), min(WIDTH, center_column + cells + 1)):
            x = ORIGIN_X + (column + 0.5) * RESOLUTION
            y = ORIGIN_Y + (HEIGHT - row - 0.5) * RESOLUTION
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                grid[row][column] = OCCUPIED


def main() -> None:
    grid = [[FREE for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for spec in (
        (0.0, 3.95, 12.0, 0.10),
        (0.0, -3.95, 12.0, 0.10),
        (-5.95, 0.0, 0.10, 8.0),
        (5.95, 0.0, 0.10, 8.0),
        (-2.9, 1.15, 1.4, 0.10),
        (0.0, 1.15, 1.6, 0.10),
        (2.9, 1.15, 1.4, 0.10),
        (3.8, -0.65, 1.6, 0.9),
        (4.45, 2.2, 1.4, 0.55),
        (4.9, -2.1, 1.2, 0.7),
    ):
        _box(grid, *spec)
    _circle(grid, 2.0, 0.8, 0.38)

    destination = Path(__file__).with_name("office_floor.pgm")
    lines = ["P2", "# H-IDE MuJoCo office floor: 12m x 8m, 5cm cells", f"{WIDTH} {HEIGHT}", "255"]
    lines.extend(" ".join(str(value) for value in row) for row in grid)
    destination.write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
