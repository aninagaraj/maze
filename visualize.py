"""Matplotlib visualisation: draw mazes, paths, recursion graphs."""

import os

import matplotlib.pyplot as plt

# -- colour palette --------------------------------------------------------
WALL_COLOR = '#e67e22'          # rich autumn orange
PATH_COLOR = '#2980b9'          # deep steel blue
ALL_COLOR  = '#8e44ad'          # purple — pops against orange walls
MARKER_COL = '#e74c3c'          # rich crimson
DPI = 300

# -- common draw helpers ---------------------------------------------------

def _draw_walls(ax, maze):
    """Draw wall segments with rounded ends and smooth joins."""
    for (xf, yf), (xt, yt), v in ((*k, v) for k, v in maze.items()):
        if v > 0:
            ax.plot([xf, xt], [yf, yt],
                    lw=8, color=WALL_COLOR, zorder=1,
                    solid_capstyle='round', solid_joinstyle='round')


def _draw_start_end(ax, x1, y1, x2, y2):
    """Mark start and end with crimson stars + soft halo."""
    # halo
    ax.scatter([x1, x2], [y1, y2], s=1800,
               color=MARKER_COL, alpha=0.12, edgecolors='none', zorder=4)
    # star
    ax.scatter([x1, x2], [y1, y2], marker='*', s=750,
               color=MARKER_COL, edgecolors='none', zorder=5)


def _draw_path_glow(ax, px, py, color, zorder=3):
    """Draw a translucent 'glow' copy behind the main line."""
    ax.plot(px, py, lw=9, color=color, alpha=0.12,
            solid_capstyle='round', solid_joinstyle='round', zorder=zorder)


def _setup_ax(ax):
    ax.set_aspect('equal')
    ax.axis('off')


_SAVE_KW = dict(dpi=DPI, bbox_inches='tight', pad_inches=0.1)


# -- public functions ------------------------------------------------------

def save_maze(maze, start, end, out_dir):
    """Save the bare maze to ``<out_dir>/maze.png``."""
    fig, ax = plt.subplots(figsize=(45, 45))
    _setup_ax(ax)
    _draw_start_end(ax, start[0], start[1], end[0], end[1])
    _draw_walls(ax, maze)
    fig.savefig(os.path.join(out_dir, 'maze.png'), **_SAVE_KW)
    plt.close(fig)


def save_shortest_path(maze, sp, start, end, out_dir):
    """Save the maze with the shortest path overlaid in blue."""
    fig, ax = plt.subplots(figsize=(45, 45))
    _setup_ax(ax)
    _draw_start_end(ax, start[0], start[1], end[0], end[1])
    _draw_walls(ax, maze)
    if sp and len(sp) > 1:
        spx, spy = zip(*sp)
        _draw_path_glow(ax, spx, spy, PATH_COLOR)
        ax.plot(spx, spy, lw=4, color=PATH_COLOR,
                solid_capstyle='round', solid_joinstyle='round', zorder=4)
    fig.savefig(os.path.join(out_dir, 'shortestpath.png'), **_SAVE_KW)
    plt.close(fig)


def save_recursion_graph(rb_stream, indices, out_dir):
    """Save the recursion‑depth step‑plot to ``recursion.png``."""
    rb = np.array(rb_stream).cumsum()

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('#fafafa')

    ax.fill_between(range(len(rb)), rb, alpha=0.10, color=PATH_COLOR)
    ax.plot(rb, drawstyle='steps-post', lw=1.5, color=PATH_COLOR)

    if indices:
        ax.scatter(indices, [rb[k] for k in indices],
                   marker='*', color=MARKER_COL, s=120,
                   edgecolors='none', zorder=3)

    ax.set_title('Recursion Depth', fontsize=18, fontweight='bold',
                 color='#333333', pad=12)
    ax.set_xlabel('Steps →', fontsize=13, color='#555555')
    ax.set_ylabel('Depth',   fontsize=13, color='#555555')
    ax.grid(axis='both', ls='solid', alpha=0.25, lw=0.5)
    ax.set_facecolor('#fafafa')
    ax.tick_params(labelsize=10, colors='#666666')

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(os.path.join(out_dir, 'recursion.png'), **_SAVE_KW)
    plt.close(fig)


def save_all_paths(maze, paths, start, end, out_dir):
    """Draw every discovered path in semi‑transparent purple."""
    fig, ax = plt.subplots(figsize=(45, 45))
    _setup_ax(ax)
    _draw_start_end(ax, start[0], start[1], end[0], end[1])
    _draw_walls(ax, maze)
    for p in paths:
        if len(p) < 2:
            continue
        spx, spy = zip(*p)
        ax.plot(spx, spy, lw=6, color=ALL_COLOR, alpha=0.08,
                solid_capstyle='round', solid_joinstyle='round', zorder=2)
        ax.plot(spx, spy, lw=2, color=ALL_COLOR, alpha=0.35,
                solid_capstyle='round', solid_joinstyle='round', zorder=3)
    fig.savefig(os.path.join(out_dir, 'all_paths.png'), **_SAVE_KW)
    plt.close(fig)


# local numpy import (used after matplotlib loads it)
import numpy as np
