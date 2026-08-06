"""Maze generation: random wall‑based and recursive‑backtracker (digger)."""

import random
import time
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def heuristic(p1, p2):
    """Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def neighbors(p, m, n, kind):
    """Valid neighbouring grid points.

    * *kind = 'w'*  – wall‑junction points  (1‑based, inclusive boundary)
    * *kind = 'r'*  – cell‑centre points     (1.5‑based, half‑step border)
    """
    nb = []
    N = (p[0], p[1] + 1)
    E = (p[0] + 1, p[1])
    S = (p[0], p[1] - 1)
    W = (p[0] - 1, p[1])

    if kind == 'w':
        if N[1] <= n:   nb.append(N)
        if E[0] <= m:   nb.append(E)
        if S[1] >= 1:   nb.append(S)
        if W[0] >= 1:   nb.append(W)
    elif kind == 'r':
        if N[1] <= n - 0.5: nb.append(N)
        if E[0] <= m - 0.5: nb.append(E)
        if S[1] >= 1.5:     nb.append(S)
        if W[0] >= 1.5:     nb.append(W)
    return nb


def unconnected(pp, m, n, maze):
    """Number of open edges adjacent to *pp* – used to find isolated cells."""
    nbs = neighbors(pp, m, n, 'w')
    return sum(
        1 for to_ in nbs
        if maze.get((pp, to_), 0) == 0 and maze.get((to_, pp), 0) == 0
    )


# ---------------------------------------------------------------------------
# random wall‑based maze
# ---------------------------------------------------------------------------

def create_maze(m, n, prob):
    """Generate a random maze by placing walls probabilistically.

    Returns ``(maze, wall_points)`` where *maze* is a ``dict`` mapping
    ``(from, to) -> 1|0``.
    """
    Xw, Yw = np.arange(1, m + 1), np.arange(1, n + 1)
    xw, yw = np.meshgrid(Xw, Yw)
    points_w = [(a, b) for a, b in zip(np.ravel(xw), np.ravel(yw))]
    maze = {}

    for from_ in points_w:
        nbs = neighbors(from_, m, n, 'w')
        for to_ in nbs.copy():
            if to_ < from_ or to_ == from_:
                nbs.remove(to_)
        for to_ in nbs:
            maze[(from_, to_)] = 1 if np.random.rand() <= prob else 0
            # outer‑boundary walls are always kept
            if (from_[1] == 1 and to_[1] == 1) or (from_[1] == n and to_[1] == n):
                maze[(from_, to_)] = 1
            if (from_[0] == 1 and to_[0] == 1) or (from_[0] == m and to_[0] == m):
                maze[(from_, to_)] = 1

    # ensure no cell is completely boxed in
    for pp in points_w:
        if unconnected(pp, m, n, maze) == 4:
            rn = np.random.randint(0, 4)
            dx = np.round(np.cos(rn * np.pi / 2))
            dy = np.round(np.sin(rn * np.pi / 2))
            maze[(pp, (pp[0] + dx, pp[1] + dy))] = 1

    return maze, points_w


# ---------------------------------------------------------------------------
# graph from wall map
# ---------------------------------------------------------------------------

def is_wall(p1, p2, maze):
    """Return *True* if a wall separates cell‑centres *p1* and *p2*."""
    if p1[0] != p2[0]:
        # vertical wall
        xmid = (p1[0] + p2[0]) / 2
        yl, yh = p1[1] - 0.5, p1[1] + 0.5
        if maze.get(((xmid, yl), (xmid, yh)), -1) == 1 or \
           maze.get(((xmid, yh), (xmid, yl)), -1) == 1:
            return True
    else:
        # horizontal wall
        ymid = (p1[1] + p2[1]) / 2
        xl, xh = p1[0] - 0.5, p1[0] + 0.5
        if maze.get(((xl, ymid), (xh, ymid)), -1) == 1 or \
           maze.get(((xh, ymid), (xl, ymid)), -1) == 1:
            return True
    return False


def create_routes(maze, m, n):
    """Build an adjacency‑list graph from navigable cell‑centre pairs.

    Returns ``(graph, cell_points)``.
    """
    Xc, Yc = np.arange(1.5, m), np.arange(1.5, n)
    xc, yc = np.meshgrid(Xc, Yc)
    points_c = [(a, b) for a, b in zip(np.ravel(xc), np.ravel(yc))]

    routes = {}
    for from_ in points_c:
        for to_ in neighbors(from_, m, n, 'r'):
            if not is_wall(from_, to_, maze):
                routes[(from_, to_)] = 1
                routes[(to_, from_)] = 1

    graph = defaultdict(list)
    for k, _ in routes.items():
        graph[k[0]].append((1, k[1]))

    return graph, points_c


# ---------------------------------------------------------------------------
# recursive‑backtracker (digger)
# ---------------------------------------------------------------------------

def maze_digger(m, n):
    """Generate a *perfect* maze via recursive backtracking.

    Returns ``(maze, routes, graph, branches)``.

    .. note::

       The finish coordinate is offset by +1 compared to the random‑wall
       generator because the digger encodes walls with a shift of ``(+2,+2)``
       from cell indices (see the wall‑construction loop below).
    """
    branches, stack, visited, routes, maze = [], [], [], {}, {}
    grid = np.empty((m, n), dtype=object)

    class Cell:
        def __init__(self, x, y):
            self.x, self.y = x, y
            self.r = self.t = self.l = self.b = True
            self.visited = False

        def get_neighbors(self):
            nb = []
            if self.y < n - 1 and not grid[self.x, self.y + 1].visited:
                nb.append((self.x, self.y + 1))
            if 1 <= self.x and not grid[self.x - 1, self.y].visited:
                nb.append((self.x - 1, self.y))
            if 1 <= self.y and not grid[self.x, self.y - 1].visited:
                nb.append((self.x, self.y - 1))
            if self.x < m - 1 and not grid[self.x + 1, self.y].visited:
                nb.append((self.x + 1, self.y))
            return nb

        def visit(self, other):
            stack.append((other.x, other.y))
            visited.append((other.x, other.y))
            routes[((self.y + 1.5, self.x + 1.5), (other.y + 1.5, other.x + 1.5))] = 1
            routes[((other.y + 1.5, other.x + 1.5), (self.y + 1.5, self.x + 1.5))] = 1

            if other.y == self.y + 1:     other.l, self.r = False, False
            if other.x == self.x + 1:     other.t, self.b = False, False
            if other.y + 1 == self.y:     other.r, self.l = False, False
            if other.x == self.x - 1:     other.b, self.t = False, False

            other.visited = True
            grid[other.x, other.y] = other

    print('Generating maze ...')
    st = time.time()

    for i in range(m):
        for j in range(n):
            grid[i, j] = Cell(i, j)

    grid[0, 0].visited = True
    current = grid[0, 0]
    stack.append((current.x, current.y))
    branches.append(stack.copy())
    visited.append((current.x, current.y))

    while len(visited) < m * n:
        nbs = current.get_neighbors()
        if nbs:
            i, j = random.choice(nbs)
            current.visit(grid[i, j])
            branches.append(stack.copy())
            current = grid[i, j]
        else:
            stack.pop()
            i, j = stack[-1]
            branches.append(stack.copy())
            current = grid[i, j]

    # build wall dictionary from cell‑wall booleans
    for i in range(m):
        for j in range(n):
            cell = grid[i, j]
            if cell.r:
                maze[(cell.y + 2, cell.x + 1), (cell.y + 2, cell.x + 2)] = 1
            if cell.t:
                maze[(cell.y + 1, cell.x + 1), (cell.y + 2, cell.x + 1)] = 1
            if cell.l:
                maze[(cell.y + 1, cell.x + 1), (cell.y + 1, cell.x + 2)] = 1
            if cell.b:
                maze[(cell.y + 1, cell.x + 2), (cell.y + 2, cell.x + 2)] = 1

    graph = defaultdict(list)
    for k, _ in routes.items():
        graph[k[0]].append((1, k[1]))

    print(f'Done in {time.time() - st:3.0f} seconds')
    return maze, routes, graph, branches
