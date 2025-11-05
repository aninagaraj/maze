#!/usr/bin/env python3
"""
Maze Generator and Path Finder
------------------------------
This script generates a maze (either random or recursive backtracking),
finds the shortest path using Dijkstra’s algorithm, and optionally
recursively enumerates all possible paths.

It can visualize the maze, shortest path, recursion statistics,
and all paths using matplotlib.

Usage:
    python maze_solver.py --size 25,35 --use-random --prob 0.35
"""

import argparse
import random
import time
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
import sys


# === Utility Functions ===

def heuristic(p1, p2):
    """Euclidean distance heuristic between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def neighbors(p, m, n, class_):
    """
    Return valid neighboring cells for a given point.

    Args:
        p: Current point (x, y)
        m, n: Maze dimensions
        class_: 'w' for wall points, 'r' for route (cell center) points
    """
    nb = []
    N, E, S, W = (p[0], p[1] + 1), (p[0] + 1, p[1]), (p[0], p[1] - 1), (p[0] - 1, p[1])

    if class_ == 'w':
        if N[1] <= n: 
            nb.append(N)
        if E[0] <= m: 
            nb.append(E)
        if S[1] >= 1: 
            nb.append(S)
        if W[0] >= 1: 
            nb.append(W)
    elif class_ == 'r':
        if N[1] <= (n - .5): 
            nb.append(N)
        if E[0] <= (m - .5): 
            nb.append(E)
        if S[1] >= 1.5: 
            nb.append(S)
        if W[0] >= 1.5: 
            nb.append(W)
    return nb


def unconnected(pp, m, n, maze):
    """Return count of unconnected neighboring walls for a given point."""
    nbs = neighbors(pp, m, n, 'w')
    cond = 0
    for to_ in nbs:
        if (maze.get((pp, to_), 0) == 0) and (maze.get((to_, pp), 0) == 0):
            cond += 1
    return cond


# === Maze Generation ===

def create_maze(m, n, prob):
    """
    Generate a random maze by probabilistically placing walls.

    Args:
        m, n: Maze dimensions
        prob: Probability of a wall between adjacent cells
    """
    Xw, Yw = np.arange(1, m + 1), np.arange(1, n + 1)
    xw, yw = np.meshgrid(Xw, Yw)
    points_w = [(a, b) for a, b in zip(np.ravel(xw), np.ravel(yw))]
    maze = {}

    # Create random wall connections
    for from_ in points_w:
        nbs = neighbors(from_, m, n, 'w')
        for to_ in nbs.copy():
            if to_ < from_ or to_ == from_:
                nbs.remove(to_)
        for to_ in nbs:
            maze[(from_, to_)] = 1 if np.random.rand() <= prob else 0
            # Keep outer boundaries always walled
            if (from_[1] == 1 and to_[1] == 1) or (from_[1] == n and to_[1] == n):
                maze[(from_, to_)] = 1
            if (from_[0] == 1 and to_[0] == 1) or (from_[0] == m and to_[0] == m):
                maze[(from_, to_)] = 1

    # Ensure no completely isolated cells
    for pp in points_w:
        if unconnected(pp, m, n, maze) == 4:
            rn = np.random.randint(0, 4)
            dx = np.round(np.cos(rn * np.pi / 2))
            dy = np.round(np.sin(rn * np.pi / 2))
            maze[(pp, (pp[0] + dx, pp[1] + dy))] = 1

    return (maze, points_w)


def is_wall(p1, p2, maze):
    """
    Determine if a wall exists between two adjacent cells.
    Used to construct navigable routes for pathfinding.
    """
    if p1[0] != p2[0]:
        # Vertical wall
        xmid = (p1[0] + p2[0]) / 2
        yl, yh = p1[1] - 0.5, p1[1] + 0.5
        if maze.get(((xmid, yl), (xmid, yh)), -1) == 1 or maze.get(((xmid, yh), (xmid, yl)), -1) == 1:
            return True
    else:
        # Horizontal wall
        ymid = (p1[1] + p2[1]) / 2
        xl, xh = p1[0] - 0.5, p1[0] + 0.5
        if maze.get(((xl, ymid), (xh, ymid)), -1) == 1 or maze.get(((xh, ymid), (xl, ymid)), -1) == 1:
            return True
    return False


def create_routes(maze):
    """
    Construct a graph of navigable routes based on maze openings.
    Each route connects adjacent cell centers not separated by walls.
    """
    routes = {}
    Xc, Yc = np.arange(1.5, m), np.arange(1.5, n)
    xc, yc = np.meshgrid(Xc, Yc)
    points_c = [(a, b) for a, b in zip(np.ravel(xc), np.ravel(yc))]

    for from_ in points_c:
        for to_ in neighbors(from_, m, n, 'r'):
            if not is_wall(from_, to_, maze):
                routes[(from_, to_)] = 1
                routes[(to_, from_)] = 1

    graph = defaultdict(list)
    for k, _ in routes.items():
        graph[k[0]].append((1, k[1]))

    return (graph, points_c)


# === Pathfinding (Dijkstra + Heuristic) ===

def Djikstra_heapq(source, destination, graph):
    """
    Find the shortest path between source and destination using Dijkstra’s algorithm
    with a Euclidean heuristic (similar to A* for continuous spaces).
    """
    h = []
    vmap = {}
    heapq.heappush(h, ((0 + heuristic(source, destination), 0, source, source)))

    while h:
        _, currcost, currvtx, prevvtx = heapq.heappop(h)
        vmap[currvtx] = prevvtx
        if currvtx == destination:
            break
        for neighcost, neigh in graph[currvtx]:
            if neigh not in vmap:
                # Avoid revisiting nodes already optimized
                vertex_in_heap = any(
                    currcost + neighcost >= d and neigh == p1 for _, d, p1, _ in h
                )
                if not vertex_in_heap:
                    heapq.heappush(
                        h,
                        (
                            currcost + neighcost + heuristic(neigh, destination),
                            currcost + neighcost,
                            neigh,
                            currvtx,
                        ),
                    )

    # Reconstruct path
    if destination not in vmap:
        return ([], list(vmap.keys()))

    ps = [destination]
    y = destination
    while y != source:
        ps.append(vmap[y])
        y = vmap[y]
    ps.reverse()
    return (ps, list(vmap.keys()))


# === Recursive Path Search ===

def find_paths(currvtx, destination, graph):
    """
    Recursively explore all possible paths from source to destination.
    Stores all paths and recursion depth stats globally.
    """
    global paths, recur, btrack, counter
    recur += 1
    if recur > 0:
        rbstream.append(1)

    if currvtx == destination:
        counter += 1
        print(f'\tPaths found: {counter}', end='\r')
        paths.append(path.copy())
        path_lengths.append(len(path))
        indices.append(len(rbstream) - 1)
        return

    for _, nextvtx in sorted(graph[currvtx], reverse=True):
        if nextvtx not in path:
            path.append(nextvtx)
            courses.append(path.copy())
            prevvtx = currvtx
            find_paths(nextvtx, destination, graph)
            if (alls != 1 and counter > 0):
                return
            btrack += 1
            rbstream.append(-1)
            currvtx = prevvtx
            path.pop()
            courses.append(path.copy())


# === Maze Digger (Recursive Backtracker Algorithm) ===

def maze_digger(m, n):
    """
    Generate a perfect maze using recursive backtracking.

    Each cell knows its walls and visitation state.
    The algorithm visits all cells and removes walls along the path.
    """
    class Cell:
        def __init__(self, x, y):
            self.x, self.y = x, y
            self.r = self.t = self.l = self.b = True
            self.visited = False

        def get_neightbors(self):
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
            """Mark a neighboring cell as visited and remove walls between."""
            stack.append((other.x, other.y))
            visited.append((other.x, other.y))
            routes[((self.y + 1.5, self.x + 1.5), (other.y + 1.5, other.x + 1.5))] = 1
            routes[((other.y + 1.5, other.x + 1.5), (self.y + 1.5, self.x + 1.5))] = 1

            # Remove walls between current and neighbor
            if other.y == self.y + 1: 
                other.l, self.r = False, False
            if other.x == self.x + 1: 
                other.t, self.b = False, False
            if other.y + 1 == self.y: 
                other.r, self.l = False, False
            if other.x == self.x - 1: 
                other.b, self.t = False, False

            other.visited = True
            grid[other.x, other.y] = other

    print('Generating maze ...')
    st = time.time()

    branches, stack, visited, routes, maze = [], [], [], {}, {}
    grid = np.empty((m, n), dtype=Cell)

    # Initialize grid cells
    for i in range(m):
        for j in range(n):
            grid[i, j] = Cell(i, j)

    # Start maze generation
    grid[0, 0].visited = True
    current = grid[0, 0]
    stack.append((current.x, current.y))
    branches.append(stack.copy())
    visited.append((current.x, current.y))

    while len(visited) < m * n:
        if current.get_neightbors():
            i, j = random.choice(current.get_neightbors())
            current.visit(grid[i, j])
            branches.append(stack.copy())
            current = grid[i, j]
        else:
            stack.pop()
            i, j = stack[-1]
            branches.append(stack.copy())
            current = grid[i, j]
        if (m * n - len(visited)) % 10000 == 0:
            print(f'Cells left to visit: {m * n - len(visited):>g}')

    # Construct walls
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

    # Convert routes to graph
    graph = defaultdict(list)
    for k, _ in routes.items():
        graph[k[0]].append((1, k[1]))

    print(f'\nDone in {time.time() - st:3.0f} seconds')
    return (maze, routes, graph, branches)


# === Main Script ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Maze Generator and Path Finder")
    parser.add_argument('--size', type=str, default="25,35", help="Maze size (e.g. 25,35)")
    parser.add_argument('--prob', type=float, default=0.35, help="Wall probability for random maze")
    parser.add_argument('--use-random', action='store_true', help="Use random maze generator")
    parser.add_argument('--find-all', action='store_true', help="Recursively find all paths")
    parser.add_argument('--draw-all-paths', action='store_true', help="Draw all paths if --find-all is used") 
    args = parser.parse_args()

    # Maze configuration
    use_random_maze_gen = args.use_random
    prob = args.prob
    m, n = map(int, args.size.split(','))

    wallcolor, spcolor, pcolor = 'orange', 'blue', 'sienna'

    # Generate until a valid path exists
    path_exists = False
    attempt = 0

    while not path_exists:
        attempt += 1
        st = time.time()
        if use_random_maze_gen:
            a, b, endx, endy = 1.5, 1.5, m - .5, n - .5
            maze, _ = create_maze(m, n, prob)
            graph, _ = create_routes(maze)
        else:
            a, b, endx, endy = 1.5, 1.5, m + .5, n + .5
            maze, _, graph, _ = maze_digger(n, m)

        sp, V = Djikstra_heapq((a, b), (endx, endy), graph)
        sd = len(sp) - 1
        if sp:
            path_exists = True
            print(f'Attempt:{attempt:>5}: {int(time.time() - st):3} seconds; success!', end='\r')
        else:
            print(f'Attempt:{attempt:>5}: {int(time.time() - st):3} seconds; no path!', end='\r')

    print()
    print('Drawing maze ...')
    st = time.time()

    # === Draw Maze and Save ===
    fig1, ax1 = plt.subplots(figsize=(45, 45))
    ax1.set_aspect('equal')
    ax1.scatter([a, endx], [b, endy], marker='*', s=750, color='red', zorder=5)

    for k, v in maze.items():
        if v > 0:
            (xf, yf), (xt, yt) = k
            ax1.plot([xf, xt], [yf, yt], lw=3, color=wallcolor, zorder=1)

    ax1.axis('off')
    fig1.savefig('maze.png', dpi=300)
    plt.close(fig1)
    print(f'Done in {int(time.time() - st)} seconds')

    print(f'Shortest distance is {sd} steps')

    # === Save Shortest Path ===
    spx, spy = [p[0] for p in sp], [p[1] for p in sp]
    fig_path, ax_path = plt.subplots(figsize=(45, 45))
    ax_path.set_aspect('equal')
    ax_path.scatter([a, endx], [b, endy], marker='*', s=750, color='red', zorder=5)
    for k, v in maze.items():
        if v > 0:
            (xf, yf), (xt, yt) = k
            ax_path.plot([xf, xt], [yf, yt], lw=3, color=wallcolor, zorder=1)
    ax_path.plot(spx, spy, color=spcolor, lw=3, zorder=3)
    ax_path.axis('off')
    fig_path.savefig('shortestpath.png', dpi=300)
    plt.close(fig_path)
    print('Saved shortest path.')

    # === Optional Recursive Pathfinding ===
    if args.find_all:
        alls = 1
        sys.setrecursionlimit(30000)
        path = [(a, b)]
        paths, courses, rbstream, path_lengths, indices = [], [], [], [], []
        recur = -1
        btrack = counter = 0

        print('Starting recursion ...')
        st = time.time()
        find_paths((a, b), (endx, endy), graph)
        print(f'\nFinished in {timedelta(seconds=time.time() - st).total_seconds():.0f} second(s)')
        sys.setrecursionlimit(3000)

        # === Recursion Depth Analysis ===
        rb = np.array(rbstream).cumsum()
        maxima = [
            rb[i] for i in range(1, len(rb) - 1)
            if rb[i] > rb[i - 1] and rb[i] > rb[i + 1]
        ]

        print()
        print(f'Paths to destination: {counter}')
        print(f'Cul-de-sacs or loops: {len(maxima) - counter}')
        if paths:
            print(f'Length of shortest path: {min(path_lengths) - 1} steps')
            print(f'Length of longest path: {max(path_lengths) - 1} steps')
        print(f'The algorithm made {recur} recursive call(s) and {btrack} backtrack(s)')

        # === Save Recursion Graph ===
        fig2, ax2 = plt.subplots(figsize=(16, 9))
        ax2.plot(rb, drawstyle='steps-post')
        if indices:
            ax2.scatter(indices, [rb[k] for k in indices], marker='*', color='red', s=75, zorder=3)
        ax2.grid(axis='y', ls='solid')
        fig2.savefig('recursion.png', dpi=300)
        plt.close(fig2)
        print('Saved recursion stats.')

        # === Draw All Paths ===
        if args.draw_all_paths and paths:
            print('Drawing all paths ...')
            fig_all, ax_all = plt.subplots(figsize=(45, 45))
            ax_all.set_aspect('equal')
            ax_all.scatter([a, endx], [b, endy], marker='*', s=750, color='red', zorder=5)
            for k, v in maze.items():
                if v > 0:
                    (xf, yf), (xt, yt) = k
                    ax_all.plot([xf, xt], [yf, yt], lw=3, color=wallcolor, zorder=1)
            for p in paths:
                spx, spy = zip(*p)
                ax_all.plot(spx, spy, color=pcolor, alpha=0.3, lw=2)
            ax_all.axis('off')
            fig_all.savefig('all_paths.png', dpi=300)
            plt.close(fig_all)
            print('Saved all paths as all_paths.png.')