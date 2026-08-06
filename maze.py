#!/usr/bin/env python3
"""
Maze Generator and Path Finder
------------------------------
This script generates a maze (either random or recursive backtracking),
finds the shortest path using Dijkstra's algorithm, and optionally
recursively enumerates all possible paths.

It can visualize the maze, shortest path, recursion statistics,
and all paths using matplotlib.

Usage:
    python maze.py -g random -p 0.35 -s 25,35 -d -o outputs
"""

import argparse
import os
import sys
import time
from datetime import timedelta

import maze_gen
import pathfind
import visualize


def main():
    parser = argparse.ArgumentParser(description="Maze Generator and Path Finder")
    parser.add_argument('-s', '--size', type=str, default="25,35",
                        help="Maze dimensions (width,height)")
    parser.add_argument('-g', '--generator', choices=('digger', 'random'),
                        default='digger',
                        help="Maze generator (default: digger)")
    parser.add_argument('-p', '--prob', type=float, default=0.35,
                        help="Wall probability for random maze  [random only]")
    parser.add_argument('-a', '--all', action='store_true',
                        help="Recursively enumerate every possible path")
    parser.add_argument('-d', '--draw', action='store_true',
                        help="Draw all discovered paths  (implies --all)")
    parser.add_argument('-o', '--out-dir', type=str, default='.',
                        help="Directory for output PNGs  (default: cwd)")
    args = parser.parse_args()

    if args.draw:
        args.all = True

    if args.generator == 'digger' and args.prob != 0.35:
        parser.error('--prob is only meaningful with --generator random')

    m, n = map(int, args.size.split(','))
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    a = b = 1.5                 # start  (top‑left)

    # ------------------------------------------------------------------
    # generate until a valid path exists
    # ------------------------------------------------------------------
    path_exists = False
    attempt = 0
    while not path_exists:
        attempt += 1
        st = time.time()
        if args.generator == 'random':
            endx, endy = m - 0.5, n - 0.5
            maze, _ = maze_gen.create_maze(m, n, args.prob)
            graph, _ = maze_gen.create_routes(maze, m, n)
        else:
            endx, endy = m + 0.5, n + 0.5
            # NOTE: maze_digger(m, n) treats its args as (rows, cols).
            #       The original code swapped them for a portrait default,
            #       so we keep that swap for backward‑compatibility.
            maze, _, graph, _ = maze_gen.maze_digger(n, m)

        sp, _ = pathfind.dijkstra((a, b), (endx, endy), graph)
        sd = len(sp) - 1
        if sp:
            path_exists = True
            print(f'Attempt:{attempt:>5}: {int(time.time() - st):3} seconds; success!')
        else:
            print(f'Attempt:{attempt:>5}: {int(time.time() - st):3} seconds; no path!')

    # ------------------------------------------------------------------
    # save images
    # ------------------------------------------------------------------
    print('Drawing maze ...')
    st = time.time()
    visualize.save_maze(maze, (a, b), (endx, endy), out_dir)
    print(f'Done in {int(time.time() - st)} seconds')

    print(f'Shortest distance is {sd} steps')

    visualize.save_shortest_path(maze, sp, (a, b), (endx, endy), out_dir)
    print('Saved shortest path.')

    # ------------------------------------------------------------------
    # optional recursive all‑paths search
    # ------------------------------------------------------------------
    if args.all:
        sys.setrecursionlimit(30000)
        ps = pathfind.PathSearch(graph, (a, b), (endx, endy))
        print('Starting recursion ...')
        st = time.time()
        ps.search()
        print()
        print(f'Finished in {timedelta(seconds=time.time() - st).total_seconds():.0f} second(s)')
        sys.setrecursionlimit(3000)

        ps.report()
        visualize.save_recursion_graph(ps.rbstream, ps.indices, out_dir)
        print('Saved recursion stats.')

        if args.draw and ps.paths:
            print('Drawing all paths ...')
            visualize.save_all_paths(maze, ps.paths, (a, b), (endx, endy), out_dir)
            print('Saved all paths as all_paths.png.')


if __name__ == '__main__':
    main()
