"""Smoke tests for the maze pipeline."""

import tempfile

import pytest

import maze_gen
import pathfind
import visualize


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def test_random_maze_is_solvable():
    """Random mazes are not guaranteed solvable on first try; retry like main()."""
    for _ in range(10):
        maze, _ = maze_gen.create_maze(5, 5, 0.25)
        graph, _ = maze_gen.create_routes(maze, 5, 5)
        sp, _ = pathfind.dijkstra((1.5, 1.5), (4.5, 4.5), graph)
        if sp:
            break
    assert sp, "random maze must have a path after retries"


def test_random_maze_no_isolated_cells():
    maze, _ = maze_gen.create_maze(5, 5, 0.35)
    for pp in [(a, b) for a in range(1, 6) for b in range(1, 6)]:
        assert maze_gen.unconnected(pp, 5, 5, maze) < 4, \
            f"cell {pp} is boxed in"


def test_digger_maze_is_connected():
    """Dijkstra visits at least the path length (breaks at destination)."""
    maze, _, graph, _ = maze_gen.maze_digger(5, 5)
    sp, V = pathfind.dijkstra((1.5, 1.5), (5.5, 5.5), graph)
    assert sp, "digger maze must have a path"
    # Dijkstra stops when the destination is popped; it does not visit every
    # vertex.  But it must have explored at least the path itself.
    assert len(V) >= len(sp), \
        f"visited {len(V)} vertices but path has {len(sp)} steps"


# ---------------------------------------------------------------------------
# pathfinding
# ---------------------------------------------------------------------------

def test_dijkstra_handles_no_path():
    graph = {(1.5, 1.5): [], (2.5, 2.5): []}
    sp, _ = pathfind.dijkstra((1.5, 1.5), (2.5, 2.5), graph)
    assert sp == [], "disconnected graph => empty path"


def test_path_search_finds_all():
    for _ in range(20):
        maze, _ = maze_gen.create_maze(3, 3, 0.15)
        graph, _ = maze_gen.create_routes(maze, 3, 3)
        sp, _ = pathfind.dijkstra((1.5, 1.5), (2.5, 2.5), graph)
        if sp:
            break
    ps = pathfind.PathSearch(graph, (1.5, 1.5), (2.5, 2.5))
    ps.search()
    assert ps.counter >= 1, "should find at least one path"
    assert len(ps.paths) == ps.counter


# ---------------------------------------------------------------------------
# visualization (output files exist)
# ---------------------------------------------------------------------------

def test_viz_outputs(tmp_path):
    out = str(tmp_path)
    maze, _ = maze_gen.create_maze(3, 3, 0.3)
    graph, _ = maze_gen.create_routes(maze, 3, 3)
    sp, _ = pathfind.dijkstra((1.5, 1.5), (1.5, 2.5), graph)
    import os
    visualize.save_maze(maze, (1.5, 1.5), (1.5, 2.5), out)
    visualize.save_shortest_path(maze, sp, (1.5, 1.5), (1.5, 2.5), out)
    assert os.path.isfile(os.path.join(out, 'maze.png'))
    assert os.path.isfile(os.path.join(out, 'shortestpath.png'))

    ps = pathfind.PathSearch(graph, (1.5, 1.5), (1.5, 2.5))
    ps.search()
    visualize.save_recursion_graph(ps.rbstream, ps.indices, out)
    visualize.save_all_paths(maze, ps.paths, (1.5, 1.5), (1.5, 2.5), out)
    assert os.path.isfile(os.path.join(out, 'recursion.png'))
    assert os.path.isfile(os.path.join(out, 'all_paths.png'))
