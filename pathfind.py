"""Pathfinding: Dijkstra (A*-like) and recursive all‑paths search."""

import heapq

import numpy as np

from maze_gen import heuristic


# ---------------------------------------------------------------------------
# shortest path  (Dijkstra + Euclidean heuristic  ≃ A*)
# ---------------------------------------------------------------------------

def dijkstra(source, destination, graph):
    """Return ``(path, visited_vertices)`` for the shortest route.

    Uses a cost‑tracking dict for O(1) duplicate‑entry checks instead
    of scanning the heap.
    """
    best_cost = {}
    h = []                # (priority, cost, vertex, predecessor)
    vmap = {}

    heapq.heappush(h, (0 + heuristic(source, destination), 0, source, source))

    while h:
        _, currcost, currvtx, prevvtx = heapq.heappop(h)
        vmap[currvtx] = prevvtx
        if currvtx == destination:
            break
        for edge_cost, neigh in graph[currvtx]:
            new_cost = currcost + edge_cost
            if neigh in vmap:
                continue
            if neigh not in best_cost or new_cost < best_cost[neigh]:
                best_cost[neigh] = new_cost
                heapq.heappush(
                    h,
                    (
                        new_cost + heuristic(neigh, destination),
                        new_cost,
                        neigh,
                        currvtx,
                    ),
                )

    if destination not in vmap:
        return [], list(vmap.keys())

    # walk back from destination
    path = [destination]
    y = destination
    while y != source:
        path.append(vmap[y])
        y = vmap[y]
    path.reverse()
    return path, list(vmap.keys())


# ---------------------------------------------------------------------------
# recursive all‑paths search
# ---------------------------------------------------------------------------

class PathSearch:
    """Depth‑first exhaustive search over every route from source to dest.

    Usage::

        ps = PathSearch(graph, source, dest)
        ps.search()
        ps.report()
    """

    def __init__(self, graph, source, destination):
        self.graph = graph
        self.source = source
        self.destination = destination

        # search state
        self.paths: list = []
        self.recur = -1
        self.btrack = 0
        self.counter = 0
        self.rbstream: list = []       # +1 = recurse, –1 = backtrack
        self.path_lengths: list = []
        self.indices: list = []        # rbstream indices where a full path is found

        # transient during recursion
        self._current_path: list = []

    # -- public API ---------------------------------------------------------

    def search(self):
        """Run the depth‑first search."""
        self._current_path = [self.source]
        self._dfs(self.source)

    def report(self):
        """Print a summary of the recursive search."""
        rb = np.array(self.rbstream).cumsum()
        maxima = [
            rb[i] for i in range(1, len(rb) - 1)
            if rb[i] > rb[i - 1] and rb[i] > rb[i + 1]
        ]
        print()
        print(f'Paths to destination: {self.counter}')
        print(f'Cul-de-sacs or loops: {len(maxima) - self.counter}')
        if self.paths:
            print(f'Length of shortest path: {min(self.path_lengths) - 1} steps')
            print(f'Length of longest path:  {max(self.path_lengths) - 1} steps')
        print(
            f'The algorithm made {self.recur} recursive call(s) '
            f'and {self.btrack} backtrack(s)'
        )

    # -- internals ----------------------------------------------------------

    def _dfs(self, currvtx):
        self.recur += 1
        if self.recur > 0:
            self.rbstream.append(1)

        if currvtx == self.destination:
            self.counter += 1
            print(f'\tPaths found: {self.counter}', end='\r')
            self.paths.append(self._current_path.copy())
            self.path_lengths.append(len(self._current_path))
            self.indices.append(len(self.rbstream) - 1)
            return

        for _, nextvtx in sorted(self.graph[currvtx], reverse=True):
            if nextvtx not in self._current_path:
                self._current_path.append(nextvtx)
                self._dfs(nextvtx)
                self.btrack += 1
                self.rbstream.append(-1)
                self._current_path.pop()
