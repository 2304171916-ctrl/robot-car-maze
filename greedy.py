import heapq
from maze import Maze


class GreedyAgent:
    """
    Greedy agent that uses heuristic-based pathfinding.
    Implements Greedy Best-First Search algorithm.
    """

    def __init__(self, maze):
        """
        Initialize Greedy agent.

        Args:
            maze: Maze instance
        """
        self.maze = maze
        self.visited_count = 0
        self.path_found = None

    def heuristic(self, position):
        """
        Calculate heuristic value (Manhattan distance to goal).

        Args:
            position: Current position

        Returns:
            Heuristic value (distance to goal)
        """
        return self.maze.get_distance(position, self.maze.goal)

    def solve_greedy_bfs(self):
        """
        Solve maze using Greedy Best-First Search.
        Always expands the node closest to the goal.

        Returns:
            Path from start to goal, or None if no path exists
        """
        self.visited_count = 0
        visited = set()
        # Priority queue: (heuristic_value, position)
        heap = [(self.heuristic(self.maze.start), self.maze.start)]
        parent = {self.maze.start: None}

        while heap:
            _, current = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)
            self.visited_count += 1

            if current == self.maze.goal:
                # Reconstruct path
                path = []
                node = self.maze.goal
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return path[::-1]

            for neighbor in self.maze.get_neighbors(current):
                if neighbor not in visited:
                    if neighbor not in parent:
                        parent[neighbor] = current
                        h_value = self.heuristic(neighbor)
                        heapq.heappush(heap, (h_value, neighbor))

        return None  # No path found

    def solve_a_star(self):
        """
        Solve maze using A* algorithm.
        Combines heuristic with actual cost (g-score).

        Returns:
            Path from start to goal, or None if no path exists
        """
        self.visited_count = 0
        visited = set()
        # Priority queue: (f_value, position)
        # f = g + h, where g is cost from start, h is heuristic
        heap = [(self.heuristic(self.maze.start), self.maze.start)]
        g_score = {self.maze.start: 0}
        parent = {self.maze.start: None}

        while heap:
            _, current = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)
            self.visited_count += 1

            if current == self.maze.goal:
                # Reconstruct path
                path = []
                node = self.maze.goal
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return path[::-1]

            current_g = g_score[current]

            for neighbor in self.maze.get_neighbors(current):
                if neighbor not in visited:
                    new_g = current_g + 1

                    if neighbor not in g_score or new_g < g_score[neighbor]:
                        g_score[neighbor] = new_g
                        h_value = self.heuristic(neighbor)
                        f_value = new_g + h_value
                        parent[neighbor] = current
                        heapq.heappush(heap, (f_value, neighbor))

        return None  # No path found

    def solve_dijkstra(self):
        """
        Solve maze using Dijkstra's algorithm.
        Guarantees shortest path (without heuristic).

        Returns:
            Path from start to goal, or None if no path exists
        """
        self.visited_count = 0
        visited = set()
        # Priority queue: (distance, position)
        heap = [(0, self.maze.start)]
        distances = {self.maze.start: 0}
        parent = {self.maze.start: None}

        while heap:
            current_dist, current = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)
            self.visited_count += 1

            if current == self.maze.goal:
                # Reconstruct path
                path = []
                node = self.maze.goal
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return path[::-1]

            for neighbor in self.maze.get_neighbors(current):
                if neighbor not in visited:
                    new_dist = current_dist + 1

                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        parent[neighbor] = current
                        heapq.heappush(heap, (new_dist, neighbor))

        return None  # No path found

    def get_path(self, algorithm="a_star"):
        """
        Get path using specified algorithm.

        Args:
            algorithm: Algorithm to use ('greedy', 'a_star', 'dijkstra')

        Returns:
            Path from start to goal
        """
        if algorithm == "greedy":
            self.path_found = self.solve_greedy_bfs()
        elif algorithm == "a_star":
            self.path_found = self.solve_a_star()
        elif algorithm == "dijkstra":
            self.path_found = self.solve_dijkstra()
        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                "Use 'greedy', 'a_star', or 'dijkstra'"
            )
        return self.path_found

    def get_visited_count(self):
        """
        Get number of nodes visited during last search.

        Returns:
            Number of visited nodes
        """
        return self.visited_count
