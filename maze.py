import numpy as np
import random
from enum import Enum


class Direction(Enum):
    """Directions that the robot can move"""
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class Maze:
    """Represents a maze environment for the robot to navigate"""

    def __init__(self, width=10, height=10, complexity=0.3):
        """
        Initialize a maze.

        Args:
            width: Width of the maze
            height: Height of the maze
            complexity: Probability of a cell being a wall (0.0 to 1.0)
        """
        self.width = width
        self.height = height
        self.complexity = complexity
        self.maze = None
        self.start = None
        self.goal = None
        self.generate_maze()

    def generate_maze(self):
        """
        Generate a random maze using a simple algorithm.
        0 = walkable path, 1 = wall
        """
        self.maze = np.random.choice(
            [0, 1], 
            size=(self.height, self.width),
            p=[1 - self.complexity, self.complexity]
        )

        # Ensure start position is walkable
        self.start = (0, 0)
        self.maze[self.start] = 0

        # Ensure goal position is walkable
        self.goal = (self.height - 1, self.width - 1)
        self.maze[self.goal] = 0

        # Create a path from start to goal (BFS-based simple path)
        self._ensure_path_exists()

    def _ensure_path_exists(self):
        """
        Ensure that there is at least one path from start to goal.
        Uses a simple flood-fill approach.
        """
        from collections import deque

        visited = set()
        queue = deque([self.start])
        visited.add(self.start)
        parent = {self.start: None}

        # BFS to find path to goal
        while queue:
            row, col = queue.popleft()

            if (row, col) == self.goal:
                # Found path, clear it
                current = self.goal
                while current is not None:
                    self.maze[current] = 0
                    current = parent.get(current)
                return

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = row + dr, col + dc
                if (
                    0 <= nr < self.height
                    and 0 <= nc < self.width
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (row, col)
                    queue.append((nr, nc))

        # If no path found, create a simple straight path
        for i in range(self.height):
            self.maze[i, 0] = 0
        for j in range(self.width):
            self.maze[self.height - 1, j] = 0

    def is_walkable(self, row, col):
        """
        Check if a position is walkable.

        Args:
            row: Row coordinate
            col: Column coordinate

        Returns:
            True if position is valid and walkable, False otherwise
        """
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        return self.maze[row, col] == 0

    def get_neighbors(self, position):
        """
        Get all walkable neighboring cells from a position.

        Args:
            position: Tuple (row, col)

        Returns:
            List of walkable neighboring positions
        """
        row, col = position
        neighbors = []

        for direction in Direction:
            dr, dc = direction.value
            new_row, new_col = row + dr, col + dc
            if self.is_walkable(new_row, new_col):
                neighbors.append((new_row, new_col))

        return neighbors

    def get_distance(self, pos1, pos2):
        """
        Calculate Manhattan distance between two positions.

        Args:
            pos1: Tuple (row, col)
            pos2: Tuple (row, col)

        Returns:
            Manhattan distance
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset(self):
        """
        Reset the maze (regenerate it).
        """
        self.generate_maze()

    def display(self):
        """
        Return a string representation of the maze.
        S = start, G = goal, # = wall, . = path
        """
        display_maze = []
        for row in range(self.height):
            row_str = ""
            for col in range(self.width):
                if (row, col) == self.start:
                    row_str += "S"
                elif (row, col) == self.goal:
                    row_str += "G"
                elif self.maze[row, col] == 1:
                    row_str += "#"
                else:
                    row_str += "."
            display_maze.append(row_str)
        return "\n".join(display_maze)
