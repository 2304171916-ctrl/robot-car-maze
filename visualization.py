import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np


class MazeVisualizer:
    """
    Visualizes maze and robot paths using matplotlib.
    """

    def __init__(self, maze):
        """
        Initialize visualizer.

        Args:
            maze: Maze instance
        """
        self.maze = maze

    def visualize_path(self, path, title="Maze Solution", filename=None):
        """
        Visualize maze with solution path.

        Args:
            path: List of positions representing the path
            title: Title of the plot
            filename: If provided, save the figure to this file
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw maze
        maze_display = np.copy(self.maze.maze).astype(float)
        ax.imshow(maze_display, cmap="gray", alpha=0.3)

        # Draw walls
        for i in range(self.maze.height):
            for j in range(self.maze.width):
                if self.maze.maze[i, j] == 1:
                    rect = mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, linewidth=0, facecolor="black"
                    )
                    ax.add_patch(rect)

        # Draw path
        if path and len(path) > 0:
            path_array = np.array(path)
            ax.plot(
                path_array[:, 1],
                path_array[:, 0],
                "b-",
                linewidth=2,
                label="Path",
                alpha=0.7,
            )
            # Draw path points
            ax.scatter(
                path_array[:, 1],
                path_array[:, 0],
                c="blue",
                s=20,
                alpha=0.5,
            )

        # Draw start
        start_row, start_col = self.maze.start
        ax.scatter(
            start_col,
            start_row,
            c="green",
            s=200,
            marker="o",
            label="Start",
            edgecolors="darkgreen",
            linewidths=2,
        )

        # Draw goal
        goal_row, goal_col = self.maze.goal
        ax.scatter(
            goal_col,
            goal_row,
            c="red",
            s=200,
            marker="*",
            label="Goal",
            edgecolors="darkred",
            linewidths=2,
        )

        ax.set_xlim(-0.5, self.maze.width - 0.5)
        ax.set_ylim(self.maze.height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {filename}")

        return fig, ax

    def visualize_multiple_paths(
        self, paths_dict, title="Maze Solutions Comparison", filename=None
    ):
        """
        Visualize and compare multiple solution paths.

        Args:
            paths_dict: Dictionary with algorithm names as keys and paths as values
            title: Title of the plot
            filename: If provided, save the figure to this file
        """
        num_paths = len(paths_dict)
        fig, axes = plt.subplots(1, num_paths, figsize=(5 * num_paths, 5))

        if num_paths == 1:
            axes = [axes]

        colors = ["blue", "purple", "cyan", "orange"]

        for idx, (algo_name, path) in enumerate(paths_dict.items()):
            ax = axes[idx]

            # Draw maze
            maze_display = np.copy(self.maze.maze).astype(float)
            ax.imshow(maze_display, cmap="gray", alpha=0.3)

            # Draw walls
            for i in range(self.maze.height):
                for j in range(self.maze.width):
                    if self.maze.maze[i, j] == 1:
                        rect = mpatches.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            linewidth=0,
                            facecolor="black",
                        )
                        ax.add_patch(rect)

            # Draw path
            if path and len(path) > 0:
                path_array = np.array(path)
                color = colors[idx % len(colors)]
                ax.plot(
                    path_array[:, 1],
                    path_array[:, 0],
                    color=color,
                    linewidth=2,
                    alpha=0.7,
                )
                ax.scatter(
                    path_array[:, 1],
                    path_array[:, 0],
                    c=color,
                    s=15,
                    alpha=0.5,
                )

            # Draw start and goal
            start_row, start_col = self.maze.start
            ax.scatter(
                start_col,
                start_row,
                c="green",
                s=150,
                marker="o",
                edgecolors="darkgreen",
                linewidths=2,
            )

            goal_row, goal_col = self.maze.goal
            ax.scatter(
                goal_col,
                goal_row,
                c="red",
                s=150,
                marker="*",
                edgecolors="darkred",
                linewidths=2,
            )

            ax.set_xlim(-0.5, self.maze.width - 0.5)
            ax.set_ylim(self.maze.height - 0.5, -0.5)
            ax.set_aspect("equal")
            path_length = len(path) if path else 0
            ax.set_title(f"{algo_name}\n(Steps: {path_length})", fontweight="bold")
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"Saved comparison to {filename}")

        return fig, axes

    def animate_path(self, path, title="Robot Navigation", filename=None):
        """
        Create animation of robot moving along path.

        Args:
            path: List of positions representing the path
            title: Title of the animation
            filename: If provided, save the animation to this file
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw static maze elements
        maze_display = np.copy(self.maze.maze).astype(float)
        ax.imshow(maze_display, cmap="gray", alpha=0.3)

        for i in range(self.maze.height):
            for j in range(self.maze.width):
                if self.maze.maze[i, j] == 1:
                    rect = mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, linewidth=0, facecolor="black"
                    )
                    ax.add_patch(rect)

        # Draw start and goal
        start_row, start_col = self.maze.start
        ax.scatter(
            start_col,
            start_row,
            c="green",
            s=200,
            marker="o",
            label="Start",
            edgecolors="darkgreen",
            linewidths=2,
        )

        goal_row, goal_col = self.maze.goal
        ax.scatter(
            goal_col,
            goal_row,
            c="red",
            s=200,
            marker="*",
            label="Goal",
            edgecolors="darkred",
            linewidths=2,
        )

        # Initialize dynamic elements
        (line,) = ax.plot([], [], "b-", linewidth=2, alpha=0.7, label="Path")
        (scatter,) = ax.plot([], [], "bo", markersize=8, alpha=0.5)
        (robot,) = ax.plot([], [], "r*", markersize=15)

        ax.set_xlim(-0.5, self.maze.width - 0.5)
        ax.set_ylim(self.maze.height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        def animate(frame):
            if frame > 0 and path:
                path_so_far = path[:frame]
                path_array = np.array(path_so_far)
                line.set_data(path_array[:, 1], path_array[:, 0])
                scatter.set_data(path_array[:, 1], path_array[:, 0])
                robot.set_data([path_array[-1, 1]], [path_array[-1, 0]])
            return line, scatter, robot

        if path:
            anim = FuncAnimation(
                fig, animate, frames=len(path), interval=100, blit=True, repeat=True
            )

            if filename:
                anim.save(filename, writer="pillow")
                print(f"Saved animation to {filename}")

        return fig, ax

    def plot_learning_curve(self, rewards, steps, filename=None):
        """
        Plot learning curves for Q-Learning agent.

        Args:
            rewards: List of episode rewards
            steps: List of episode steps
            filename: If provided, save the figure to this file
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot rewards
        ax1.plot(rewards, linewidth=1, alpha=0.7)
        # Moving average
        window = min(50, len(rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax1.plot(moving_avg, linewidth=2, color="red", label=f"Moving Avg (window={window})")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Q-Learning: Episode Rewards")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot steps
        ax2.plot(steps, linewidth=1, alpha=0.7)
        if window > 1:
            moving_avg_steps = np.convolve(steps, np.ones(window) / window, mode="valid")
            ax2.plot(
                moving_avg_steps, linewidth=2, color="red", label=f"Moving Avg (window={window})"
            )
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Number of Steps")
        ax2.set_title("Q-Learning: Steps per Episode")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"Saved learning curve to {filename}")

        return fig, (ax1, ax2)
