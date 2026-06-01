#!/usr/bin/env python3
"""
Main script for Robot Car Maze Solver.
Demonstrates Q-Learning and Greedy algorithms for maze navigation.
"""

import argparse
import sys
from maze import Maze
from q_learning import QLearningAgent
from greedy import GreedyAgent
from visualization import MazeVisualizer
import matplotlib.pyplot as plt


def main():
    """
    Main function to run the maze solver demonstration.
    """
    parser = argparse.ArgumentParser(
        description="Robot Car Maze Solver using Q-Learning and Greedy algorithms"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=15,
        help="Width of the maze (default: 15)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=15,
        help="Height of the maze (default: 15)",
    )
    parser.add_argument(
        "--complexity",
        type=float,
        default=0.25,
        help="Maze complexity (wall probability, 0.0-1.0, default: 0.25)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of Q-Learning episodes (default: 500)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for Q-Learning (default: 0.1)",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.95,
        help="Discount factor for Q-Learning (default: 0.95)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Initial epsilon for Q-Learning (default: 0.2)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization (do not display plots)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        import numpy as np
        import random
        np.random.seed(args.seed)
        random.seed(args.seed)

    print("="*60)
    print("Robot Car Maze Solver")
    print("="*60)

    # Create maze
    print(f"\nGenerating maze ({args.height}x{args.width})...")
    maze = Maze(
        width=args.width,
        height=args.height,
        complexity=args.complexity,
    )
    print("Maze generated successfully!")

    # Display maze
    print("\nMaze layout (S=start, G=goal, #=wall, .=path):")
    print(maze.display())

    # Create visualizer
    visualizer = MazeVisualizer(maze)

    # Test Greedy Algorithms
    print("\n" + "="*60)
    print("Testing Greedy Algorithms")
    print("="*60)

    greedy_agent = GreedyAgent(maze)
    greedy_results = {}

    algorithms = ["dijkstra", "a_star", "greedy"]
    for algo in algorithms:
        print(f"\nRunning {algo.upper()}...")
        path = greedy_agent.get_path(algorithm=algo)
        visited = greedy_agent.get_visited_count()

        if path:
            path_length = len(path) - 1  # Steps, not nodes
            print(f"  Path found! Steps: {path_length}, Nodes visited: {visited}")
            greedy_results[algo.upper()] = path
        else:
            print(f"  No path found!")
            greedy_results[algo.upper()] = None

    # Train Q-Learning Agent
    print("\n" + "="*60)
    print("Training Q-Learning Agent")
    print("="*60)

    ql_agent = QLearningAgent(
        maze,
        learning_rate=args.learning_rate,
        discount_factor=args.discount,
        epsilon=args.epsilon,
    )

    ql_agent.train(num_episodes=args.episodes)

    # Test Q-Learning Agent
    print("\nTesting trained Q-Learning agent...")
    ql_path = ql_agent.get_policy_path()
    if ql_path:
        ql_steps = len(ql_path) - 1
        print(f"  Path found! Steps: {ql_steps}")
    else:
        print("  No path found!")

    greedy_results["Q-Learning"] = ql_path

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"\nMaze size: {args.height}x{args.width}")
    print(f"Start: {maze.start}")
    print(f"Goal: {maze.goal}")
    print("\nResults:")
    for algo_name, path in greedy_results.items():
        if path:
            steps = len(path) - 1
            print(f"  {algo_name:12} - Steps: {steps:4d}")
        else:
            print(f"  {algo_name:12} - No solution found")

    # Visualization
    if not args.no_viz:
        print("\n" + "="*60)
        print("Generating Visualizations")
        print("="*60)

        # Single path visualization (Q-Learning)
        if ql_path:
            print("\nVisualizing Q-Learning solution...")
            visualizer.visualize_path(
                ql_path,
                title="Maze Solution: Q-Learning",
                filename=f"{args.save_dir}/qlearning_solution.png",
            )

        # Compare all algorithms
        print("Comparing all algorithms...")
        valid_results = {
            k: v for k, v in greedy_results.items() if v is not None
        }
        if len(valid_results) > 0:
            visualizer.visualize_multiple_paths(
                valid_results,
                title="Maze Solutions: Algorithm Comparison",
                filename=f"{args.save_dir}/comparison.png",
            )

        # Plot learning curves
        print("Plotting Q-Learning learning curves...")
        visualizer.plot_learning_curve(
            ql_agent.episode_rewards,
            ql_agent.episode_steps,
            filename=f"{args.save_dir}/learning_curves.png",
        )

        print("\nDisplaying plots...")
        plt.show()

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
