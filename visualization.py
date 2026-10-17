"""
Visualization module - Displays routes and environment
"""

import matplotlib.pyplot as plt
import numpy as np

def visualize_path(env, path, title="Delivery Route"):
    """
    Visualize the path taken by the courier
    """
    if path is None:
        print("No path to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create color grid
    color_grid = np.zeros((env.size, env.size, 3))
    
    for i in range(env.size):
        for j in range(env.size):
            if env.grid[i][j] == 1:  # Hospital
                color_grid[i, j] = [0, 1, 0]  # Green
            elif env.grid[i][j] == 2:  # Depot
                color_grid[i, j] = [0, 0, 1]  # Blue
            elif env.traffic[i][j] >= 3:  # Heavy traffic
                color_grid[i, j] = [1, 0, 0]  # Red
            else:
                color_grid[i, j] = [0.9, 0.9, 0.9]  # Light gray
    
    ax.imshow(color_grid, interpolation='nearest')
    
    # Plot the path
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, 'y-', linewidth=2, marker='o', markersize=8, 
                markerfacecolor='yellow', markeredgecolor='black', label='Route')
        
        # Mark start and end
        ax.plot(path_x[0], path_y[0], 'bo', markersize=15, label='Start (Depot)')
        ax.plot(path_x[-1], path_y[-1], 'r*', markersize=15, label='End (Hospital)')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_visualization(env, path, filename, title="Delivery Route"):
    """Save visualization as an image file (PNG)"""
    if path is None:
        print("No path to save")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create color grid
    color_grid = np.zeros((env.size, env.size, 3))
    
    for i in range(env.size):
        for j in range(env.size):
            if env.grid[i][j] == 1:  # Hospital
                color_grid[i, j] = [0, 1, 0]
            elif env.grid[i][j] == 2:  # Depot
                color_grid[i, j] = [0, 0, 1]
            elif env.traffic[i][j] >= 3:  # Traffic
                color_grid[i, j] = [1, 0, 0]
            else:
                color_grid[i, j] = [0.9, 0.9, 0.9]
    
    ax.imshow(color_grid, interpolation='nearest')
    
    # Plot path
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, 'y-', linewidth=3, marker='o', markersize=8,
                markerfacecolor='yellow', markeredgecolor='black', label='Route')
        ax.plot(path_x[0], path_y[0], 'bo', markersize=15, label='Depot')
        ax.plot(path_x[-1], path_y[-1], 'r*', markersize=15, label='Hospital')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save as image
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()  # Don't show, just save


def save_all_routes(env, bfs_path, astar_path, priority_path, emergency, filename):
    """Save comparison of all routes as one image"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ["BFS (Uninformed)", "A* (Informed)", f"A* Priority ({emergency['urgency']})"]
    paths = [bfs_path, astar_path, priority_path]
    
    for idx, (ax, title, path) in enumerate(zip(axes, titles, paths)):
        # Create grid
        color_grid = np.zeros((env.size, env.size, 3))
        for i in range(env.size):
            for j in range(env.size):
                if env.grid[i][j] == 1:
                    color_grid[i, j] = [0, 1, 0]
                elif env.grid[i][j] == 2:
                    color_grid[i, j] = [0, 0, 1]
                elif env.traffic[i][j] >= 3:
                    color_grid[i, j] = [1, 0, 0]
                else:
                    color_grid[i, j] = [0.9, 0.9, 0.9]
        
        ax.imshow(color_grid, interpolation='nearest')
        
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, 'yellow', linewidth=2, marker='o', markersize=6)
            ax.plot(path_x[0], path_y[0], 'bo', markersize=10)
            ax.plot(path_x[-1], path_y[-1], 'r*', markersize=10)
            ax.text(0.02, 0.98, f"Steps: {len(path)-1}", transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Route Comparison - {emergency['supply'].upper()} Delivery")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()    


def compare_routes(env, bfs_path, astar_path, emergency):
    """
    Compare BFS and A* routes side by side
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    titles = ["BFS Route", f"A* Route (Priority: {emergency['urgency']})"]
    paths = [bfs_path, astar_path]
    
    for idx, (ax, title, path) in enumerate(zip(axes, titles, paths)):
        # Create color grid
        color_grid = np.zeros((env.size, env.size, 3))
        
        for i in range(env.size):
            for j in range(env.size):
                if env.grid[i][j] == 1:
                    color_grid[i, j] = [0, 1, 0]
                elif env.grid[i][j] == 2:
                    color_grid[i, j] = [0, 0, 1]
                elif env.traffic[i][j] >= 3:
                    color_grid[i, j] = [1, 0, 0]
                else:
                    color_grid[i, j] = [0.9, 0.9, 0.9]
        
        ax.imshow(color_grid, interpolation='nearest')
        
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, 'yellow', linewidth=2, marker='o', markersize=6)
            ax.plot(path_x[0], path_y[0], 'go', markersize=12)
            ax.plot(path_x[-1], path_y[-1], 'r*', markersize=12)
        
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
        
        # Add info text
        if path:
            info = f"Length: {len(path)} steps"
            ax.text(0.02, 0.98, info, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f"Route Comparison - {emergency['supply'].upper()} Delivery ({emergency['urgency']})")
    plt.tight_layout()
    plt.show()


def print_path_details(path, algorithm_name):
    """Print detailed path information"""
    if path is None:
        print(f"{algorithm_name}: No path found")
        return
    
    print(f"\n=== {algorithm_name} Path Details ===")
    print(f"Steps: {len(path) - 1}")
    print(f"Path length: {len(path)} nodes")
    print("Route:")
    for i, step in enumerate(path):
        print(f"  Step {i}: {step}")
    print("=" * 35)


# Simple fallback visualization if matplotlib fails
def simple_print_path(env, path, title="Route"):
    """Text-based visualization if matplotlib isn't working"""
    if path is None:
        print("No path found")
        return
    
    print(f"\n📍 {title}")
    print(f"   Path: {' → '.join([str(p) for p in path])}")
    print(f"   Total steps: {len(path) - 1}")


if __name__ == "__main__":
    print("Visualization module loaded")