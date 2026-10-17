"""
Run this ONCE to generate all images for your presentation
"""

from environment import MedicalEnvironment
from search import bfs, astar, astar_with_priority
from emergencies import get_all_emergencies
from visualization import save_visualization, save_all_routes

def generate_all_images():
    print("Generating presentation images...")
    print("="*50)
    
    # Create environment
    env = MedicalEnvironment(size=10)
    
    # Get all emergencies
    emergencies = get_all_emergencies()
    
    # For each emergency, generate images
    for emergency in emergencies:
        print(f"\nProcessing {emergency['id']} - {emergency['supply']}...")
        
        start = env.depot
        goal = emergency['hospital']
        
        # Run algorithms
        bfs_path = bfs(env, start, goal)
        astar_path = astar(env, start, goal)
        
        weight = {"critical": 5.0, "moderate": 2.0, "low": 1.0}
        priority_path = astar_with_priority(env, start, goal, 
                                            urgency_weight=weight.get(emergency['urgency'], 1.0))
        
        # Save individual route images
        if bfs_path:
            save_visualization(env, bfs_path, 
                              f"images/{emergency['id']}_bfs.png",
                              f"BFS Route - {emergency['supply']}")
        
        if priority_path:
            save_visualization(env, priority_path,
                              f"images/{emergency['id']}_priority.png",
                              f"A* Priority Route - {emergency['supply']} ({emergency['urgency']})")
        
        # Save comparison image
        save_all_routes(env, bfs_path, astar_path, priority_path, emergency,
                       f"images/{emergency['id']}_comparison.png")
    
    print("\n" + "="*50)
    print("✅ All images saved to 'images/' folder!")
    print("You can now insert these PNG files into your presentation slides.")

if __name__ == "__main__":
    import os
    os.makedirs("images", exist_ok=True)
    generate_all_images()