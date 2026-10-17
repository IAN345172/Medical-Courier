"""
Search module - Implements BFS, DFS, and A* algorithms
"""

from collections import deque
import heapq

def bfs(env, start, goal):
    """
    Breadth-First Search (Uninformed)
    Guarantees shortest path in terms of number of steps
    """
    if start == goal:
        return [start]
    
    queue = deque([(start, [start])])
    visited = {start}
    
    nodes_expanded = 0
    
    while queue:
        current_pos, path = queue.popleft()
        nodes_expanded += 1
        
        for neighbor in env.get_neighbors(current_pos):
            if neighbor not in visited:
                if neighbor == goal:
                    print(f"BFS expanded {nodes_expanded} nodes")
                    return path + [neighbor]
                
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    print(f"BFS expanded {nodes_expanded} nodes - No path found")
    return None


def dfs(env, start, goal, max_depth=100):
    """
    Depth-First Search (Uninformed)
    Does NOT guarantee shortest path
    """
    if start == goal:
        return [start]
    
    stack = [(start, [start])]
    visited = {start}
    nodes_expanded = 0
    
    while stack:
        current_pos, path = stack.pop()
        nodes_expanded += 1
        
        if len(path) > max_depth:
            continue
        
        for neighbor in env.get_neighbors(current_pos):
            if neighbor not in visited:
                if neighbor == goal:
                    print(f"DFS expanded {nodes_expanded} nodes")
                    return path + [neighbor]
                
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
    
    print(f"DFS expanded {nodes_expanded} nodes - No path found")
    return None


def manhattan_distance(pos1, pos2):
    """Manhattan distance heuristic for A*"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def astar(env, start, goal):
    """
    A* Search (Informed)
    Uses Manhattan distance heuristic
    """
    if start == goal:
        return [start]
    
    # Priority queue: (f_score, g_score, position, path)
    heap = [(manhattan_distance(start, goal), 0, start, [start])]
    visited = {start: 0}  # position -> g_score
    
    nodes_expanded = 0
    
    while heap:
        f_score, g_score, current_pos, path = heapq.heappop(heap)
        nodes_expanded += 1
        
        if current_pos == goal:
            print(f"A* expanded {nodes_expanded} nodes")
            return path
        
        for neighbor in env.get_neighbors(current_pos):
            new_g = g_score + 1 + env.get_traffic_cost(neighbor) * 0.5  # Traffic penalty
            
            if neighbor not in visited or new_g < visited[neighbor]:
                visited[neighbor] = new_g
                h_score = manhattan_distance(neighbor, goal)
                new_f = new_g + h_score
                heapq.heappush(heap, (new_f, new_g, neighbor, path + [neighbor]))
    
    print(f"A* expanded {nodes_expanded} nodes - No path found")
    return None


def astar_with_priority(env, start, goal, urgency_weight=1.0):
    """
    A* Search with medical priority weighting
    Higher urgency = lower heuristic weight (faster but possibly longer path)
    """
    if start == goal:
        return [start]
    
    def heuristic(pos):
        # Weighted heuristic based on urgency
        base_h = manhattan_distance(pos, goal)
        return base_h / urgency_weight  # Higher urgency = smaller h = faster routing
    
    heap = [(heuristic(start), 0, start, [start])]
    visited = {start: 0}
    nodes_expanded = 0
    
    while heap:
        f_score, g_score, current_pos, path = heapq.heappop(heap)
        nodes_expanded += 1
        
        if current_pos == goal:
            print(f"A* (priority={urgency_weight}) expanded {nodes_expanded} nodes")
            return path
        
        for neighbor in env.get_neighbors(current_pos):
            traffic_penalty = env.get_traffic_cost(neighbor) * 0.3
            new_g = g_score + 1 + traffic_penalty
            
            if neighbor not in visited or new_g < visited[neighbor]:
                visited[neighbor] = new_g
                new_f = new_g + heuristic(neighbor)
                heapq.heappush(heap, (new_f, new_g, neighbor, path + [neighbor]))
    
    print(f"A* priority expanded {nodes_expanded} nodes - No path found")
    return None


if __name__ == "__main__":
    print("Search algorithms module loaded. Test with environment in main.py")