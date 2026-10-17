"""
Environment module - Creates the grid world for medical deliveries
"""

import random

class MedicalEnvironment:
    def __init__(self, size=10):
        self.size = size
        self.grid = self.create_grid()
        self.depot = self.find_depot()
        self.hospitals = self.find_hospitals()
        self.traffic = self.generate_traffic()
    
    def create_grid(self):
        """Create a 10x10 grid with hospitals and depot"""
        # 0 = road, 1 = hospital, 2 = depot, 3 = traffic/obstacle
        grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        
        # Place depot at center
        depot_pos = (self.size // 2, self.size // 2)
        grid[depot_pos[0]][depot_pos[1]] = 2
        
        # Place 5 random hospitals
        hospitals_placed = 0
        while hospitals_placed < 5:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if grid[x][y] == 0:
                grid[x][y] = 1
                hospitals_placed += 1
        
        return grid
    
    def find_depot(self):
        """Find depot position in grid"""
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 2:
                    return (i, j)
        return (self.size // 2, self.size // 2)
    
    def find_hospitals(self):
        """Find all hospital positions"""
        hospitals = []
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 1:
                    hospitals.append((i, j))
        return hospitals
    
    def generate_traffic(self):
        """Generate random traffic cells"""
        traffic_grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for _ in range(8):  # 8 traffic cells
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if self.grid[x][y] == 0:  # Only on roads
                traffic_grid[x][y] = random.randint(1, 3)  # Traffic level 1-3
        return traffic_grid
    
    def is_blocked(self, pos):
        """Check if a position is blocked by traffic"""
        x, y = pos
        return self.traffic[x][y] >= 3  # Level 3 traffic is blocked
    
    def get_traffic_cost(self, pos):
        """Get traffic penalty for a position"""
        x, y = pos
        return self.traffic[x][y]
    
    def get_neighbors(self, pos):
        """Get valid neighboring positions"""
        x, y = pos
        neighbors = []
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if not self.is_blocked((nx, ny)):
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def display(self):
        """Print the environment grid"""
        symbols = {0: '.', 1: 'H', 2: 'D', 3: 'X'}
        print("\n=== Environment Map ===")
        print("Legend: D=Depot, H=Hospital, .=Road, X=Traffic\n")
        
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if self.traffic[i][j] >= 3:
                    row += " X "
                elif self.grid[i][j] == 1:
                    row += " H "
                elif self.grid[i][j] == 2:
                    row += " D "
                else:
                    row += " . "
            print(row)
        print()


if __name__ == "__main__":
    # Test the environment
    env = MedicalEnvironment(10)
    env.display()
    print(f"Depot at: {env.depot}")
    print(f"Hospitals at: {env.hospitals}")