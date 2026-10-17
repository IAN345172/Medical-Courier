"""
Agent module - Defines the Medical Courier Agent using PEAS framework
"""

from search import bfs, dfs, astar_with_priority

class MedicalCourierAgent:
    """
    PEAS Framework:
    - Performance: Delivery time, priority fulfillment, fuel efficiency
    - Environment: Grid city with hospitals, depot, traffic
    - Actuators: Move, pick up supply, deliver supply
    - Sensors: Position, hospital locations, urgency level, traffic ahead
    """
    
    def __init__(self, environment, knowledge_base, ml_predictor):
        self.env = environment
        self.kb = knowledge_base
        self.ml = ml_predictor
        self.position = environment.depot
        self.current_supply = None
        self.delivery_history = []
    
    def perceive(self, emergency):
        """Sense the current emergency request"""
        return {
            "position": self.position,
            "goal": emergency["hospital"],
            "urgency": emergency["urgency"],
            "supply": emergency["supply"],
            "time_window": emergency["time_window"],
            "traffic_ahead": self.check_traffic_ahead(emergency["hospital"])
        }
    
    def check_traffic_ahead(self, goal):
        """Check if there's traffic on the path to goal (simplified)"""
        x, y = self.position
        gx, gy = goal
        # Check a few cells in the general direction
        if abs(gx - x) > 0:
            check_x = x + (1 if gx > x else -1)
            if 0 <= check_x < self.env.size:
                return self.env.traffic[check_x][y] >= 2
        return False
    
    def decide_action(self, emergency):
        """Decide which search algorithm to use based on urgency"""
        perception = self.perceive(emergency)
        
        # Apply knowledge base rules
        priority_score, rules = self.kb.infer_priority(emergency, self.env)
        
        # Choose algorithm based on urgency
        if emergency["urgency"] == "critical":
            print("⚠️ CRITICAL EMERGENCY - Using A* with maximum priority")
            path = astar_with_priority(self.env, self.position, emergency["hospital"], urgency_weight=5.0)
        elif emergency["urgency"] == "moderate":
            print("📋 MODERATE EMERGENCY - Using A* with medium priority")
            path = astar_with_priority(self.env, self.position, emergency["hospital"], urgency_weight=2.0)
        else:
            print("✅ LOW PRIORITY - Using BFS for optimal route")
            path = bfs(self.env, self.position, emergency["hospital"])
        
        # Use ML to predict if delivery will be on time
        if path:
            distance = len(path)
            traffic_cost = sum(self.env.get_traffic_cost(pos) for pos in path)
            ml_prediction = self.ml.predict(distance, emergency["urgency"], traffic_cost, emergency["time_window"])
            
            return {
                "action": "deliver",
                "path": path,
                "priority_score": priority_score,
                "rules_applied": rules,
                "ml_prediction": ml_prediction
            }
        else:
            return {
                "action": "no_path_found",
                "path": None,
                "priority_score": priority_score,
                "rules_applied": rules,
                "ml_prediction": "Cannot deliver - no path"
            }
    
    def act(self, decision, emergency):
        """Execute the delivery action"""
        if decision["action"] == "deliver" and decision["path"]:
            self.position = emergency["hospital"]
            self.current_supply = emergency["supply"]
            
            delivery_record = {
                "hospital": emergency["hospital"],
                "supply": emergency["supply"],
                "urgency": emergency["urgency"],
                "path_length": len(decision["path"]),
                "ml_prediction": decision["ml_prediction"],
                "rules_applied": decision["rules_applied"]
            }
            self.delivery_history.append(delivery_record)
            return True
        return False


if __name__ == "__main__":
    # Test the agent (requires other modules)
    print("Agent module loaded. Run main.py to test full system.")