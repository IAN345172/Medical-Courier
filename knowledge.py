"""
Knowledge module - Rule-based system for medical emergency reasoning
"""

class MedicalKnowledgeBase:
    def __init__(self):
        self.rules = []
        self.facts = []
        self.load_rules()
    
    def load_rules(self):
        """Load all knowledge base rules"""
        
        # Rule 1: Critical emergencies get highest priority
        self.rules.append({
            "id": "R001",
            "condition": lambda e, env: e.get("urgency") == "critical",
            "action": "assign_highest_priority",
            "priority_bonus": 10,
            "reason": "Life-threatening situation requires immediate response"
        })
        
        # Rule 2: Check tight time windows
        self.rules.append({
            "id": "R002",
            "condition": lambda e, env: e.get("time_window", 30) < 10,
            "action": "expedite_delivery",
            "priority_bonus": 5,
            "reason": "Tight delivery window - expedite required"
        })
        
        # Rule 3: Blood supplies are always critical
        self.rules.append({
            "id": "R003",
            "condition": lambda e, env: e.get("supply") == "blood",
            "action": "prioritize_delivery",
            "priority_bonus": 8,
            "reason": "Blood supplies have short shelf life"
        })
        
        # Rule 4: Oxygen supplies need special handling
        self.rules.append({
            "id": "R004",
            "condition": lambda e, env: e.get("supply") == "oxygen",
            "action": "handle_with_care",
            "priority_bonus": 6,
            "reason": "Oxygen is critical for respiratory patients"
        })
        
        # Rule 5: Avoid heavy traffic for critical deliveries
        self.rules.append({
            "id": "R005",
            "condition": lambda e, env: e.get("urgency") == "critical" and env and env.check_traffic_ahead(e.get("hospital", (0,0))),
            "action": "recalculate_route",
            "priority_bonus": 3,
            "reason": "Critical supply cannot be delayed by traffic"
        })
        
        # Rule 6: Multiple deliveries in same area
        self.rules.append({
            "id": "R006",
            "condition": lambda e, env: e.get("batch", False),
            "action": "optimize_route",
            "priority_bonus": 2,
            "reason": "Batch deliveries can be optimized"
        })
        
        # Rule 7: Moderate urgency with enough time
        self.rules.append({
            "id": "R007",
            "condition": lambda e, env: e.get("urgency") == "moderate" and e.get("time_window", 30) > 20,
            "action": "standard_delivery",
            "priority_bonus": 0,
            "reason": "Sufficient time for standard delivery"
        })
        
        # Rule 8: Low priority supplies
        self.rules.append({
            "id": "R008",
            "condition": lambda e, env: e.get("urgency") == "low",
            "action": "defer_delivery",
            "priority_bonus": -2,
            "reason": "Low priority can be deferred if needed"
        })
    
    def infer_priority(self, emergency, env=None):
        """
        Apply all rules to determine priority score and actions
        Returns: (priority_score, applied_rules_list)
        """
        priority_score = 0
        applied_rules = []
        
        for rule in self.rules:
            try:
                # Check if rule condition matches
                if rule["condition"](emergency, env):
                    priority_score += rule["priority_bonus"]
                    applied_rules.append({
                        "rule_id": rule["id"],
                        "action": rule["action"],
                        "reason": rule["reason"],
                        "bonus": rule["priority_bonus"]
                    })
            except Exception as e:
                # Skip rules that might have environment dependencies
                pass
        
        # Ensure priority score doesn't go negative
        priority_score = max(0, priority_score)
        
        # Add base priority for urgency level
        urgency_base = {"critical": 10, "moderate": 5, "low": 1}
        priority_score += urgency_base.get(emergency.get("urgency", "low"), 1)
        
        return priority_score, applied_rules
    
    def explain_reasoning(self, emergency, env=None):
        """Generate human-readable explanation of reasoning"""
        score, rules = self.infer_priority(emergency, env)
        
        explanation = "\n=== KNOWLEDGE BASE REASONING ===\n"
        explanation += f"Emergency: {emergency.get('urgency', 'unknown').upper()}\n"
        explanation += f"Supply: {emergency.get('supply', 'unknown')}\n"
        explanation += f"Time window: {emergency.get('time_window', 'N/A')} minutes\n\n"
        
        explanation += "Applied Rules:\n"
        for rule in rules:
            explanation += f"  • {rule['rule_id']}: {rule['reason']} (+{rule['bonus']})\n"
        
        explanation += f"\nFinal Priority Score: {score}\n"
        
        if score >= 15:
            explanation += "Decision: IMMEDIATE DISPATCH REQUIRED\n"
        elif score >= 8:
            explanation += "Decision: HIGH PRIORITY - Dispatch soon\n"
        elif score >= 3:
            explanation += "Decision: NORMAL PRIORITY - Schedule delivery\n"
        else:
            explanation += "Decision: LOW PRIORITY - Can be batched\n"
        
        explanation += "=" * 35 + "\n"
        return explanation
    
    def add_fact(self, fact):
        """Add a fact to the knowledge base"""
        self.facts.append(fact)
    
    def query(self, fact_pattern):
        """Simple query to check if a fact exists"""
        return any(fact_pattern in str(fact) for fact in self.facts)


if __name__ == "__main__":
    # Test the knowledge base
    kb = MedicalKnowledgeBase()
    
    test_emergency = {
        "urgency": "critical",
        "supply": "blood",
        "time_window": 8,
        "hospital": (5, 5)
    }
    
    print(kb.explain_reasoning(test_emergency))