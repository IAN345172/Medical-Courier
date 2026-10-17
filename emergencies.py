"""
Emergency module - Contains emergency request data
"""

# List of emergency requests
EMERGENCIES = [
    {
        "id": "E001",
        "hospital": (0, 3),
        "urgency": "critical",
        "supply": "blood",
        "time_window": 8,
        "description": "Trauma patient needs emergency blood transfusion"
    },
    {
        "id": "E002",
        "hospital": (6, 4),
        "urgency": "critical",
        "supply": "oxygen",
        "time_window": 10,
        "description": "Respiratory distress patient"
    },
    {
        "id": "E003",
        "hospital": (1, 6),
        "urgency": "moderate",
        "supply": "medicine",
        "time_window": 20,
        "description": "Regular medication delivery"
    },
    {
        "id": "E004",
        "hospital": (8, 7),
        "urgency": "moderate",
        "supply": "vaccine",
        "time_window": 25,
        "description": "Vaccine delivery (temperature sensitive)"
    },
    {
        "id": "E005",
        "hospital": (3, 1),
        "urgency": "low",
        "supply": "equipment",
        "time_window": 45,
        "description": "Medical equipment - non-urgent"
    }
]


def get_emergency_by_id(emergency_id):
    """Get a specific emergency by ID"""
    for e in EMERGENCIES:
        if e["id"] == emergency_id:
            return e
    return None


def get_emergencies_by_urgency(urgency_level):
    """Get all emergencies of a specific urgency"""
    return [e for e in EMERGENCIES if e["urgency"] == urgency_level]


def get_critical_emergencies():
    """Get all critical emergencies"""
    return get_emergencies_by_urgency("critical")


def get_all_emergencies():
    """Return all emergencies"""
    return EMERGENCIES


if __name__ == "__main__":
    print("=== Emergency Requests ===")
    for e in EMERGENCIES:
        print(f"{e['id']}: {e['urgency'].upper()} - {e['description']}")
        print(f"  Hospital: {e['hospital']}, Time window: {e['time_window']} min\n")