"""
Machine Learning module - Predicts delivery success/failure
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class DeliveryTimePredictor:
    """Predict if a delivery will be on time or delayed"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=20, random_state=42)
        self.is_trained = False
    
    def generate_training_data(self, num_samples=500):
        """Generate synthetic training data for delivery prediction"""
        np.random.seed(42)
        
        X = []  # Features
        y = []  # Labels: 1 = on time, 0 = delayed
        
        for _ in range(num_samples):
            # Features:
            # [distance, urgency_level, traffic_level, time_window, supply_type]
            
            distance = np.random.randint(1, 30)
            
            urgency_level = np.random.choice([3, 2, 1], p=[0.3, 0.4, 0.3])
            
            traffic_level = np.random.randint(0, 11)
            
            time_window = np.random.randint(5, 40)
            
            supply_type = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])  # 0=blood, 1=medicine, 2=equipment
            
            X.append([distance, urgency_level, traffic_level, time_window, supply_type])
            
            # Calculate if delivery will be on time
            # Base travel time = distance + traffic_level * 1.5
            estimated_time = distance + (traffic_level * 1.5)
            
            # Urgency bonus: critical (3) gets 20% time reduction
            if urgency_level == 3:
                estimated_time *= 0.8
            
            # Blood supplies (0) need faster delivery
            if supply_type == 0:
                estimated_time *= 0.9
            
            # On time if estimated_time <= time_window
            y.append(1 if estimated_time <= time_window else 0)
        
        return np.array(X), np.array(y)
    
    def train(self, verbose=True):
        """Train the machine learning model"""
        X, y = self.generate_training_data(800)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if verbose:
            print(f"\n=== ML Model Training Results ===")
            print(f"Model: Random Forest Classifier")
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Delayed', 'On Time']))
            print("=" * 35 + "\n")
        
        return accuracy
    
    def predict(self, distance, urgency, traffic_level, time_window, supply_type="medicine"):
        """
        Predict if delivery will be on time
        urgency: 'critical', 'moderate', or 'low'
        supply_type: 'blood', 'medicine', or 'equipment'
        """
        if not self.is_trained:
            self.train(verbose=False)
        
        # Convert urgency to numeric
        urgency_map = {"critical": 3, "moderate": 2, "low": 1}
        urgency_num = urgency_map.get(urgency, 2)
        
        # Convert supply to numeric
        supply_map = {"blood": 0, "medicine": 1, "equipment": 2}
        supply_num = supply_map.get(supply_type, 1)
        
        features = np.array([[distance, urgency_num, traffic_level, time_window, supply_num]])
        
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        result = {
            "status": "On Time" if prediction == 1 else "Delayed",
            "confidence": max(probability) * 100,
            "probability_on_time": probability[1] * 100,
            "probability_delayed": probability[0] * 100
        }
        
        return result
    
    def predict_batch(self, deliveries):
        """Predict for multiple deliveries at once"""
        results = []
        for delivery in deliveries:
            result = self.predict(
                delivery["distance"],
                delivery["urgency"],
                delivery["traffic_level"],
                delivery["time_window"],
                delivery.get("supply_type", "medicine")
            )
            results.append(result)
        return results


class EmergencyClassifier:
    """Bonus ML: Classify emergency severity based on symptoms"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=15, random_state=42)
        self.is_trained = False
    
    def generate_training_data(self, num_samples=400):
        """Generate synthetic patient symptom data"""
        X = []
        y = []  # 2 = critical, 1 = moderate, 0 = low
        
        for _ in range(num_samples):
            # Features: [heart_rate, blood_pressure, breathing_rate, oxygen_level, pain_level]
            heart_rate = np.random.randint(50, 140)
            bp_systolic = np.random.randint(80, 200)
            breathing_rate = np.random.randint(10, 40)
            oxygen_level = np.random.randint(70, 100)
            pain_level = np.random.randint(0, 11)
            
            X.append([heart_rate, bp_systolic, breathing_rate, oxygen_level, pain_level])
            
            # Determine severity based on symptoms
            severity = 0  # low
            if heart_rate > 120 or heart_rate < 50:
                severity += 1
            if bp_systolic > 160 or bp_systolic < 90:
                severity += 1
            if breathing_rate > 25 or breathing_rate < 12:
                severity += 1
            if oxygen_level < 90:
                severity += 2
            if pain_level > 7:
                severity += 1
            
            if severity >= 3:
                y.append(2)  # critical
            elif severity >= 1:
                y.append(1)  # moderate
            else:
                y.append(0)  # low
        
        return np.array(X), np.array(y)
    
    def train(self, verbose=True):
        X, y = self.generate_training_data(500)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        if verbose:
            accuracy = self.model.score(X_test, y_test)
            print(f"Emergency Classifier Accuracy: {accuracy:.2%}")
        
        return self.model
    
    def classify_emergency(self, heart_rate, bp_systolic, breathing_rate, oxygen_level, pain_level):
        """Classify emergency severity from vital signs"""
        if not self.is_trained:
            self.train(verbose=False)
        
        features = np.array([[heart_rate, bp_systolic, breathing_rate, oxygen_level, pain_level]])
        prediction = self.model.predict(features)[0]
        
        severity_map = {2: "critical", 1: "moderate", 0: "low"}
        return severity_map[prediction]


if __name__ == "__main__":
    # Test ML model
    predictor = DeliveryTimePredictor()
    predictor.train()
    
    # Test prediction
    result = predictor.predict(distance=12, urgency="critical", traffic_level=3, time_window=15)
    print(f"Prediction: {result}")