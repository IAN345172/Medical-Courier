"""
Web-Based Medical Emergency Courier System
Run with: streamlit run app.py
Then open http://localhost:8501 in your browser
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import time

# Import your existing modules
from environment import MedicalEnvironment
from search import bfs, dfs, astar, astar_with_priority
from knowledge import MedicalKnowledgeBase
from ml_model import DeliveryTimePredictor
from emergencies import get_all_emergencies

# Page configuration
st.set_page_config(
    page_title="Medical Emergency Courier",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .emergency-card {
        background-color: #fff3e0;
        border-left: 5px solid #ff4b4b;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .route-info {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state (persists across interactions)
if 'env' not in st.session_state:
    st.session_state.env = MedicalEnvironment(size=10)
    st.session_state.kb = MedicalKnowledgeBase()
    st.session_state.ml = DeliveryTimePredictor()
    st.session_state.ml.train(verbose=False)
    st.session_state.emergency_history = []
    st.session_state.current_response = None

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🏥 Intelligent Medical Emergency Courier System</h1>
    <p style="color: #ccc; margin: 0;">AI-Powered Autonomous Delivery for Critical Medical Supplies</p>
</div>
""", unsafe_allow_html=True)

# Create two main columns
left_col, right_col = st.columns([1, 1.5], gap="large")

# ==================== LEFT COLUMN: Emergency Input Form ====================
with left_col:
    st.markdown("## 📞 Submit Emergency Request")
    st.markdown("---")
    
    with st.form(key="emergency_form"):
        st.markdown("### 📍 Destination")
        
        # Hospital selection
        hospitals = st.session_state.env.hospitals
        hospital_options = {}
        for i, h in enumerate(hospitals):
            hospital_options[f"Hospital {i+1} at ({h[0]}, {h[1]})"] = h
        
        selected_hospital = st.selectbox(
            "Select Hospital", 
            list(hospital_options.keys()),
            help="Choose the hospital that needs medical supplies"
        )
        hospital_coords = hospital_options[selected_hospital]
        
        st.markdown("### 🚨 Emergency Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Urgency selection with visual indicators
            urgency = st.select_slider(
                "Urgency Level",
                options=["low", "moderate", "critical"],
                value="moderate",
                format_func=lambda x: {
                    "low": "🟢 Low",
                    "moderate": "🟡 Moderate", 
                    "critical": "🔴 CRITICAL"
                }[x]
            )
        
        with col2:
            # Supply type
            supply = st.selectbox(
                "Medical Supply",
                ["🩸 Blood", "💨 Oxygen", "💊 Medicine", "💉 Vaccine", "🔧 Equipment"],
                format_func=lambda x: x
            )
            # Clean supply name for processing
            supply_clean = supply.split(" ")[1] if " " in supply else supply
        
        col3, col4 = st.columns(2)
        
        with col3:
            time_window = st.number_input(
                "Time Window (minutes)",
                min_value=5,
                max_value=60,
                value=15,
                step=5,
                help="Must deliver within this many minutes"
            )
        
        with col4:
            # Optional patient condition
            patient_condition = st.text_input(
                "Patient Condition (optional)",
                placeholder="e.g., Trauma, Respiratory distress, Routine checkup",
                help="Additional context for knowledge base"
            )
        
        st.markdown("### ⚙️ Algorithm Selection")
        
        algorithm = st.radio(
            "Choose Routing Algorithm",
            ["🚀 Auto (Based on Urgency)", "🔍 BFS", "📊 A* Standard", "⚡ A* Priority"],
            help="Auto selects the best algorithm based on emergency urgency"
        )
        
        # Submit button
        submitted = st.form_submit_button("🚨 SUBMIT EMERGENCY REQUEST", use_container_width=True)
        
        if submitted:
            # Create emergency request
            emergency = {
                "id": f"REQ_{len(st.session_state.emergency_history) + 1}",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "hospital": hospital_coords,
                "urgency": urgency,
                "supply": supply_clean,
                "time_window": time_window,
                "condition": patient_condition if patient_condition else "Not specified",
                "algorithm_choice": algorithm
            }
            
            # Find route
            start = st.session_state.env.depot
            goal = hospital_coords
            
            weight_map = {"critical": 5.0, "moderate": 2.0, "low": 1.0}
            
            # Measure computation time
            start_time = time.time()
            
            # Select algorithm based on user choice
            if algorithm == "🔍 BFS":
                path = bfs(st.session_state.env, start, goal)
                algo_used = "BFS"
            elif algorithm == "📊 A* Standard":
                path = astar(st.session_state.env, start, goal)
                algo_used = "A* Standard"
            elif algorithm == "⚡ A* Priority":
                path = astar_with_priority(st.session_state.env, start, goal, weight_map[urgency])
                algo_used = "A* Priority"
            else:  # Auto
                if urgency == "critical":
                    path = astar_with_priority(st.session_state.env, start, goal, weight_map[urgency])
                    algo_used = "A* Priority (Auto-selected for critical)"
                elif urgency == "moderate":
                    path = astar(st.session_state.env, start, goal)
                    algo_used = "A* Standard (Auto-selected for moderate)"
                else:
                    path = bfs(st.session_state.env, start, goal)
                    algo_used = "BFS (Auto-selected for low urgency)"
            
            computation_time = time.time() - start_time
            
            # Get ML prediction
            if path:
                traffic_level = sum(st.session_state.env.get_traffic_cost(p) for p in path[:5])
                ml_result = st.session_state.ml.predict(
                    distance=len(path),
                    urgency=urgency,
                    traffic_level=traffic_level,
                    time_window=time_window,
                    supply_type=supply_clean
                )
            else:
                ml_result = {"status": "No route", "confidence": 0, "probability_on_time": 0, "probability_delayed": 0}
            
            # Get knowledge base reasoning
            priority_score, rules = st.session_state.kb.infer_priority(emergency, None)
            
            # Store response
            st.session_state.current_response = {
                "emergency": emergency,
                "path": path,
                "steps": len(path) - 1 if path else None,
                "computation_time": computation_time,
                "algorithm_used": algo_used,
                "ml_prediction": ml_result,
                "priority_score": priority_score,
                "rules": rules,
                "traffic_cost": traffic_level if path else None
            }
            
            # Add to history
            st.session_state.emergency_history.append(st.session_state.current_response)
            
            st.rerun()

# ==================== RIGHT COLUMN: Results Display ====================
with right_col:
    if st.session_state.current_response:
        response = st.session_state.current_response
        
        if response["path"]:
            # Success message
            st.markdown(f"""
            <div class="emergency-card">
                <h3 style="color: #ff4b4b; margin: 0;">🚨 EMERGENCY RESPONSE</h3>
                <p style="margin: 0;">Request #{response['emergency']['id']} at {response['emergency']['timestamp']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics row
            st.markdown("### 📊 Delivery Metrics")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Total Steps", response["steps"], delta="Route found")
            with col_b:
                st.metric("Computation Time", f"{response['computation_time']*1000:.1f} ms")
            with col_c:
                st.metric("Algorithm Used", response["algorithm_used"])
            with col_d:
                status_color = "🟢" if response["ml_prediction"]["status"] == "On Time" else "🔴"
                st.metric("ML Prediction", f"{status_color} {response['ml_prediction']['status']}")
            
            # Route details
            st.markdown("### 📍 Delivery Route")
            
            # Format path for display - FIXED THIS LINE
            path_display = " → ".join([f"({p[0]},{p[1]})" for p in response["path"]])
            st.code(f"Depot {st.session_state.env.depot} → Hospital {response['emergency']['hospital']}\n{path_display}", language="text")
            
            # Route visualization
            st.markdown("### 🗺️ Route Visualization")
            
            # Create the visualization
            fig, ax = plt.subplots(figsize=(8, 8))
            
            color_grid = np.zeros((st.session_state.env.size, st.session_state.env.size, 3))
            
            for i in range(st.session_state.env.size):
                for j in range(st.session_state.env.size):
                    if st.session_state.env.grid[i][j] == 1:  # Hospital
                        color_grid[i, j] = [0, 1, 0]
                    elif st.session_state.env.grid[i][j] == 2:  # Depot
                        color_grid[i, j] = [0, 0, 1]
                    elif st.session_state.env.traffic[i][j] >= 3:  # Heavy traffic
                        color_grid[i, j] = [1, 0, 0]
                    elif st.session_state.env.traffic[i][j] > 0:  # Light traffic
                        color_grid[i, j] = [1, 0.5, 0]
                    else:
                        color_grid[i, j] = [0.9, 0.9, 0.9]
            
            ax.imshow(color_grid, interpolation='nearest')
            
            # Draw the path
            if response["path"]:
                path_x = [p[1] for p in response["path"]]
                path_y = [p[0] for p in response["path"]]
                ax.plot(path_x, path_y, 'y-', linewidth=3, marker='o', markersize=8,
                        markerfacecolor='yellow', markeredgecolor='black', 
                        label='Delivery Route', zorder=5)
                ax.plot(path_x[0], path_y[0], 'bo', markersize=18, label='🚚 Depot', zorder=6)
                ax.plot(path_x[-1], path_y[-1], 'r*', markersize=22, label='🏥 Hospital', zorder=6)
            
            # Mark all hospitals
            for h in st.session_state.env.hospitals:
                if h != response["emergency"]["hospital"]:
                    ax.plot(h[1], h[0], 'g^', markersize=10, alpha=0.5)
            
            ax.set_title(f"{response['emergency']['urgency'].upper()} {response['emergency']['supply'].upper()} Delivery", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            
            # Knowledge Base Reasoning
            with st.expander("🧠 AI Reasoning (Knowledge Base)", expanded=True):
                st.markdown(f"**Priority Score:** {response['priority_score']}")
                st.markdown("**Rules Applied:**")
                if response['rules']:
                    for rule in response['rules']:
                        st.markdown(f"- {rule['reason']} (+{rule['bonus']})")
                else:
                    st.markdown("- No specific rules triggered for this urgency level")
            
            # ML Prediction Details
            with st.expander("🤖 Machine Learning Prediction", expanded=True):
                st.markdown(f"**Prediction:** {response['ml_prediction']['status']}")
                st.markdown(f"**Confidence:** {response['ml_prediction']['confidence']:.1f}%")
                st.markdown(f"**Probability On Time:** {response['ml_prediction']['probability_on_time']:.1f}%")
                st.markdown(f"**Probability Delayed:** {response['ml_prediction']['probability_delayed']:.1f}%")
                
                # Progress bar for confidence
                st.progress(response['ml_prediction']['confidence'] / 100)
            
            # Traffic information
            if response.get('traffic_cost'):
                with st.expander("🚦 Traffic Analysis"):
                    st.markdown(f"**Traffic Cost:** {response['traffic_cost']:.0f}")
                    if response['traffic_cost'] > 10:
                        st.warning("⚠️ Heavy traffic detected on route")
                    else:
                        st.success("✅ Low traffic on route")
        
        else:
            st.error("❌ NO ROUTE FOUND - Destination unreachable due to traffic obstacles!")
            st.markdown("Please try a different hospital or wait for traffic to clear.")
    
    else:
        # No emergency submitted yet
        st.info("👈 **Submit an emergency request** using the form on the left")
        st.markdown("""
        ### How to use:
        1. Select a hospital from the dropdown
        2. Choose urgency level (Critical/Moderate/Low)
        3. Select medical supply type
        4. Set time window for delivery
        5. Click "SUBMIT EMERGENCY REQUEST"
        
        The system will:
        - Find the optimal route using AI search algorithms
        - Apply medical knowledge rules
        - Predict on-time delivery using ML
        - Visualize the route on the map
        """)
        
        # Show environment map preview
        st.markdown("### 🗺️ Current Environment")
        fig, ax = plt.subplots(figsize=(8, 6))
        color_grid = np.zeros((st.session_state.env.size, st.session_state.env.size, 3))
        
        for i in range(st.session_state.env.size):
            for j in range(st.session_state.env.size):
                if st.session_state.env.grid[i][j] == 1:
                    color_grid[i, j] = [0, 1, 0]
                elif st.session_state.env.grid[i][j] == 2:
                    color_grid[i, j] = [0, 0, 1]
                elif st.session_state.env.traffic[i][j] >= 3:
                    color_grid[i, j] = [1, 0, 0]
                else:
                    color_grid[i, j] = [0.9, 0.9, 0.9]
        
        ax.imshow(color_grid, interpolation='nearest')
        ax.plot(st.session_state.env.depot[1], st.session_state.env.depot[0], 'bs', markersize=15, label='Depot')
        for h in st.session_state.env.hospitals:
            ax.plot(h[1], h[0], 'g^', markersize=12, label='Hospital' if h == st.session_state.env.hospitals[0] else "")
        
        ax.set_title("Medical Delivery Network")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

# ==================== BOTTOM SECTION: Emergency History ====================
st.markdown("---")
st.markdown("## 📜 Emergency Response History")

if st.session_state.emergency_history:
    # Create history dataframe
    history_data = []
    for idx, resp in enumerate(st.session_state.emergency_history):
        if resp["path"]:
            history_data.append({
                "Request ID": resp["emergency"]["id"],
                "Time": resp["emergency"]["timestamp"],
                "Hospital": str(resp["emergency"]["hospital"]),
                "Urgency": resp["emergency"]["urgency"].upper(),
                "Supply": resp["emergency"]["supply"],
                "Steps": resp["steps"],
                "Algorithm": resp["algorithm_used"],
                "ML Result": resp["ml_prediction"]["status"],
                "Time Window": f"{resp['emergency']['time_window']} min"
            })
    
    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.emergency_history = []
            st.session_state.current_response = None
            st.rerun()
else:
    st.caption("No emergencies submitted yet. Use the form above to submit your first request.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏥 Intelligent Medical Emergency Courier System | AI-Powered Delivery Optimization</p>
    <p>Technologies: Python • Streamlit • BFS/DFS/A* • Random Forest ML • Knowledge Base Rules</p>
</div>
""", unsafe_allow_html=True)