import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_surrogate():
    print("--- Training Neural Surrogate (PINN) ---")
    
    data_path = "data/training_data.csv"
    output_path = "data/physics_surrogate.pkl"
    
    if not os.path.exists(data_path):
        print("❌ Error: No training data found.")
        return

    # 1. Load Data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples.")
    
    if len(df) < 5:
        print("⚠️ Warning: Very low data volume. Model will be inaccurate.")
    
    # 2. Preprocess
    # Features X: Mass, Cost
    # Labels Y: Thrust, PhysicsSafe (Bool -> Int)
    
    # Handle missing/dirty data
    df = df.fillna(0)
    
    X = df[["geometry_mass", "cost_estimate"]].values
    
    # Convert Boolean 'physics_safe' to float 1.0/0.0
    safe_score = df["physics_safe"].astype(int).values
    thrust = df["flight_thrust_req_n"].values
    
    Y = np.column_stack((thrust, safe_score))
    
    # Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train MLP (Neural Net)
    # 2 Hidden Layers of 16 neurons
    model = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=1000, random_state=42)
    model.fit(X_scaled, Y)
    
    score = model.score(X_scaled, Y)
    print(f"Training R^2 Score: {score:.4f}")
    
    # 4. Save
    artifact = {
        "model": model,
        "scaler": scaler,
        "features": ["geometry_mass", "cost_estimate"],
        "labels": ["flight_thrust_req_n", "physics_safe"]
    }
    
    joblib.dump(artifact, output_path)
    print(f"✅ Model saved to {output_path}")

if __name__ == "__main__":
    train_surrogate()
