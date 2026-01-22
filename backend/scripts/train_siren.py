
import numpy as np
import json
import os
import time

# --- 1. Minimal NumPy SIREN Implementation ---

class SineLayer:
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first
        
        # Initialize weights
        limit = np.sqrt(6 / in_features) / omega_0
        if is_first:
            limit = 1 / in_features
        
        self.weight = np.random.uniform(-limit, limit, (in_features, out_features))
        self.bias = np.zeros(out_features) if bias else None
        
        # Cache for backprop
        self.input = None
        self.linear_out = None
    
    def forward(self, x):
        self.input = x
        self.linear_out = np.dot(x, self.weight)
        if self.bias is not None:
            self.linear_out += self.bias
        return np.sin(self.omega_0 * self.linear_out)

    def backward(self, grad_output, learning_rate):
        # grad_output is dL/d(activation)
        # Activation is sin(omega * linear)
        # d(sin)/d(linear) = omega * cos(omega * linear)
        
        d_activation = self.omega_0 * np.cos(self.omega_0 * self.linear_out)
        d_linear = grad_output * d_activation # Element-wise
        
        # d_linear/d_weight = input.T
        grad_weight = np.dot(self.input.T, d_linear)
        
        # d_linear/d_bias = sum(d_linear, axis=0)
        grad_bias = np.sum(d_linear, axis=0) if self.bias is not None else None
        
        # d_linear/d_input = weight.T
        grad_input = np.dot(d_linear, self.weight.T)
        
        # Update parameters
        self.weight -= learning_rate * grad_weight
        if self.bias is not None:
            self.bias -= learning_rate * grad_bias
            
        return grad_input

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        limit = np.sqrt(6 / in_features) / 30
        self.weight = np.random.uniform(-limit, limit, (in_features, out_features))
        self.bias = np.zeros(out_features)
        
        self.input = None
        
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, grad_output, learning_rate):
        grad_weight = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weight.T)
        
        self.weight -= learning_rate * grad_weight
        self.bias -= learning_rate * grad_bias
        
        return grad_input

class SirenSDF:
    def __init__(self, in_features=3, hidden_features=64, hidden_layers=3, out_features=1):
        self.layers = []
        # First Layer
        self.layers.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=30))
        # Hidden Layers
        for _ in range(hidden_layers):
            self.layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=30))
        # Output Layer
        self.layers.append(LinearLayer(hidden_features, out_features))
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, grad_output, learning_rate):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

# --- 2. Training Data ---
# --- 2. Training Data ---
import argparse

def get_batch(batch_size=4096, bounds=None):
    if bounds:
        min_b = np.array(bounds['min'])
        max_b = np.array(bounds['max'])
        return np.random.uniform(min_b, max_b, (batch_size, 3))
    else:
        return np.random.uniform(-1, 1, (batch_size, 3))

def sd_sphere(coords, radius=0.5):
    return np.linalg.norm(coords, axis=1, keepdims=True) - radius

def sd_box(coords, size=0.5):
    # size is half-extent (like radius)
    q = np.abs(coords) - size
    outer = np.linalg.norm(np.maximum(q, 0.0), axis=1, keepdims=True)
    inner_max = np.maximum(q[:, 0], np.maximum(q[:, 1], q[:, 2]))
    inner = np.minimum(inner_max, 0.0).reshape(-1, 1)
    return outer + inner

# --- 3. Training Loop ---
def train(shape_type='sphere', region=None):
    model = SirenSDF(hidden_features=32, hidden_layers=3)
    lr = 1e-4
    
    # Calculate transform (Center & Scale) to map region to [-1, 1]
    center = np.zeros(3)
    scale = 1.0
    
    if region:
        min_b = np.array(region['min'])
        max_b = np.array(region['max'])
        center = (min_b + max_b) / 2.0
        # Uniform scale (max dimension)
        scale = np.max(max_b - min_b) / 2.0
        # Avoid division by zero
        if scale < 1e-6: scale = 1.0

    print(f"Training NumPy SIREN on {shape_type.upper()}...")
    if region:
        print(f"Region: {region} | Center: {center} | Scale: {scale}")

    start_time = time.time()
    
    for step in range(1001):
        # Sample points in world space
        coords_world = get_batch(batch_size=4096, bounds=region)
        
        # Calculate Ground Truth in world space
        if shape_type == 'box':
            gt = sd_box(coords_world, size=0.4)
        else:
            gt = sd_sphere(coords_world, radius=0.5)
        
        # Normalize inputs for Network: (x - center) / scale -> [-1, 1]
        coords_input = (coords_world - center) / scale
        
        # Forward
        pred = model.forward(coords_input)
        
        # Loss (MSE) - Model learns World Distance directly
        # (Note: Gradients will be non-unit, but that's fine for simple viz)
        diff = pred - gt
        loss = np.mean(diff**2)
        
        # Backward
        grad_loss = 2 * diff / coords_world.shape[0]
        model.backward(grad_loss, lr)
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")
            
    print(f"Training Complete in {time.time() - start_time:.2f}s")
    
    return model, {"center": center.tolist(), "scale": float(scale)}

# --- 4. Export ---
def export_weights(model, filepath="siren_weights.json"):
    # ... existing export ...
    pass 

# --- 5. API Entry Point ---
def train_from_design(design_content, region=None):
    """
    Train SIREN network from design JSON (API entry point).
    """
    geometry = design_content.get('geometry', 'box')
    
    shape_map = {
        'box': 'box',
        'sphere': 'sphere',
        'cylinder': 'cylinder'
    }
    shape_type = shape_map.get(geometry, 'box')
    
    # Train
    model, transform = train(shape_type, region)
    
    # Serialize Weights
    layers_data = []
    
    for layer in model.layers:
        is_sine = isinstance(layer, SineLayer)
        
        weight = layer.weight.flatten().tolist()
        bias = layer.bias.flatten().tolist()
        
        layers_data.append({
            "weight": weight,
            "bias": bias,
            "is_sine": is_sine,
            "in": layer.in_features,
            "out": layer.out_features
        })
    
    # Append transform to first layer or return separately?
    # Our API returns { weights, metadata }. We can add transform to metadata.
    # The calling function in main.py constructs the response using 'weights' return only?
    # Let's check main.py usage.
    
    # Actually train_from_design returns `layers_data`.
    # I should pack transform into layers_data or return tuple?
    # Modify main.py to handle tuple return OR pack it.
    
    # Let's cheat and attach it to the list object or first layer? 
    # Cleaner: Return tuple (weights, transform) and update main.py.
    
    return layers_data, transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', type=str, default='sphere', choices=['sphere', 'box'])
    args = parser.parse_args()

    model, _ = train(args.shape)
    export_path = os.path.join(os.getcwd(), "backend/scripts/siren_sphere_weights.json")
    # ... needs update if we want to use export_weights ...


