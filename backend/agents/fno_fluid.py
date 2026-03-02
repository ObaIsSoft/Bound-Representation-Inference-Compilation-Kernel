"""
FNO Fluid - Fourier Neural Operator for CFD

EXPERIMENTAL: Requires training on OpenFOAM data.

This module provides:
1. FluidFNO - Neural operator architecture
2. Training pipeline for OpenFOAM data
3. Inference for fast predictions (once trained)

Training Requirements:
- 1000+ OpenFOAM simulations for basic accuracy
- 10,000+ for production use
- Validation against wind tunnel data

Reference:
- Li et al. (2021) "Fourier Neural Operator for Parametric PDEs"

Author: BRICK OS Team
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import json

logger = logging.getLogger(__name__)

# Optional PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.fft as fft
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available - FNO disabled")


class FourierLayer(nn.Module):
    """Fourier integral operator layer (Li et al. 2021)."""
    
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.width = width
        self.modes = modes
        
        # Complex weights for Fourier modes
        self.weights = nn.Parameter(
            torch.randn(modes, modes, width, width, dtype=torch.cfloat) * 0.02
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFT, multiply weights, inverse FFT."""
        batch_size, width, nx, ny = x.shape
        
        # FFT
        x_ft = fft.rfft2(x, dim=(-2, -1))
        
        # Initialize output
        out_ft = torch.zeros_like(x_ft)
        
        # Multiply relevant Fourier modes
        modes_x = min(self.modes, x_ft.shape[-2])
        modes_y = min(self.modes, x_ft.shape[-1])
        
        for i in range(modes_x):
            for j in range(modes_y):
                out_ft[:, :, i, j] = torch.einsum(
                    'bi,io->bo',
                    x_ft[:, :, i, j],
                    self.weights[i, j]
                )
        
        # Inverse FFT
        x = fft.irfft2(out_ft, s=(nx, ny), dim=(-2, -1))
        return x


class FluidFNO(nn.Module):
    """
    Fourier Neural Operator for fluid dynamics.
    
    Architecture:
        Input [3, nx, ny] -> Lift -> 4x Fourier Layers -> Project -> Output [3, nx, ny]
    
    Input channels:
        0: Geometry mask (1=fluid, 0=obstacle)
        1: Reynolds number (normalized)
        2: Mach number (normalized)
    
    Output channels:
        0: u (x-velocity)
        1: v (y-velocity)
        2: p (pressure)
    """
    
    def __init__(
        self,
        width: int = 64,
        modes: int = 12,
        layers: int = 4,
        input_channels: int = 3,
        output_channels: int = 3
    ):
        super().__init__()
        
        self.width = width
        self.modes = modes
        self.layers = layers
        
        # Lift: input -> width
        self.lift = nn.Conv2d(input_channels, width, kernel_size=1)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for _ in range(layers):
            self.fourier_layers.append(FourierLayer(width, modes))
            self.w_layers.append(nn.Conv2d(width, width, kernel_size=1))
        
        # Project: width -> output
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, output_channels, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Lift
        x = self.lift(x)
        
        # Fourier layers with residual connections
        for fourier, w in zip(self.fourier_layers, self.w_layers):
            x1 = fourier(x)
            x2 = w(x)
            x = torch.relu(x1 + x2)
        
        # Project
        x = self.project(x)
        return x


class FNOTrainer:
    """
    Training pipeline for FluidFNO.
    
    Requires training data from OpenFOAM simulations.
    """
    
    def __init__(
        self,
        model: Optional[FluidFNO] = None,
        device: str = "cpu",
        learning_rate: float = 1e-3
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for training")
        
        self.device = device
        self.model = model or FluidFNO(width=64, modes=12, layers=4)
        self.model.to(device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        self.criterion = nn.MSELoss()
        
        self.training_history = []
    
    def train_epoch(
        self,
        training_data: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            training_data: List of (input, output) tuples
                input: [3, nx, ny] - geometry, Re, Mach
                output: [3, nx, ny] - u, v, p
        
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for input_np, output_np in training_data:
            # Convert to tensors
            input_tensor = torch.tensor(
                input_np, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            target = torch.tensor(
                output_np, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            prediction = self.model(input_tensor)
            loss = self.criterion(prediction, target)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(training_data)
    
    def train(
        self,
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        validation_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        epochs: int = 100,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train FNO model.
        
        Args:
            training_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of training epochs
            save_dir: Directory to save checkpoints
        
        Returns:
            Training history
        """
        logger.info(f"Starting FNO training: {epochs} epochs")
        logger.info(f"Training samples: {len(training_data)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(training_data)
            
            # Validation
            val_loss = None
            if validation_data:
                val_loss = self.validate(validation_data)
            
            # Log
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            if epoch % 10 == 0:
                log_msg = f"Epoch {epoch}/{epochs}, Train: {train_loss:.6f}"
                if val_loss:
                    log_msg += f", Val: {val_loss:.6f}"
                logger.info(log_msg)
            
            # Save checkpoint
            if save_dir and val_loss and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(Path(save_dir) / "fno_best.pt")
        
        # Save final
        if save_dir:
            self.save(Path(save_dir) / "fno_final.pt")
            # Save history
            with open(Path(save_dir) / "training_history.json", "w") as f:
                json.dump(self.training_history, f, indent=2)
        
        return {
            'epochs': epochs,
            'final_train_loss': train_loss,
            'best_val_loss': best_val_loss,
            'history': self.training_history
        }
    
    def validate(
        self,
        validation_data: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for input_np, output_np in validation_data:
                input_tensor = torch.tensor(
                    input_np, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                target = torch.tensor(
                    output_np, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                prediction = self.model(input_tensor)
                loss = self.criterion(prediction, target)
                total_loss += loss.item()
        
        return total_loss / len(validation_data)
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        logger.info(f"Loaded checkpoint from {path}")


class FNODataGenerator:
    """
    Generate training data from OpenFOAM simulations.
    
    This creates the dataset needed to train FNO:
    1. Run OpenFOAM for various geometries/Re/Mach
    2. Extract flow fields
    3. Create (input, output) pairs for training
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_parameter_space(
        self,
        n_samples: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter space for training data.
        
        Returns list of parameter dicts with:
        - shape_type
        - length
        - velocity (for Re)
        - temperature (for Mach)
        """
        np.random.seed(42)
        
        samples = []
        shapes = ["cylinder", "box", "airfoil"]
        
        for i in range(n_samples):
            shape = np.random.choice(shapes)
            
            # Log-uniform for length (0.01m to 10m)
            length = 10 ** np.random.uniform(-2, 1)
            
            # Log-uniform for velocity (0.1 to 100 m/s)
            velocity = 10 ** np.random.uniform(-1, 2)
            
            # Temperature for Mach variation
            temperature = np.random.uniform(250, 350)
            
            samples.append({
                'id': i,
                'shape_type': shape,
                'length': length,
                'velocity': velocity,
                'temperature': temperature
            })
        
        return samples
    
    def extract_from_openfoam(
        self,
        case_dir: str,
        nx: int = 64,
        ny: int = 64
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract training sample from OpenFOAM case.
        
        Returns:
            (input_field, output_field) where:
            - input: [3, nx, ny] - geometry mask, Re, Mach
            - output: [3, nx, ny] - u, v, p
        """
        # This would parse OpenFOAM output and interpolate to grid
        # For now, return None (placeholder)
        
        # TODO: Implement OpenFOAM field extraction
        # 1. Read mesh from constant/polyMesh
        # 2. Read U, p from latest time directory
        # 3. Interpolate to regular grid (nx x ny)
        # 4. Create geometry mask
        # 5. Return (input, output)
        
        logger.warning("OpenFOAM field extraction not yet implemented")
        return None
    
    def create_training_dataset(
        self,
        openfoam_cases: List[str],
        output_file: str = "training_data.npz"
    ):
        """
        Create training dataset from OpenFOAM cases.
        
        Args:
            openfoam_cases: List of OpenFOAM case directories
            output_file: Output NPZ file
        """
        inputs = []
        outputs = []
        
        for case_dir in openfoam_cases:
            result = self.extract_from_openfoam(case_dir)
            if result:
                inp, out = result
                inputs.append(inp)
                outputs.append(out)
        
        if inputs:
            np.savez(
                self.output_dir / output_file,
                inputs=np.array(inputs),
                outputs=np.array(outputs)
            )
            logger.info(f"Saved {len(inputs)} training samples")
        else:
            logger.error("No training data generated")


def train_fno_from_openfoam(
    openfoam_cases_dir: str,
    output_model_path: str,
    epochs: int = 100
) -> Dict[str, Any]:
    """
    Train FNO from OpenFOAM simulation data.
    
    This is the main training pipeline:
    1. Extract data from OpenFOAM cases
    2. Train FNO
    3. Save trained model
    
    Args:
        openfoam_cases_dir: Directory containing OpenFOAM cases
        output_model_path: Where to save trained model
        epochs: Training epochs
    
    Returns:
        Training metrics
    """
    # Generate training data
    generator = FNODataGenerator(output_dir="training_data")
    
    # Find all OpenFOAM cases
    cases = [
        str(d) for d in Path(openfoam_cases_dir).iterdir()
        if d.is_dir() and (d / "system" / "controlDict").exists()
    ]
    
    logger.info(f"Found {len(cases)} OpenFOAM cases")
    
    # Extract training data
    generator.create_training_dataset(cases)
    
    # Load training data
    data = np.load("training_data/training_data.npz")
    inputs = data['inputs']
    outputs = data['outputs']
    
    # Split train/val
    n_train = int(0.8 * len(inputs))
    train_data = list(zip(inputs[:n_train], outputs[:n_train]))
    val_data = list(zip(inputs[n_train:], outputs[n_train:]))
    
    # Train
    trainer = FNOTrainer()
    results = trainer.train(
        train_data,
        val_data,
        epochs=epochs,
        save_dir="training_data"
    )
    
    # Save final model
    trainer.save(output_model_path)
    
    return results


if __name__ == "__main__":
    # Example: Generate parameter space
    generator = FNODataGenerator("training_data")
    params = generator.generate_parameter_space(n_samples=1000)
    
    print(f"Generated {len(params)} parameter sets")
    print("Example:", params[0])
    
    # Save parameter space
    with open("training_data/parameter_space.json", "w") as f:
        json.dump(params, f, indent=2)
    
    print("\nNext steps:")
    print("1. Run OpenFOAM for each parameter set")
    print("2. Extract flow fields with extract_from_openfoam()")
    print("3. Train FNO with train_fno_from_openfoam()")
