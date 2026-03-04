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

# Optional PyTorch - all FNO classes defined inside this block
try:
    import torch
    import torch.nn as nn
    import torch.fft as fft
    HAS_TORCH = True
    
    class FourierLayer(nn.Module):
        """Fourier integral operator layer (Li et al. 2021)."""
        
        def __init__(self, width: int, modes: int):
            super().__init__()
            self.width = width
            self.modes = modes
            
            # Complex weights for Fourier modes
            self.weights = nn.Parameter(
                torch.randn(modes, width, width, 2) * 0.02
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply Fourier integral operator."""
            # x: (batch, channels, n)
            B, C, N = x.shape
            
            # FFT to frequency domain
            x_ft = fft.rfft(x, dim=-1)  # (batch, channels, n//2+1)
            
            # Multiply relevant Fourier modes by complex weights
            out_ft = torch.zeros_like(x_ft)
            
            # Multiply modes 1 to self.modes
            for mode in range(min(self.modes, x_ft.shape[-1])):
                # Weight matrix
                w = torch.view_as_complex(self.weights[mode])  # (width, width)
                # Apply to each batch
                for b in range(B):
                    out_ft[b, :, mode] = torch.matmul(w, x_ft[b, :, mode])
            
            # Inverse FFT back to spatial domain
            x = fft.irfft(out_ft, n=N, dim=-1)
            return x
    
    
    class FluidFNO(nn.Module):
        """Fourier Neural Operator for fluid drag prediction.
        
        Architecture: 
            Lift -> Fourier Layers -> Project
            
        Input:  (batch, 4) - [Re, shape_type_encoded, AR, porosity]
        Output: (batch, 1) - [Cd]
        """
        
        def __init__(
            self,
            modes: int = 12,
            width: int = 32,
            n_layers: int = 4,
            input_dim: int = 4,
            output_dim: int = 1
        ):
            super().__init__()
            
            self.modes = modes
            self.width = width
            self.n_layers = n_layers
            
            # Lift: input_dim -> width
            self.lift = nn.Linear(input_dim, width)
            
            # Fourier layers
            self.fourier_layers = nn.ModuleList([
                FourierLayer(width, modes) for _ in range(n_layers)
            ])
            
            # W layers (local linear transformation)
            self.w_layers = nn.ModuleList([
                nn.Conv1d(width, width, 1) for _ in range(n_layers)
            ])
            
            # Activation
            self.activation = nn.GELU()
            
            # Project: width -> output_dim
            self.project = nn.Sequential(
                nn.Linear(width, width // 2),
                nn.GELU(),
                nn.Linear(width // 2, output_dim)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through FNO.
            
            Args:
                x: Input parameters (batch, input_dim)
            
            Returns:
                Cd prediction (batch, output_dim)
            """
            # Lift to higher dimensional channel space
            x = self.lift(x)  # (batch, width)
            
            # Add dummy spatial dimension for Fourier layers
            # Since we have parameter inputs not spatial fields,
            # we treat each parameter as a "spatial" point
            x = x.unsqueeze(-1)  # (batch, width, 1)
            x = x.repeat(1, 1, 4)  # (batch, width, 4) - small spatial dim
            
            # Fourier layers with residual connections
            for fourier, w in zip(self.fourier_layers, self.w_layers):
                x1 = fourier(x)
                x2 = w(x)
                x = self.activation(x1 + x2)
            
            # Global average pooling over spatial dimension
            x = x.mean(dim=-1)  # (batch, width)
            
            # Project to output
            x = self.project(x)  # (batch, output_dim)
            
            return x
    
    
    class FNOTrainer:
        """Training pipeline for FluidFNO on OpenFOAM data."""
        
        def __init__(
            self,
            model: FluidFNO,
            device: str = "auto",
            learning_rate: float = 1e-3
        ):
            self.model = model
            self.device = self._get_device(device)
            self.model.to(self.device)
            
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate
            )
            self.criterion = nn.MSELoss()
        
        def _get_device(self, device: str) -> torch.device:
            """Get torch device."""
            if device == "auto":
                return torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            return torch.device(device)
        
        def train_epoch(
            self,
            train_loader: torch.utils.data.DataLoader
        ) -> float:
            """Train for one epoch."""
            self.model.train()
            total_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            return total_loss / len(train_loader)
        
        def validate(
            self,
            val_loader: torch.utils.data.DataLoader
        ) -> Tuple[float, Dict[str, float]]:
            """Validate model."""
            self.model.eval()
            total_loss = 0.0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    predictions.extend(outputs.cpu().numpy())
                    actuals.extend(targets.cpu().numpy())
            
            # Calculate metrics
            predictions = np.array(predictions).flatten()
            actuals = np.array(actuals).flatten()
            
            mse = np.mean((predictions - actuals) ** 2)
            mae = np.mean(np.abs(predictions - actuals))
            r2 = 1 - np.sum((predictions - actuals) ** 2) / np.sum(
                (actuals - np.mean(actuals)) ** 2
            )
            
            metrics = {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "mean_error": np.mean(predictions - actuals)
            }
            
            return total_loss / len(val_loader), metrics
        
        def save_checkpoint(
            self,
            path: str,
            epoch: int,
            metrics: Dict[str, float]
        ):
            """Save model checkpoint."""
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics
            }, path)
            logger.info(f"Checkpoint saved: {path}")
        
        def load_checkpoint(self, path: str) -> Dict[str, Any]:
            """Load model checkpoint."""
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(f"Checkpoint loaded: {path}")
            return checkpoint
    
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available - FNO disabled")
    
    # Create dummy classes for type hints
    class FourierLayer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available")
    
    class FluidFNO:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available")
    
    class FNOTrainer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available")
