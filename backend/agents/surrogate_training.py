"""
Physics-Informed Neural Operator Training Pipeline

Generates synthetic training data from analytical solutions and trains
the Fourier Neural Operator for fast stress prediction.

Training Data:
- 1000+ synthetic cantilever beam simulations
- Varying: geometry (L, W, H), material (E, ν), loading (P, M)
- Labels: Stress field (σxx, σyy, σzz, σxy, σyz, σzx)

Author: BRICK OS Team
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available - surrogate training disabled")


class SyntheticBeamDataset(Dataset):
    """
    Generate synthetic training data for cantilever beams
    
    Uses analytical beam theory to generate stress fields quickly
    without running expensive FEA for each sample.
    """
    
    def __init__(self, n_samples: int = 1000, n_points: int = 100):
        """
        Initialize synthetic dataset
        
        Args:
            n_samples: Number of synthetic samples to generate
            n_points: Number of spatial points per sample
        """
        self.n_samples = n_samples
        self.n_points = n_points
        self.samples = []
        
        logger.info(f"Generating {n_samples} synthetic beam samples...")
        self._generate_data()
        
    def _generate_data(self):
        """Generate synthetic stress fields using analytical solutions"""
        np.random.seed(42)  # Reproducibility
        
        for i in range(self.n_samples):
            # Random beam parameters (within reasonable bounds)
            length = np.random.uniform(0.5, 5.0)        # m
            width = np.random.uniform(0.01, 0.5)        # m
            height = np.random.uniform(0.01, 0.5)       # m
            E = np.random.uniform(10e9, 300e9)          # Pa
            nu = np.random.uniform(0.25, 0.35)          # -
            tip_load = np.random.uniform(10, 10000)     # N
            
            # Input features: [E, nu, rho, load] at each point
            # Plus spatial coordinates [x, y, z]
            x = np.linspace(0, length, self.n_points)
            y = np.zeros(self.n_points)  # Neutral axis
            z = np.zeros(self.n_points)
            
            # Material properties (constant across beam)
            E_vec = np.full(self.n_points, E)
            nu_vec = np.full(self.n_points, nu)
            load_vec = np.full(self.n_points, tip_load)
            rho = 2700  # kg/m³ (typical density)
            rho_vec = np.full(self.n_points, rho)
            
            # Input: [n_points, 7] -> [E, nu, rho, load, x, y, z]
            input_features = np.column_stack([
                E_vec, nu_vec, rho_vec, load_vec, x, y, z
            ])
            
            # Analytical stress field (Euler-Bernoulli beam theory)
            # σxx = M*y/I, where M = P*(L-x) for cantilever with tip load
            I = width * height ** 3 / 12
            y_surface = height / 2  # Maximum stress at surface
            
            # Bending moment varies linearly: M(x) = P * (L - x)
            M = tip_load * (length - x)
            
            # Bending stress: σxx = M*y/I
            sigma_xx = M * y_surface / I
            
            # Other stress components (simplified, approximately zero for slender beams)
            sigma_yy = np.zeros(self.n_points)
            sigma_zz = np.zeros(self.n_points)
            sigma_xy = np.zeros(self.n_points)  # Would be non-zero for Timoshenko
            sigma_yz = np.zeros(self.n_points)
            sigma_zx = np.zeros(self.n_points)
            
            # Output: [n_points, 6] -> [σxx, σyy, σzz, σxy, σyz, σzx]
            stress_field = np.column_stack([
                sigma_xx, sigma_yy, sigma_zz,
                sigma_xy, sigma_yz, sigma_zx
            ])
            
            self.samples.append({
                'input': input_features.astype(np.float32),
                'output': stress_field.astype(np.float32),
                'metadata': {
                    'length': length,
                    'width': width,
                    'height': height,
                    'E': E,
                    'nu': nu,
                    'tip_load': tip_load,
                    'I': I
                }
            })
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Generated {i+1}/{self.n_samples} samples")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input': torch.from_numpy(sample['input']),
            'output': torch.from_numpy(sample['output'])
        }


class FNOTrainer:
    """
    Trainer for Physics-Informed Neural Operator
    
    Handles:
    - Data loading
    - Model training
    - Validation
    - Checkpointing
    """
    
    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        n_epochs: int = 100,
        device: str = 'auto'
    ):
        """
        Initialize trainer
        
        Args:
            model: Neural operator model (e.g., PhysicsInformedNeuralOperator)
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            device: 'cuda', 'cpu', or 'auto'
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for training")
        
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"FNO Trainer initialized on {self.device}")
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            x = batch['input'].to(self.device)      # [batch, n_points, in_channels]
            y_true = batch['output'].to(self.device) # [batch, n_points, out_channels]
            
            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            
            # Data loss (MSE)
            data_loss = nn.MSELoss()(y_pred, y_true)
            
            # Physics loss (simplified - would need coordinates for full implementation)
            physics_loss = self._compute_physics_loss(y_pred, x)
            
            # Combined loss
            loss = data_loss + 0.1 * physics_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def _compute_physics_loss(self, pred_stress, input_features):
        """
        Compute simplified physics-informed loss
        
        For full implementation, would compute:
        - Equilibrium: ∇·σ = 0
        - Positivity: Von Mises > 0
        """
        # Simplified: ensure stress is reasonable
        # Von Mises stress should be positive
        sxx, syy, szz = pred_stress[..., 0], pred_stress[..., 1], pred_stress[..., 2]
        sxy, syz, szx = pred_stress[..., 3], pred_stress[..., 4], pred_stress[..., 5]
        
        vm_stress = torch.sqrt(0.5 * (
            (sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 +
            6 * (sxy**2 + syz**2 + szx**2)
        ) + 1e-8)
        
        # Penalty for negative Von Mises (unphysical)
        positivity_loss = torch.mean(torch.relu(-vm_stress))
        
        return positivity_loss
    
    def validate(self, dataloader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['input'].to(self.device)
                y_true = batch['output'].to(self.device)
                
                y_pred = self.model(x)
                loss = nn.MSELoss()(y_pred, y_true)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def train(
        self,
        train_dataset,
        val_dataset=None,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Full training loop
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            save_path: Path to save best model checkpoint
            
        Returns:
            Training history dict
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Single process for stability
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        best_val_loss = float('inf')
        
        logger.info(f"Starting training for {self.n_epochs} epochs...")
        
        for epoch in range(self.n_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    self.save_checkpoint(save_path, epoch, best_val_loss)
            
            # Logging
            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{self.n_epochs} - Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                logger.info(msg)
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'n_epochs': self.n_epochs,
            'final_lr': self.optimizer.param_groups[0]['lr']
        }
        
        return history
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")


def train_surrogate_model(
    n_samples: int = 1000,
    n_epochs: int = 100,
    save_dir: str = "./models"
) -> Dict:
    """
    Convenience function to train the surrogate model
    
    Args:
        n_samples: Number of synthetic training samples
        n_epochs: Training epochs
        save_dir: Directory to save model
        
    Returns:
        Training history
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for training")
    
    from backend.agents.structural_agent import PhysicsInformedNeuralOperator
    
    # Create model
    model = PhysicsInformedNeuralOperator(
        in_channels=7,   # [E, nu, rho, load, x, y, z]
        out_channels=6,  # [σxx, σyy, σzz, σxy, σyz, σzx]
        modes=12,
        width=64,
        n_layers=4
    )
    
    # Create datasets
    dataset = SyntheticBeamDataset(n_samples=n_samples)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create trainer
    trainer = FNOTrainer(
        model=model,
        learning_rate=1e-3,
        batch_size=16,
        n_epochs=n_epochs
    )
    
    # Train
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / "fno_surrogate_best.pt"
    
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_path=str(save_path)
    )
    
    # Mark model as trained
    model.is_trained = True
    
    logger.info("Surrogate model training complete!")
    logger.info(f"Final train loss: {history['train_losses'][-1]:.6f}")
    if history['val_losses']:
        logger.info(f"Final val loss: {history['val_losses'][-1]:.6f}")
    
    return history


if __name__ == "__main__":
    # Run training when executed directly
    logging.basicConfig(level=logging.INFO)
    
    try:
        history = train_surrogate_model(
            n_samples=1000,
            n_epochs=100,
            save_dir="./models"
        )
        print("Training complete!")
        print(f"Final train loss: {history['train_losses'][-1]:.6f}")
    except Exception as e:
        print(f"Training failed: {e}")
