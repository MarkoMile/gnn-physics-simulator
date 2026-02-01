"""
Loss functions for GNN physics simulation.

This module implements various loss functions for training
the GNN simulator.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """
    Mean Squared Error loss for acceleration prediction.
    """
    
    def __init__(self):
        super().__init__()
        # TODO: Implement MSE loss
    
    def forward(self, pred, target):
        """
        Compute MSE loss.
        
        Args:
            pred: Predicted accelerations
            target: Ground truth accelerations
            
        Returns:
            Loss value
        """
        # TODO: Implement forward pass
        raise NotImplementedError("MSELoss forward not implemented yet")


class RolloutLoss(nn.Module):
    """
    Loss for multi-step rollout predictions.
    
    Penalizes accumulated error over multiple time steps.
    """
    
    def __init__(self, rollout_steps: int = 10, discount: float = 0.9):
        """
        Initialize rollout loss.
        
        Args:
            rollout_steps: Number of rollout steps
            discount: Discount factor for future steps
        """
        super().__init__()
        # TODO: Implement rollout loss
        raise NotImplementedError("RolloutLoss not implemented yet")
    
    def forward(self, model, initial_state, target_trajectory):
        """
        Compute rollout loss.
        
        Args:
            model: GNN simulator
            initial_state: Initial particle state
            target_trajectory: Ground truth trajectory
            
        Returns:
            Rollout loss value
        """
        # TODO: Implement forward pass
        raise NotImplementedError("RolloutLoss forward not implemented yet")


class EnergyConservationLoss(nn.Module):
    """
    Loss penalizing energy non-conservation.
    
    Encourages the model to respect energy conservation.
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred_state, initial_state, masses):
        """
        Compute energy conservation loss.
        
        Args:
            pred_state: Predicted state (positions, velocities)
            initial_state: Initial state
            masses: Particle masses
            
        Returns:
            Energy conservation loss
        """
        # TODO: Implement energy conservation loss
        raise NotImplementedError("EnergyConservationLoss forward not implemented yet")
