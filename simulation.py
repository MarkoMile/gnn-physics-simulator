from integrators import get_integrator

"""
N-body gravitational simulation.

This module implements the N-body problem for generating
training data for the GNN simulator.
"""

import numpy as np
from typing import Tuple, Optional


class NBodySimulation:
    """
    N-body gravitational simulation.
    
    Simulates N particles interacting via gravitational forces.
    """
    
    def __init__(
        self,
        num_particles: int,
        gravitational_constant: float = 1.0,
        softening_length: float = 0.01,
        integrator: str = 'rk4'
    ):
        """
        Initialize N-body simulation.
        
        Args:
            num_particles: Number of particles
            gravitational_constant: Gravitational constant G
            softening_length: Softening parameter to avoid singularities
            integrator: Integration method name
        """
        self.num_particles = num_particles
        self.gravitational_constant = gravitational_constant
        self.softening_length = softening_length
        self.integrator = get_integrator(integrator)
    def initialize_random_state(
        self,
        position_scale: float = 1.0,
        velocity_scale: float = 0.1,
        mass_range: Tuple[float, float] = (0.5, 1.5)
    ):
        """
        Initialize particles with random positions, velocities, and masses.
        
        Args:
            position_scale: Scale for initial positions
            velocity_scale: Scale for initial velocities
            mass_range: Range for random masses (min, max)
        """
        self.positions = np.random.uniform(-position_scale, position_scale, (self.num_particles, 3))
        self.velocities = np.random.uniform(-velocity_scale, velocity_scale, (self.num_particles, 3))
        self.masses = np.random.uniform(mass_range[0], mass_range[1], self.num_particles)
    
    def compute_gravitational_forces(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        Compute gravitational forces between all particles.
        
        Args:
            positions: Particle positions [N, 3]
            masses: Particle masses [N]
            
        Returns:
            Accelerations [N, 3]
        """
        # Displacement vectors between all pairs [N, N, 3]
        # r[i, j, :] is the vector from particle i to particle j
        r = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        
        # Squared distances [N, N]
        r_sq = np.sum(r**2, axis=-1)
        
        # Add softening length squared to avoid singularity (and division by zero)
        r_sq_softened = r_sq + self.softening_length**2
        
        # Inverse distance cubed [N, N]
        inv_r3 = r_sq_softened ** (-1.5)
        
        # Acceleration computation: 
        # masses array is [N], we reshape to [1, N, 1] to broadcast over interacting particles (j) and 3D coords
        # sum over axis=1 which represents the contributions of all particles j to particle i
        accelerations = self.gravitational_constant * np.sum(
            masses[np.newaxis, :, np.newaxis] * inv_r3[:, :, np.newaxis] * r,
            axis=1
        )
        
        return accelerations
    
    def simulate(
        self,
        total_time: float,
        dt: float,
        save_every: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the simulation.
        
        Args:
            total_time: Total simulation time
            dt: Time step
            save_every: Save state every N steps
            
        Returns:
            Tuple of (positions, velocities, times) arrays
        """
        if not hasattr(self, 'positions') or not hasattr(self, 'velocities') or not hasattr(self, 'masses'):
            raise ValueError("Simulation state not initialized. Call initialize_random_state() first.")
            
        total_steps = int(total_time / dt)
        
        # Arrays to store the history
        num_saved = (total_steps // save_every) + 1
        history_positions = np.zeros((num_saved, self.num_particles, 3))
        history_velocities = np.zeros((num_saved, self.num_particles, 3))
        history_times = np.zeros(num_saved)
        
        # Extract current state to iterate over
        pos = self.positions.copy()
        vel = self.velocities.copy()
        
        # to pass to the integrator taking just the positions array
        def force_fn(positions_array):
            return self.compute_gravitational_forces(positions_array, self.masses)
            
        save_idx = 0
        for step in range(total_steps + 1):
            # Save current state
            if step % save_every == 0:
                history_positions[save_idx] = pos
                history_velocities[save_idx] = vel
                history_times[save_idx] = step * dt
                save_idx += 1
                
            # Perform integration step (skip on the very last step since we just wanted to save it)
            if step < total_steps:
                pos, vel = self.integrator.step((pos, vel), dt, force_fn)
                
        # Update internal class state
        self.positions = pos
        self.velocities = vel
        
        # Truncate arrays if total_steps wasn't perfectly divisible by save_every
        history_positions = history_positions[:save_idx]
        history_velocities = history_velocities[:save_idx]
        history_times = history_times[:save_idx]
        
        return history_positions, history_velocities, history_times
    
    def compute_total_energy(self) -> float:
        """
        Compute total energy (kinetic + potential).
        
        Returns:
            Total energy of the system
        """
        if not hasattr(self, 'positions') or not hasattr(self, 'velocities') or not hasattr(self, 'masses'):
            raise ValueError("Simulation state not initialized. Call initialize_random_state() first.")
            
        # Kinetic Energy: K = 0.5 * sum(m * v^2)
        v_sq = np.sum(self.velocities**2, axis=-1)
        kinetic_energy = 0.5 * np.sum(self.masses * v_sq)
        
        # Potential Energy: U = -G * sum_{i < j} (m_i * m_j / sqrt(r_{ij}^2 + eps^2))
        r = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        r_sq = np.sum(r**2, axis=-1)
        
        # Add softening length squared to distance calculation
        r_sq_softened = r_sq + self.softening_length**2
        
        # Computes interactions matrix
        potential_matrix = -self.gravitational_constant * (
            self.masses[:, np.newaxis] * self.masses[np.newaxis, :]
        ) / np.sqrt(r_sq_softened)
        
        # We only take the upper triangle (k=1 excludes diagonal) 
        # to ensure we don't double count pairs and avoid self-interaction
        potential_energy = np.sum(np.triu(potential_matrix, k=1))
        
        return float(kinetic_energy + potential_energy)


"""
Dataset generation utilities.

This module provides functions for generating and saving
simulation datasets for training the GNN.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any


def generate_dataset(
    num_trajectories: int,
    num_particles: int,
    total_time: float,
    dt: float,
    integrator: str = 'rk4',
    save_path: str = None,
    **simulation_kwargs
) -> Dict[str, np.ndarray]:
    """
    Generate a dataset of N-body trajectories.
    
    Args:
        num_trajectories: Number of trajectories to generate
        num_particles: Number of particles per trajectory
        total_time: Total simulation time per trajectory
        dt: Time step
        integrator: Integration method
        save_path: Optional path to save dataset
        **simulation_kwargs: Additional simulation parameters
        
    Returns:
        Dictionary containing positions, velocities, masses, etc.
    """
    # TODO: Implement dataset generation
    raise NotImplementedError("generate_dataset not implemented yet")


def save_dataset(data: Dict[str, np.ndarray], save_path: str):
    """
    Save dataset to disk.
    
    Args:
        data: Dataset dictionary
        save_path: Path to save directory
    """
    # TODO: Implement dataset saving
    raise NotImplementedError("save_dataset not implemented yet")


def load_dataset(load_path: str) -> Dict[str, np.ndarray]:
    """
    Load dataset from disk.
    
    Args:
        load_path: Path to dataset directory
        
    Returns:
        Dataset dictionary
    """
    # TODO: Implement dataset loading
    raise NotImplementedError("load_dataset not implemented yet")


