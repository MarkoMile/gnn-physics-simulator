from integrators import get_integrator

"""
N-body gravitational simulation.

This module implements the N-body problem for generating
training data for the GNN simulator.
"""

import numpy as np
import os
import json
from typing import Tuple, Optional, Dict, Any


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

        #Initialize to 2D
        self.positions[:, 2] = 0.0
        self.velocities[:, 2] = 0.0
    
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
) -> Dict[str, Any]:
    """
    Generate a dataset of N-body trajectories matching DeepMind's sequence structure.
    
    Args:
        num_trajectories: Number of trajectories to generate
        num_particles: Number of particles per trajectory
        total_time: Total simulation time per trajectory
        dt: Time step
        integrator: Integration method
        save_path: Optional path to save dataset
        **simulation_kwargs: Additional simulation parameters
        
    Returns:
        Dictionary containing trajectories with positions, types, masses, etc.
    """
    from tqdm import tqdm
    
    gravitational_constant = simulation_kwargs.get('gravitational_constant', 1.0)
    softening_length = simulation_kwargs.get('softening_length', 0.01)
    
    position_scale = simulation_kwargs.get('position_scale', 1.0)
    velocity_scale = simulation_kwargs.get('velocity_scale', 0.1)
    mass_range = simulation_kwargs.get('mass_range', (0.5, 1.5))
    save_every = simulation_kwargs.get('save_every', 10)
    
    trajectories = []
    
    for _ in tqdm(range(num_trajectories), desc="Generating Trajectories"):
        sim = NBodySimulation(
            num_particles=num_particles,
            gravitational_constant=gravitational_constant,
            softening_length=softening_length,
            integrator=integrator
        )
        sim.initialize_random_state(
            position_scale=position_scale,
            velocity_scale=velocity_scale,
            mass_range=mass_range
        )
        
        history_positions, history_velocities, _ = sim.simulate(
            total_time=total_time,
            dt=dt,
            save_every=save_every
        )
        
        # Save trajectory in a format mimicking DeepMind's TFRecords
        traj_data = {
            'particle_type': np.zeros(num_particles, dtype=np.int64),  # Type 0 for all gravitational particles
            'position': history_positions.astype(np.float32),
            'velocity': history_velocities.astype(np.float32),
            'mass': sim.masses.astype(np.float32)
        }
        trajectories.append(traj_data)
        
    dataset = {'trajectories': trajectories}
    
    if save_path:
        save_dataset(dataset, save_path, dt * save_every)
        
    return dataset


def save_dataset(dataset: Dict[str, Any], save_path: str, save_dt: float):
    """
    Save dataset to disk in NPZ format along with metadata.json.
    
    Args:
        dataset: Dataset dictionary containing list of trajectories
        save_path: Path to save directory
        save_dt: The time delta between saved frames (dt * save_every)
    """
    os.makedirs(save_path, exist_ok=True)
    
    all_vels = []
    all_accs = []
    
    trajectories = dataset['trajectories']
    dim = trajectories[0]['position'].shape[-1]
    
    # Compute global kinematic stats across the dataset for the metadata.json
    for traj in trajectories:
        pos = traj['position']
        # Compute derived velocities and accelerations for normalization (like DeepMind)
        vel = pos[1:] - pos[:-1]
        acc = vel[1:] - vel[:-1]
        
        all_vels.append(vel.reshape(-1, dim))
        all_accs.append(acc.reshape(-1, dim))
        
    all_vels = np.concatenate(all_vels, axis=0)
    all_accs = np.concatenate(all_accs, axis=0)
    
    metadata = {
        "bounds": [[-5.0, 5.0] for _ in range(dim)], 
        "sequence_length": trajectories[0]['position'].shape[0],
        "default_connectivity_radius": 1.0,
        "dim": dim,
        "dt": save_dt,
        "vel_mean": all_vels.mean(axis=0).tolist(),
        "vel_std": all_vels.std(axis=0).tolist(),
        "acc_mean": all_accs.mean(axis=0).tolist(),
        "acc_std": all_accs.std(axis=0).tolist()
    }
    
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Split trajectories
    num_traj = len(trajectories)
    train_idx = int(0.8 * num_traj)
    valid_idx = int(0.9 * num_traj)
    
    splits = {
        'train': trajectories[:train_idx],
        'valid': trajectories[train_idx:valid_idx],
        'test': trajectories[valid_idx:]
    }
    
    for split_name, split_trajs in splits.items():
        if len(split_trajs) > 0:
            # We compress them as a sequence of dicts to a single .npz
            save_dict = {f"trajectory_{i}": traj for i, traj in enumerate(split_trajs)}
            np.savez_compressed(os.path.join(save_path, f"{split_name}.npz"), **save_dict)


def load_dataset(load_path: str) -> Dict[str, Any]:
    """
    Load dataset from disk.
    
    Args:
        load_path: Path to dataset directory
        
    Returns:
        Dataset dictionary with splits and metadata
    """
    dataset = {}
    with open(os.path.join(load_path, "metadata.json"), "r") as f:
        dataset['metadata'] = json.load(f)
        
    dataset['splits'] = {}
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(load_path, f"{split}.npz")
        if os.path.exists(split_path):
            data = np.load(split_path, allow_pickle=True)
            dataset['splits'][split] = [data[k].item() for k in data.files]
            
    return dataset


