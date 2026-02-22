"""
Dataset classes for particle simulations.

This module implements PyTorch Dataset classes for loading
and processing particle trajectory data for GNN training.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
import os
import json
import numpy as np


class ParticleDataset(Dataset):
    """
    Dataset for general particle simulations (N-body, Fluid Dynamics, etc).
    
    Handles loading raw sequence data, performing dynamic random-walk noise 
    injection during training, and building the PyTorch tensors.
    """
    
    def __init__(self, data_path: str, split: str = 'train', dataset_format: str = 'npz', noise_std: float = 0.0003):
        """
        Initialize the dataset, load data, and apply dynamic training noise.
        """
        raw_data = load_raw_data(data_path, dataset_format=dataset_format)
        self.metadata = raw_data['metadata']
        self.trajectories = raw_data['splits'].get(split, [])
        self.is_training = (split == 'train')
        self.noise_std = noise_std
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """
        Retrieves a trajectory, dynamically deriving input nodes and target accelerations.
        Injects a random-walk noise pattern on inputs if training.
        """
        traj = self.trajectories[idx]
        pos = traj['position']         # [T, N, D]
        vel = traj['velocity']         # [T, N, D]
        particle_type = traj['particle_type'] # [N]
        mass = traj['mass']            # [N]
        
        # Deepmind models condition on previous C=5 velocities
        history_window = 5
        
        # Randomly select a valid start timestep 
        T = pos.shape[0]
        t = np.random.randint(history_window, T - 1)
        
        # Slice the input feature history
        input_vel = vel[t - history_window:t].copy()  # [C, N, D]
        input_pos = pos[t - 1].copy()                 # [N, D] (most recent position)
        
        target_acc = vel[t] - vel[t-1]
        
        if self.is_training and self.noise_std > 0.0:
            # 1. Generate independent noise for each step in history
            noise = np.random.normal(loc=0.0, scale=self.noise_std, size=input_vel.shape).astype(np.float32)
            
            # 2. Accumulate it as a random walk across the history sequence
            acc_noise = np.cumsum(noise, axis=0)
            
            # 3. Add accumulated noise to the input velocities
            input_vel += acc_noise
            
            # 4. Correct the position to match the shifted final velocity
            input_pos += acc_noise[-1]
            
            # 5. Adjust target acceleration to undo the injected noise!
            target_acc -= acc_noise[-1]
        
        # Flatten history velocities [N, C*D] for MLP input
        N, D = input_pos.shape
        # Move N to front, then flatten C and D
        input_vel_flat = input_vel.transpose(1, 0, 2).reshape(N, history_window * D)
        
        return {
            'pos': torch.from_numpy(input_pos),
            'vel_history': torch.from_numpy(input_vel_flat),
            'particle_type': torch.from_numpy(particle_type),
            'mass': torch.from_numpy(mass),
            'target_acc': torch.from_numpy(target_acc)
        }


"""
Data preprocessing utilities.

This module contains functions for:
- Loading raw simulation data
- Normalizing particle states
- Computing graph connectivity
- Train/validation/test splitting
"""


def load_raw_data(data_path: str, dataset_format: str = "npz") -> Dict[str, Any]:
    """
    Load raw simulation data from files, supporting either native NPZ or DeepMind TFRecord formats.
    
    Args:
        data_path: Path to raw data directory
        dataset_format: Format to load ('npz' or 'tfrecord')
        
    Returns:
        Dictionary containing metadata and dataset splits ('train', 'valid', 'test').
        Each split contains a list of trajectory dictionaries.
    """
    dataset = {'splits': {}}
    
    # 1. Load Metadata
    meta_path = os.path.join(data_path, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found in {data_path}")
        
    with open(meta_path, "r") as f:
        dataset['metadata'] = json.load(f)
        
    # 2. Route loader based on format requested
    if dataset_format == "npz":
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(data_path, f"{split}.npz")
            if os.path.exists(split_path):
                data = np.load(split_path, allow_pickle=True)
                # Convert the NPZ dictionary keys back into a list of trajectory dicts
                dataset['splits'][split] = [data[k].item() for k in data.files]
            else:
                dataset['splits'][split] = []
                
    elif dataset_format == "tfrecord":
        dataset['splits'] = _parse_tfrecords(data_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}. Use 'npz' or 'tfrecord'.")
        
    return dataset


def _parse_tfrecords(data_path: str) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """
    Dynamically loads TensorFlow to parse DeepMind TFRecord `SequenceExample` protobufs,
    converting them firmly into pure NumPy dictionaries matching the .npz structure.
    """
    try:
        import tensorflow as tf
        # Turn off TF noisy logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        raise ImportError(
            "TensorFlow is required to parse .tfrecord files. "
            "Please install it via `pip install tensorflow` or switch to the 'npz' dataset format."
        )

    splits = {}
    
    # Matching deepmind 'reading_utils.py' definitions
    feature_description = {
        'position': tf.io.VarLenFeature(tf.string),
    }
    context_features = {
        'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'particle_type': tf.io.VarLenFeature(tf.string)
    }
    
    # Retrieve dim and sequence_length from metadata to shape the arrays correctly
    meta_path = os.path.join(data_path, "metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    seq_len = metadata['sequence_length']
    dim = metadata['dim']

    for split_name in ['train', 'valid', 'test']:
        tfrecord_path = os.path.join(data_path, f"{split_name}.tfrecord")
        trajectories = []
        
        if os.path.exists(tfrecord_path):
            raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
            
            for serialized_example in raw_dataset:
                # Parse sequence example
                context, sequence = tf.io.parse_single_sequence_example(
                    serialized_example,
                    context_features=context_features,
                    sequence_features=feature_description
                )
                
                # Extract and cast particle_type from context
                p_type_bytes = context['particle_type'].values.numpy()
                particle_type = np.frombuffer(p_type_bytes[0], dtype=np.int64)
                
                # Extract and cast positions from sequence
                pos_bytes = sequence['position'].values.numpy()
                # TF stores the position frames as a list of bytes. 
                # According to deepmind reading_utils, sequence_length + 1 frames present
                positions = []
                for step_bytes in pos_bytes:
                    step_pos = np.frombuffer(step_bytes, dtype=np.float32)
                    positions.append(step_pos)
                
                # Reshape positions to [sequence_length + 1, num_particles, dim]
                # Note: Deepmind's data often includes a 0th frame for velocity derivation.
                positions = np.array(positions)
                positions = positions.reshape(seq_len + 1, -1, dim)
                
                # Reconstruct full trajectory dictionary mapping to our native NPZ schema!
                # DeepMind datasets don't explicitly store velocity or mass, so we
                # derive velocity via backward difference, and set mass to 1.0 globally.
                velocities = positions[1:] - positions[:-1]
                
                # Clip the position 0th element to sync shapes [T, N, D]
                synced_positions = positions[1:]
                
                num_particles = synced_positions.shape[1]
                masses = np.ones(num_particles, dtype=np.float32)
                
                traj_dict = {
                    'particle_type': particle_type,
                    'position': synced_positions,
                    'velocity': velocities,
                    'mass': masses
                }
                trajectories.append(traj_dict)
                
        splits[split_name] = trajectories
        
    return splits



def compute_connectivity(positions, connectivity_radius: float):
    """
    Compute graph connectivity based on particle positions.
    
    Args:
        positions: Particle positions [N, 3]
        connectivity_radius: Maximum distance for edge creation
        
    Returns:
        Edge index tensor [2, E]
    """
    # TODO: Implement connectivity computation
    raise NotImplementedError("compute_connectivity not implemented yet")


"""
Graph construction utilities.

This module provides functions for converting particle states
to graph representations suitable for GNN processing.
"""

import torch


def build_graph(positions, velocities, masses=None, connectivity_radius=None):
    """
    Build a graph from particle states.
    
    Args:
        positions: Particle positions [N, 3]
        velocities: Particle velocities [N, 3]
        masses: Optional particle masses [N, 1]
        connectivity_radius: Radius for edge creation (None = fully connected)
        
    Returns:
        Graph data structure with nodes, edges, and features
    """
    # TODO: Implement graph construction
    raise NotImplementedError("build_graph not implemented yet")


def compute_edge_features(positions, edge_index):
    """
    Compute edge features from particle positions.
    
    Args:
        positions: Particle positions [N, 3]
        edge_index: Edge connectivity [2, E]
        
    Returns:
        Edge features [E, F_e]
    """
    # TODO: Implement edge feature computation
    raise NotImplementedError("compute_edge_features not implemented yet")


def batch_graphs(graphs):
    """
    Batch multiple graphs for parallel processing.
    
    Args:
        graphs: List of graph data structures
        
    Returns:
        Batched graph
    """
    # TODO: Implement graph batching
    raise NotImplementedError("batch_graphs not implemented yet")


