"""
Graph Neural Network Simulator.

This module implements the main GNN-based physics simulator,
following the Interaction Network / Graph Network architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class GNNSimulator(nn.Module):
    """
    Graph Neural Network for simulating particle dynamics.
    
    Architecture:
    1. Encoder: Embeds node and edge features
    2. Processor: Multiple message passing steps
    3. Decoder: Predicts accelerations/next state
    """
    
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int,
        num_message_passing_steps: int,
        output_dim: int,
        num_particle_types: int = 9,
        particle_emb_dim: int = 16
    ):
        """
        Initialize the GNN Simulator.
        
        Args:
            node_features: Input node feature dimension
            edge_features: Input edge feature dimension
            hidden_dim: Hidden layer dimension
            num_message_passing_steps: Number of message passing iterations
            output_dim: Output dimension (typically 3 for acceleration)
        """
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        
        # 1. Initialize Dynamic Normalizers
        # Node features (velocity history + mass, etc)
        self.node_normalizer = Normalizer(size=node_features)
        # Edge features (relative displacement vector + distance magnitude)
        self.edge_normalizer = Normalizer(size=edge_features)
        # Target acceleration normalizer (output)
        self.output_normalizer = Normalizer(size=output_dim)
        
        self.encoder_node = NodeEncoder(node_features, num_particle_types, particle_emb_dim, hidden_dim, hidden_dim)
        self.encoder_edge = EdgeEncoder(edge_features, hidden_dim, hidden_dim)
        
        self.processor = nn.ModuleList([
            InteractionNetwork(hidden_dim) for _ in range(num_message_passing_steps)
        ])
        
        self.decoder = Decoder(hidden_dim, hidden_dim, output_dim)
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GNNSimulator':
        """
        Instantiate the GNNSimulator dynamically from the loaded project config.yaml dictionary.
        Automatically mathematically derives the matrix sizing constraints from sequence lengths and dimensionality.
        """
        model_cfg = config['model']
        data_cfg = config['data']
        
        # Dimensions
        spatial_dim = data_cfg['spatial_dim']
        history_window = data_cfg['history_window']
        
        # Calculate dynamic tensor sizes!
        # Node features: (Velocity history frames * Dimension) + Scalar Mass
        node_features = (history_window * spatial_dim) + 1
        
        # Edge features: Relative directional displacement vector + Scalar absolute length
        edge_features = spatial_dim + 1
        
        return cls(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=model_cfg.get('hidden_dim', 128),
            num_message_passing_steps=model_cfg.get('message_passing_steps', 10),
            output_dim=spatial_dim,
            num_particle_types=9, # Default across DeepMind datasets
            particle_emb_dim=16
        )
    
    def forward(self, graph) -> torch.Tensor:
        """
        Forward pass through the simulator mapping states to output accelerations.
        """
        # 1. Normalize graph physical inputs dynamically
        normalized_nodes = self.node_normalizer(graph.x, accumulate=self.training)
        normalized_edges = self.edge_normalizer(graph.edge_attr, accumulate=self.training)
        
        # 2. Encode inputs to 128-D latents
        x = self.encoder_node(normalized_nodes, graph.particle_type)
        edge_attr = self.encoder_edge(normalized_edges)
        
        # 3. Processor: M message passing rounds
        for layer in self.processor:
            x, edge_attr = layer(x, graph.edge_index, edge_attr)
            
        # 4. Decode node latents to raw dimensionless predictions
        predicted_accelerations = self.decoder(x)
        
        # 5. Inverse normalize yielding physical m/s^2 matrices
        return self.output_normalizer.inverse(predicted_accelerations)

"""
Normalizer module.

This module implements a normalizer for graph neural networks.
"""

class Normalizer(nn.Module):
    """
    Dynamically tracks the mean and variance of node/edge features 
    as they are passed through the network during training, to normalize
    them without needing pre-computed dataset statistics.
    """
    def __init__(self, size: int, max_accumulations: int = 10**6, epsilon: float = 1e-8):
        super().__init__()
        self.max_accumulations = max_accumulations
        self.epsilon = epsilon
        
        # Buffer so they are saved to state_dict but not treated as trainable parameters
        self.register_buffer('acc_count', torch.tensor(0, dtype=torch.float32))
        self.register_buffer('num_accumulations', torch.tensor(0, dtype=torch.float32))
        self.register_buffer('acc_sum', torch.zeros(size, dtype=torch.float32))
        self.register_buffer('acc_sum_squared', torch.zeros(size, dtype=torch.float32))

    def forward(self, batched_data: torch.Tensor, accumulate: bool = True) -> torch.Tensor:
        """
        Normalizes the input and conditionally accumulates its statistics.
        """
        if accumulate and self.num_accumulations < self.max_accumulations:
            self._accumulate(batched_data)
            
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Un-normalizes the data (used for translating predicted accelerations back to physics).
        """
        return normalized_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data: torch.Tensor):
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, dim=0)
        data_sum_squared = torch.sum(batched_data**2, dim=0)
        
        self.acc_sum += data_sum
        self.acc_sum_squared += data_sum_squared
        self.acc_count += count
        self.num_accumulations += 1

    def _mean(self) -> torch.Tensor:
        safe_count = torch.clamp(self.acc_count, min=1.0)
        return self.acc_sum / safe_count

    def _std_with_epsilon(self) -> torch.Tensor:
        safe_count = torch.clamp(self.acc_count, min=1.0)
        variance = (self.acc_sum_squared / safe_count) - self._mean()**2
        # Ensure non-negative variance before square root
        safe_variance = torch.clamp(variance, min=0.0)
        return torch.sqrt(safe_variance + self.epsilon)



"""
Message Passing Neural Network layers.

This module implements various message passing layers
for graph neural networks.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from typing import Tuple


class InteractionNetwork(MessagePassing):
    """
    Interaction Network layer for physics simulation.
    Uses PyTorch Geometric's MessagePassing abstraction to compute forces seamlessly.
    """
    def __init__(self, hidden_dim: int):
        super().__init__(aggr='add') # Forces aggregate cumulatively at receiving nodes
        
        # Edge update network
        # Combines sender node, receiver node, and edge feature back into a latent edge
        self.edge_mlp = build_mlp(hidden_dim * 3, hidden_dim, hidden_dim, layernorm=True)
        
        # Node update network
        # Combines original node feature with aggregated incoming messages
        self.node_mlp = build_mlp(hidden_dim * 2, hidden_dim, hidden_dim, layernorm=True)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        senders, receivers = edge_index[0], edge_index[1]
        
        # 1. Evaluate Edge function
        # Concat [sending_node, receiving_node, edge_feature]
        edge_inputs = torch.cat([x[senders], x[receivers], edge_attr], dim=-1)
        updated_edges = self.edge_mlp(edge_inputs)
        
        # Apply residual connection on edges
        edge_attr = edge_attr + updated_edges
        
        # 2. Aggregate forces and Evaluate Node Function
        # We explicitly pass size=(N, N) so that even if a node has no incoming edges (e.g. isolated droplet),
        # its aggregated force matrix correctly initializes as all zeros instead of crashing the dimension scatter
        aggregated_messages = self.propagate(edge_index, size=(x.size(0), x.size(0)), updated_edges=edge_attr)
        
        node_inputs = torch.cat([x, aggregated_messages], dim=-1)
        updated_nodes = self.node_mlp(node_inputs)
        
        # Apply residual connection on nodes
        x = x + updated_nodes
        
        return x, edge_attr
        
    def message(self, updated_edges: torch.Tensor) -> torch.Tensor:
        """The message passed along each structural link is simply the updated edge attribute."""
        return updated_edges


"""
Encoder and Decoder networks.

This module implements encoder and decoder MLPs
for the GNN simulator.
"""

import torch
import torch.nn as nn


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, layernorm: bool = True) -> nn.Sequential:
    """
    Builds a standard Multi-Layer Perceptron (MLP) according to DeepMind's specifications.
    Specifically: Two hidden layers, ReLU activations, linear output, optional LayerNorm.
    """
    layers = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    ]
    
    if layernorm:
        layers.append(nn.LayerNorm(output_dim))
        
    return nn.Sequential(*layers)


class NodeEncoder(nn.Module):
    """
    Encoder for node features.
    
    Maps raw particle features (position, velocity, mass)
    to latent representations.
    """
    
    def __init__(self, node_feature_dim: int, num_particle_types: int, particle_emb_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize node encoder.
        
        Args:
            node_feature_dim: Input feature dimension (e.g. historical velocities + mass)
            num_particle_types: Total distinct physical materials (e.g. 9 for WaterDropSample)
            particle_emb_dim: Dimensionality for the learned material embedding (usually 16)
            hidden_dim: Hidden layer dimension
            output_dim: Output latent dimension (usually 128)
        """
        super().__init__()
        
        # DeepMind explicitly embeds the integer material types (Water vs Sand etc) 
        # into a 16-D continuous dense space so the model can learn material properties.
        self.particle_embedding = nn.Embedding(num_particle_types, particle_emb_dim)
        
        # The MLP will consume both the physical metric features and that mapped material space
        total_input_dim = node_feature_dim + particle_emb_dim
        self.mlp = build_mlp(total_input_dim, hidden_dim, output_dim, layernorm=True)
        
    def forward(self, node_features: torch.Tensor, particle_types: torch.Tensor) -> torch.Tensor:
        """Encode node features by combining physical arrays with material embeddings."""
        # shape [N] -> [N, Emb_Dim]
        type_embeddings = self.particle_embedding(particle_types)
        
        # Merge physical tensors with categorical tensors: shape [N, Features + Emb_Dim]
        x = torch.cat([node_features, type_embeddings], dim=-1)
        
        # Map through MLP into latent bounds [N, Output_Dim]
        return self.mlp(x)


class EdgeEncoder(nn.Module):
    """
    Encoder for edge features.
    
    Maps raw edge features (relative position, distance)
    to latent representations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Edges just evaluate the spatial displacement directly through an MLP
        self.mlp = build_mlp(input_dim, hidden_dim, output_dim, layernorm=True)
        
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """Map edge relative distances to a latent representation"""
        return self.mlp(edge_features)


class Decoder(nn.Module):
    """
    Decoder for predicting accelerations.
    
    Maps latent node representations to predicted accelerations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # The decoder takes the final processed node latents and maps them to physics vectors
        self.mlp = build_mlp(input_dim, hidden_dim, output_dim, layernorm=False)
        
    def forward(self, node_latents: torch.Tensor) -> torch.Tensor:
        """Map final node hidden representations into un-normalized physics vectors (accelerations)"""
        return self.mlp(node_latents)


