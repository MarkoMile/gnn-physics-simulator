# GNN Physics Simulator

A PyTorch Geometric implementation of [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405) (Sanchez-Gonzalez et al., 2020). Trains a Graph Neural Network to predict particle accelerations from local interactions, reproducing DeepMind's architecture for simulating fluids, rigid bodies, and gravitational systems.

## Project Structure

```
gnn-physics-simulator/
├── config.yaml              # Central configuration (model, training, data, simulation)
├── models.py                # GNNSimulator: Encoder → Processor → Decoder architecture
├── dataset.py               # ParticleDataset: TFRecord/NPZ loading, noise injection, graph building
├── train.py                 # Training loop: normalized loss, LR decay, checkpointing
├── simulation.py            # N-Body & Fluid (pseudo-SPH) generators with dataset export
├── integrators.py           # Euler, RK4 numerical integrators
├── scripts/
│   ├── generate_dataset.py  # CLI to generate synthetic datasets from config.yaml
│   └── download_dataset.py  # Download DeepMind datasets (WaterDrop, etc.)
├── utils/
│   ├── train_kaggle.ipynb           # Kaggle notebook: train on full WaterDrop dataset
│   └── train_generated_kaggle.ipynb # Kaggle notebook: generate + train synthetic data
├── Project.ipynb            # Project presentation notebook
├── TestSim.ipynb            # N-Body simulation testing and visualization
└── data/                    # Dataset directory (WaterDrop, WaterDropSample, generated)
```

## Installation

```bash
git clone https://github.com/MarkoMile/gnn-physics-simulator.git
cd gnn-physics-simulator

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

### Train on DeepMind's WaterDrop Dataset

```bash
# 1. Download the WaterDrop dataset
python scripts/download_dataset.py

# 2. Train the model
python train.py --config config.yaml

# With Weights & Biases logging:
python train.py --config config.yaml --wandb
```

### Generate and Train on Synthetic Data (WIP)

```bash
# Generate an N-Body or Fluid dataset from config.yaml
python scripts/generate_dataset.py --config config.yaml --output data/Generated

# Train on the generated data (set dataset_path and dataset_format in config.yaml)
python train.py --config config.yaml
```

### Train on Kaggle (GPU)

Upload `utils/train_kaggle.ipynb` to Kaggle for GPU-accelerated training on the full WaterDrop dataset, or `utils/train_generated_kaggle.ipynb` to generate and train on synthetic data.

## Model Architecture

The GNN simulator follows DeepMind's Encode-Process-Decode architecture:

1. **Encoder**: Embeds node features (velocity history, boundary distances, mass, particle type) and edge features (relative displacement, distance) into 128-D latent vectors via 3-layer MLPs with LayerNorm.
2. **Processor**: 10 sequential `InteractionNetwork` message passing blocks with residual connections on both nodes and edges, using sum aggregation.
3. **Decoder**: Projects latent node states to predicted accelerations (no LayerNorm).

**Important features matching the paper:**
- Dynamic online normalization of inputs and outputs
- Random-walk noise injection during training

## Configuration

All parameters are centralized in `config.yaml`:

See [config.yaml](config.yaml).

## Simulation Types (WIP)

The dataset generator supports two simulation types via `config.yaml`:

- **`n_body`**: Gravitational N-body simulation with softened Newtonian forces
- **`fluid`**: Pseudo-SPH fluid with downward gravity, inter-particle repulsion, and rigid bounding box walls

## References

- [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405) — Sanchez-Gonzalez et al., 2020
- [Interaction Networks for Learning about Objects, Relations and Physics](https://arxiv.org/abs/1612.00222)
- [Stanford CS224W: Simulating Complex Physics with Graph Networks](https://medium.com/stanford-cs224w/simulating-complex-physics-with-graph-networks-step-by-step-177354cb9b05)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)

## Notes

This project was developed as an optional project for the course Numerical Algorithms and Numerical Software (E231) at the Faculty of Technical Sciences, University of Novi Sad.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
