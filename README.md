# GNN Physics Simulator

Graph Neural Network (GNN) simulator for particle dynamics in PyTorch; Extension to generate a custom gravitational N-body dataset using classical numerical integrators.

Inspired by Stanford CS224W blog post: https://medium.com/stanford-cs224w/simulating-complex-physics-with-graph-networks-step-by-step-177354cb9b05

## Project Structure

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the complete file structure.

## Implementation Progress

### Phase 1: Data Generation

- [ ] Implement numerical integrators (Euler, RK4, Leapfrog, Verlet)
- [ ] Implement N-body gravitational simulation
- [ ] Create dataset generation script
- [ ] Generate training/validation/test datasets

### Phase 2: Data Processing

- [ ] Implement graph construction utilities
- [ ] Create PyTorch Dataset classes
- [ ] Implement data preprocessing pipeline
- [ ] Add data normalization/standardization

### Phase 3: Model Development

- [ ] Implement node/edge encoders
- [ ] Implement message passing layers
- [ ] Implement decoder network
- [ ] Build complete GNN simulator model

### Phase 4: Training

- [ ] Implement training loop
- [ ] Add loss functions (MSE, rollout loss)
- [ ] Implement learning rate schedulers
- [ ] Add checkpointing and logging

### Phase 5: Evaluation

- [ ] Implement evaluation metrics
- [ ] Add rollout evaluation
- [ ] Create visualization utilities
- [ ] Compare GNN vs numerical integrators

### Phase 6: Documentation & Polish

- [ ] Add Google Colab support
- [ ] Write unit tests
- [ ] Complete documentation
- [ ] Final experiments and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gnn-physics-simulator.git
cd gnn-physics-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Generate Dataset

Generate N-body simulation trajectories using numerical integrators:

```bash
python scripts/generate_dataset.py --config configs/default.yaml
```

### 2. Preprocess Data

Preprocess and split the data:

```bash
python scripts/preprocess_data.py --input data/raw --output data/processed
```

### 3. Train Model

Train the GNN simulator:

```bash
python scripts/train.py --config configs/default.yaml
```

### 4. Evaluate Model

Evaluate the trained model:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --test-data data/processed/test
```

## Numerical Integrators

The project implements the following numerical integrators for generating training data:

- **Euler**: First-order explicit method
- **RK4**: Fourth-order Runge-Kutta method
- **Leapfrog**: Symplectic integrator (Störmer-Verlet)
- **Verlet**: Velocity Verlet integrator

Compare integrators:

```bash
python scripts/compare_integrators.py --integrators euler rk4 leapfrog verlet
```

## Model Architecture

The GNN simulator follows the Interaction Network architecture:

1. **Encoder**: Embeds particle features (position, velocity, mass) to latent space
2. **Processor**: Multiple message passing steps to propagate information
3. **Decoder**: Predicts accelerations from latent node representations

## References

- [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405)
- [Interaction Networks for Learning about Objects, Relations and Physics](https://arxiv.org/abs/1612.00222)
- [Stanford CS224W Blog Post](https://medium.com/stanford-cs224w/simulating-complex-physics-with-graph-networks-step-by-step-177354cb9b05)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
