# GNN Physics Simulator

Graph Neural Network (GNN) simulator for particle dynamics in PyTorch; Extension to generate a custom gravitational N-body dataset using classical numerical integrators.

Inspired by Stanford CS224W blog post: https://medium.com/stanford-cs224w/simulating-complex-physics-with-graph-networks-step-by-step-177354cb9b05

## Project Structure

A minimal project layout featuring two primary notebooks using independent Python modules.

```
gnn-physics-simulator/
├── Project.ipynb       # Main project notebook (theory, model definition, training)
├── TestSim.ipynb       # Testing/visualization for the N-Body numerical simulation
├── simulation.py       # Core N-Body generator
├── integrators.py      # Euler, RK4, Leapfrog, and Verlet integrators
├── models.py           # PyTorch GNN components
├── dataset.py          # PyTorch Geometric dataset generation
└── utils.py            # Graph construction and helper methods
```

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
```

## Quick Start
Open `Project.ipynb` in your IDE or Jupyter interface to explore the project presentation, execute dataset generation, and train or load the primary GNN model. Alternatively, exploring the `TestSim.ipynb` notebook provides isolated visualization boundaries for tracking energy conservation of the ground-truth integration methods.

## Numerical Integrators

The project implements the following numerical integrators for generating training data:

- **Euler**: First-order explicit method
- **RK4**: Fourth-order Runge-Kutta method
- **Leapfrog**: Symplectic integrator (Störmer-Verlet) (TODO: implement)
- **Verlet**: Velocity Verlet integrator (TODO: implement)


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

## Notes

This project was done as an optional project for the course Numerical Algorithms and Numerical Software (E231) at the Faculty of Technical Sciences, University of Novi Sad.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
